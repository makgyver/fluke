"""Implementation of the FedHP: Federated Learning with Hyperspherical Prototypical Regularization
[FedHP24]_ algorithm.
References:
    .. [FedHP24] Samuele Fonio, Mirko Polato, Roberto Esposito.
       FedHP: Federated Learning with Hyperspherical Prototypical Regularization. In ESANN (2024).
       URL: https://www.esann.org/sites/default/files/proceedings/2024/ES2024-183.pdf
"""
import copy
import sys
from typing import Any, Iterable

import torch
import torch.optim as optim
from rich.progress import track
from torch import nn
from torch.optim.optimizer import Optimizer as Optimizer

sys.path.append(".")
sys.path.append("..")
from .. import GlobalSettings  # NOQA
from ..client import Client, PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import get_activation_size  # NOQA
from . import PersonalizedFL  # NOQA


class ProtoNet(nn.Module):
    def __init__(self, encoder: nn.Module, n_protos: int, proto_size: int):
        super(ProtoNet, self).__init__()
        self._encoder = encoder
        self.prototypes = nn.Parameter(torch.rand((n_protos, proto_size)),
                                       requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self._encoder(x)
        dists = -torch.norm(embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1)
        return embeddings, dists


class FedHPModel(nn.Module):
    def __init__(self,
                 model: nn.Module):
        super().__init__()
        self.model: nn.Module = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.model(x)[1], dim=1)


class SeparationLoss(nn.Module):
    """Large margin separation between hyperspherical protoypes"""

    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, protos: torch.Tensor) -> torch.Tensor:
        # protos: (N_prototypes x Embedding_dimension)
        M = torch.matmul(protos, protos.transpose(0, 1)) - 2 * torch.eye(
            protos.shape[0]).to(protos.device)
        return M.max(dim=1)[0].mean()


class FedHPClient(PFLClient):
    def __init__(self,
                 index: int,
                 model: nn.Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int,
                 n_protos: int,
                 lam: float,
                 **kwargs: dict[str, Any]):
        embedding_size = get_activation_size(model, train_set.tensors[0][0])
        super().__init__(index=index, model=ProtoNet(model, n_protos, embedding_size),
                         train_set=train_set, test_set=test_set, optimizer_cfg=optimizer_cfg,
                         loss_fn=loss_fn, local_epochs=local_epochs, **kwargs)
        self.hyper_params.update(n_protos=n_protos, lam=lam)
        self.model = self.personalized_model
        self.anchors = None
        self.proto_opt = None

    def receive_model(self) -> None:
        if self.anchors is None:
            msg = self.channel.receive(self, self.server, msg_type="anchors")
            self.anchors = msg.payload.data
        msg = self.channel.receive(self, self.server, msg_type="prototypes")
        self.model.prototypes.data = msg.payload

    def send_model(self) -> None:
        self.channel.send(Message(copy.deepcopy(self.model.prototypes.data),
                          "prototypes", self), self.server)

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)
        self.model.train()
        self.model.to(self.device)

        def filter_fun(model): return [param for name, param in model.named_parameters()
                                       if 'prototype' not in name]

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model, filter_fun=filter_fun)

        if self.proto_opt is None:
            proto_params = [p for name, p in self.model.named_parameters() if 'proto' in name]
            self.proto_opt = optim.Adam(proto_params, lr=0.005)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                self.proto_opt.zero_grad()
                _, dists = self.model.forward(X)
                loss = self.hyper_params.loss_fn(dists, y)
                loss_proto = torch.mean(1 - nn.CosineSimilarity(dim=1)
                                        (self.model.prototypes, self.anchors))
                loss += self.hyper_params.lam * loss_proto
                loss.backward()
                self.optimizer.step()
                self.proto_opt.step()
                running_loss += loss.item()
            self.scheduler.step()
        running_loss /= (epochs * len(self.train_set))
        self.model.to("cpu")
        clear_cache()
        return running_loss

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if test_set is not None and self.anchors is not None:
            model = FedHPModel(self.model)
            return evaluator.evaluate(self._last_round, model, test_set, device=self.device)
        return {}

    def finalize(self) -> None:
        self.fit()
        metrics = self.evaluate(GlobalSettings().get_evaluator(), self.test_set)
        if metrics:
            self._notify_evaluation(-1, "post-fit", metrics)


class FedHPServer(Server):
    def __init__(self,
                 model: nn.Module,
                 test_set: FastDataLoader,
                 clients: Iterable[PFLClient],
                 weighted: bool = True,
                 n_protos: int = 10,
                 embedding_size: int = 100,
                 **kwargs: dict[str, Any]):
        super().__init__(model=ProtoNet(model, n_protos, embedding_size),
                         test_set=None,
                         clients=clients,
                         weighted=weighted)
        self.hyper_params.update(n_protos=n_protos,
                                 embedding_size=embedding_size)
        self.device = GlobalSettings().get_device()
        self.anchors = None
        self.prototypes = None
        self.clients_class_weights = None

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1, finalize: bool = True) -> None:
        if self.rounds == 0:
            self.anchors = self._hyperspherical_embedding().data
            self.channel.broadcast(Message(self.anchors, "anchors", self), self.clients)
            self.prototypes = copy.deepcopy(self.anchors)
            client = {c.index: c.train_set.tensors[1] for c in self.clients}
            # Count the occurrences of each class for each client.
            # This is "illegal" in fluke :)
            n_classes = self.clients[0].train_set.num_labels
            class_counts = {client_idx: torch.bincount(client_data, minlength=n_classes).tolist()
                            for client_idx, client_data in enumerate(client.values())}
            if self.hyper_params.weighted:
                tensor_class_counts = torch.empty((len(class_counts[0]), self.n_clients))
                for ind, val in enumerate(class_counts.values()):
                    tensor_class_counts[:, ind] = torch.tensor(val)
                col_sums = tensor_class_counts.sum(dim=0, keepdim=True)
                self.clients_class_weights = tensor_class_counts / col_sums
            else:
                self.clients_class_weights = torch.ones((len(class_counts[0]), self.n_clients))

        return super().fit(n_rounds=n_rounds, eligible_perc=eligible_perc, finalize=finalize)

    def _hyperspherical_embedding(self):
        """
        Function to learn the prototypes according to the ``SeparationLoss`` minimization.
        """
        lr = 0.1
        momentum = 0.9
        n_steps = 1000
        wd = 1e-4
        # torch.manual_seed(seed)
        mapping = torch.rand((self.hyper_params.n_protos, self.hyper_params.embedding_size),
                             device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([mapping], lr=lr, momentum=momentum, weight_decay=wd)
        loss_fn = SeparationLoss()
        for _ in track(range(n_steps), "[SERVER] Learning prototypes..."):
            with torch.no_grad():
                mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
            optimizer.zero_grad()
            loss = loss_fn(mapping)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
        return mapping.detach()

    def broadcast_model(self, eligible: Iterable[FedHPClient]) -> None:
        self.channel.broadcast(Message(self.prototypes, "prototypes", self), eligible)

    def get_client_models(self, eligible: Iterable[Client], state_dict: bool = False) -> list[Any]:
        return [self.channel.receive(self, client, "prototypes").payload for client in eligible]

    def aggregate(self, eligible: Iterable[FedHPClient]) -> None:
        clients_prototypes = self.get_client_models(eligible)
        clients_weights = self.clients_class_weights[:, [client.index for client in eligible]].T
        avg_proto = torch.zeros_like(clients_prototypes[0])

        for i in range(len(clients_prototypes)):
            avg_proto += clients_prototypes[i] * clients_weights[i, :].unsqueeze(-1)
        avg_proto /= clients_weights.sum(dim=0).unsqueeze(-1)
        self.prototypes = avg_proto


class FedHP(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedHPClient

    def get_server_class(self) -> Server:
        return FedHPServer

    def get_optimizer_class(self) -> Optimizer:
        return torch.optim.Adam
