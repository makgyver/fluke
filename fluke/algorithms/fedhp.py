"""Implementation of the FedHP: Federated Learning with Hyperspherical Prototypical Regularization
[FedHP24]_ algorithm.

References:
    .. [FedHP24] Samuele Fonio, Mirko Polato, Roberto Esposito.
       FedHP: Federated Learning with Hyperspherical Prototypical Regularization. In ESANN (2024).
       URL: https://www.esann.org/sites/default/files/proceedings/2024/ES2024-183.pdf
"""
import copy
import sys
from typing import Collection, Generator, Literal

import torch
import torch.optim as optim
from rich.progress import track
from torch import nn
from torch.optim.optimizer import Optimizer as Optimizer

sys.path.append(".")
sys.path.append("..")
from .. import FlukeENV  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..server import Server  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from ..utils.model import get_activation_size, unwrap  # NOQA
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

    def __init__(self, reduction: Literal["mean", "sum"] = "mean"):
        super(SeparationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, protos: torch.Tensor) -> torch.Tensor:
        # protos: (N_prototypes x Embedding_dimension)
        M = torch.matmul(protos, protos.transpose(0, 1)) - 2 * torch.eye(
            protos.shape[0]).to(protos.device)
        loss = M.max(dim=1)[0]
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FedHPClient(Client):
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
                 fine_tuning_epochs: int = 0,
                 clipping: float = 0,
                 **kwargs):
        super().__init__(index=index,
                         train_set=train_set, test_set=test_set, optimizer_cfg=optimizer_cfg,
                         loss_fn=loss_fn, local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, clipping=clipping, **kwargs)
        embedding_size = get_activation_size(model, train_set.tensors[0][0])
        self.model = ProtoNet(model, n_protos, embedding_size)
        self.hyper_params.update(n_protos=n_protos, lam=lam)
        self.anchors = None
        # This optimizer must be saved in the cache along with the model from which its parameters
        # are taken. See the property `proto_opt`.
        self._modopt.additional = {"proto_opt": None}
        self._attr_to_cache.append("anchors")
        self._save_to_cache()

    @property
    def proto_opt(self) -> Optimizer:
        """The optimizer for the prototypes.

        Returns:
            Optimizer: The optimizer for the prototypes.
        """
        return self._modopt.additional["proto_opt"]

    @proto_opt.setter
    def proto_opt(self, value: Optimizer) -> None:
        self._modopt.additional["proto_opt"] = value

    def receive_model(self) -> None:
        if self.anchors is None:
            msg = self.channel.receive(self.index, "server", msg_type="anchors")
            self.anchors = msg.payload.data
        msg = self.channel.receive(self.index, "server", msg_type="prototypes")
        self.model.prototypes.data = msg.payload

    def send_model(self) -> None:
        self.channel.send(Message(copy.deepcopy(self.model.prototypes.data),
                          "prototypes", self.index, inmemory=True), "server")

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (override_local_epochs if override_local_epochs > 0
                       else self.hyper_params.local_epochs)
        self.model.train()
        self.model.to(self.device)

        def filter_fun(model): return [param for name, param in model.named_parameters()
                                       if 'prototype' not in name]

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(
                self.model, filter_fun=filter_fun)

        if self.proto_opt is None:
            proto_params = [p for name, p in self.model.named_parameters()
                            if 'proto' in name]
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
                                        (unwrap(self.model).prototypes, self.anchors))
                loss += self.hyper_params.lam * loss_proto
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                self.proto_opt.step()
                running_loss += loss.item()
            self.scheduler.step()
        running_loss /= (epochs * len(self.train_set))
        self.model.cpu()
        clear_cuda_cache()
        return running_loss

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if test_set is not None and self.anchors is not None:
            model = FedHPModel(self.model)
            return evaluator.evaluate(self._last_round, model, test_set, device=self.device)
        return {}

    def finalize(self) -> None:
        self._load_from_cache()
        self.receive_model()
        self.fit(self.hyper_params.fine_tuning_epochs)
        metrics = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
        if metrics:
            self.notify(event="client_evaluation", round=-1,
                        client_id=self.index, phase="post-fit", evals=metrics)
        self._save_to_cache()


class FedHPServer(Server):
    def __init__(self,
                 model: nn.Module,
                 test_set: FastDataLoader,
                 clients: Collection[Client],
                 weighted: bool = True,
                 n_protos: int = 10,
                 embedding_size: int = 100,
                 **kwargs):
        super().__init__(model=ProtoNet(model, n_protos, embedding_size),
                         test_set=None,
                         clients=clients,
                         weighted=weighted)
        self.hyper_params.update(n_protos=n_protos,
                                 embedding_size=embedding_size)
        self.device = FlukeENV().get_device()
        self.anchors = None
        self.prototypes = None
        self.clients_class_weights = None

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1, finalize: bool = True) -> None:
        if self.rounds == 0:
            self.anchors = self._hyperspherical_embedding().data
            self.channel.broadcast(Message(self.anchors, "anchors", "server"),
                                   [c.index for c in self.clients])
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

    def broadcast_model(self, eligible: Collection[FedHPClient]) -> None:
        self.channel.broadcast(Message(self.prototypes, "prototypes", "server"),
                               [c.index for c in eligible])

    def receive_client_models(self,
                              eligible: Collection[Client],
                              state_dict: bool = False) -> Generator[nn.Module, None, None]:
        for client in eligible:
            yield self.channel.receive("server", client.index, "prototypes").payload

    def aggregate(self,
                  eligible: Collection[FedHPClient],
                  client_models: Collection[nn.Module]) -> None:
        clients_prototypes = client_models
        clients_weights = self.clients_class_weights[:, [client.index for client in eligible]].T
        avg_proto = None

        for i, client_proto in enumerate(clients_prototypes):
            if avg_proto is None:
                avg_proto = torch.zeros_like(client_proto)
            avg_proto += client_proto * clients_weights[i, :].unsqueeze(-1)
        avg_proto /= clients_weights.sum(dim=0).unsqueeze(-1)
        self.prototypes = avg_proto


class FedHP(PersonalizedFL):

    def get_client_class(self) -> type[Client]:
        return FedHPClient

    def get_server_class(self) -> type[Server]:
        return FedHPServer

    def get_optimizer_class(self) -> type[Optimizer]:
        return torch.optim.Adam
