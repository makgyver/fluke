from copy import deepcopy
from typing import Sequence, Callable, Dict
import torch
from torch import nn
from rich.progress import track
import sys

sys.path.append(".")
sys.path.append("..")

from .. import GlobalSettings  # NOQA
from ..server import Server  # NOQA
from ..client import PFLClient  # NOQA
from ..data import FastTensorDataLoader  # NOQA
from ..comm import Message  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from . import PersonalizedFL  # NOQA
from ..evaluation import ClassificationEval  # NOQA


class ProtoNet(nn.Module):

    def __init__(self, encoder: nn.Module, n_protos: int):
        super(ProtoNet, self).__init__()
        self._encoder = encoder
        self.prototypes = nn.Parameter(torch.rand((n_protos, encoder.output_size)),
                                       requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self._encoder(x)
        dists = -torch.norm(embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1)
        return embeddings, dists


class FedHyperProtoModel(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 device: torch.device):
        super().__init__()
        self.model: nn.Module = model
        self.device: torch.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.model(x)[1], dim=1)


class SeparationLoss(nn.Module):
    """Large margin separation between hyperspherical protoypes"""

    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, protos: torch.Tensor):
        """
        Args:
            protos (torch.Tensor): (N_prototypes x Embedding_dimension)
        """
        M = torch.matmul(protos, protos.transpose(0, 1)) - 2 * torch.eye(
            protos.shape[0]).to(protos.device)
        return M.max(dim=1)[0].mean()


class FedHyperProtoClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: nn.Module,
                 train_set: FastTensorDataLoader,
                 test_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 n_protos: int,
                 lam: float):
        super().__init__(index, ProtoNet(model, n_protos), train_set,
                         test_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update(n_protos=n_protos, lam=lam)
        self.model = self.personalized_model
        self.initial_prototypes = None

    def _receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.prototypes.data = deepcopy(msg.payload)

    def _receive_protos(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="protos")
        self.initial_prototypes = deepcopy(msg.payload)

    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.prototypes.data), "model", self), self.server)

    def fit(self, override_local_epochs: int = 0) -> None:
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)
        if self.initial_prototypes is None:
            self._receive_protos()
        self._receive_model()
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                _, dists = self.model.forward(X)
                loss = self.hyper_params.loss_fn(dists, y)
                loss_proto = torch.mean(1 - nn.CosineSimilarity(dim=1)
                                        (self.model.prototypes, self.initial_prototypes))
                loss += self.hyper_params.lam * loss_proto
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.model.to("cpu")
        clear_cache()
        self._send_model()

    def evaluate(self) -> Dict[str, float]:
        if self.test_set is not None:
            if self.initial_prototypes is None:
                self._receive_protos()
                self.channel.send(Message(self.server.prototypes, "model", self.server), self)
                self._receive_model()

            model = FedHyperProtoModel(self.model, self.device)
            return ClassificationEval(None,
                                      self.hyper_params.n_protos).evaluate(model,
                                                                           self.test_set)
        return {}


class FedHyperProtoServer(Server):

    def __init__(self,
                 model: nn.Module,
                 test_data: FastTensorDataLoader,
                 clients: Sequence[PFLClient],
                 eval_every: int = 1,
                 weighted: bool = True,
                 n_protos: int = 10,
                 embedding_size: int = 100):
        super().__init__(None, None, clients, eval_every, weighted)
        self.hyper_params.update(n_protos=n_protos,
                                 embedding_size=embedding_size)
        self.device = GlobalSettings().get_device()
        self.prototypes = None

    def _broadcast_model(self, eligible: Sequence[PFLClient]) -> None:
        # This funciton broadcasts the prototypes to the clients
        self.channel.broadcast(Message(self.prototypes, "model", self), eligible)

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1) -> None:
        self.prototypes = self._hyperspherical_embedding()
        self.channel.broadcast(Message(self.prototypes, "protos", self), self.clients)
        return super().fit(n_rounds, eligible_perc)

    def _hyperspherical_embedding(self, seed: int = 0):
        """
        Function to learn the prototypes according to the separationLoss Minimization
        embedding_dimension
        """
        lr = 0.1
        momentum = 0.9
        n_steps = 1000
        wd = 1e-4
        torch.manual_seed(seed)
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

    def _get_client_models(self, eligible: Sequence[PFLClient], state_dict: bool = False):
        return [self.channel.receive(self, client, "model").payload for client in eligible]

    def _aggregate(self, eligible: Sequence[PFLClient]) -> None:
        clients_protos = self._get_client_models(eligible)

        avg_proto = torch.zeros_like(clients_protos[0])
        for protos in clients_protos:
            avg_proto += protos
        avg_proto /= len(clients_protos)

        self.prototypes = avg_proto


class FedHyperProto(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedHyperProtoClient

    def get_server_class(self) -> Server:
        return FedHyperProtoServer
