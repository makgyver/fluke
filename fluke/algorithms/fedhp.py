"""Implementation of the FedHP: Federated Learning with Hyperspherical Prototypical Regularization
[FedHP24]_ algorithm.

References:
    .. [FedHP24] Samuele Fonio, Mirko Polato, Roberto Esposito.
       FedHP: Federated Learning with Hyperspherical Prototypical Regularization. In ESANN (2024)
"""
import sys
from typing import Any, Iterable

import torch
from rich.progress import track
from torch import nn
from torch.optim.optimizer import Optimizer as Optimizer

sys.path.append(".")
sys.path.append("..")

from .. import GlobalSettings  # NOQA
from ..client import PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from . import PersonalizedFL  # NOQA


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
        super().__init__(index=index, model=ProtoNet(model, n_protos), train_set=train_set,
                         test_set=test_set, optimizer_cfg=optimizer_cfg, loss_fn=loss_fn,
                         local_epochs=local_epochs, **kwargs)
        self.hyper_params.update(n_protos=n_protos, lam=lam)
        self.model = self.personalized_model
        self.initial_prototypes = None

    def _receive_protos(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="protos")
        self.initial_prototypes = msg.payload

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)
        if self.initial_prototypes is None:
            self._receive_protos()

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
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
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.model.to("cpu")
        clear_cache()
        return running_loss

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if test_set is not None and self.initial_prototypes is not None:
            model = FedHPModel(self.model)
            return evaluator.evaluate(self._last_round, model, test_set)
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
        super().__init__(model=ProtoNet(model, n_protos),
                         test_set=None,
                         clients=clients,
                         weighted=weighted)
        self.hyper_params.update(n_protos=n_protos,
                                 embedding_size=embedding_size)
        self.device = GlobalSettings().get_device()
        # self.prototypes = None

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1, finalize: bool = True):
        if self.rounds == 0:
            self.model.prototypes.data = self._hyperspherical_embedding()
            self.channel.broadcast(Message(self.model.prototypes.data,
                                   "protos", self), self.clients)
        return super().fit(n_rounds=n_rounds, eligible_perc=eligible_perc, finalize=finalize)

    def _hyperspherical_embedding(self, seed: int = 0):
        """
        Function to learn the prototypes according to the ``SeparationLoss`` minimization.
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

    # def _get_client_models(self, eligible: Iterable[PFLClient], state_dict: bool = False):
    #     return [self.channel.receive(self, client, "model").payload for client in eligible]


class FedHP(PersonalizedFL):

    def can_override_optimizer(self) -> bool:
        return False

    def get_client_class(self) -> PFLClient:
        return FedHPClient

    def get_server_class(self) -> Server:
        return FedHPServer

    def get_optimizer_class(self) -> Optimizer:
        return torch.optim.Adam
