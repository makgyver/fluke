"""Implementation of the FedProto [FedProto22]_ algorithm.

References:
    .. [FedProto22] Yue Tan, Guodong Long, Lu Liu, Tianyi Zhou, Qinghua Lu, Jing Jiang, Chengqi
       Zhang. FedProto: Federated Prototype Learning across Heterogeneous Clients. In AAAI (2022).
       URL: https://arxiv.org/abs/2105.00243
"""
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..client import PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..nets import EncoderHeadNet  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from . import PersonalizedFL  # NOQA

__all__ = [
    "FedProtoModel",
    "FedProtoClient",
    "FedProtoServer",
    "FedProto"
]


class FedProtoModel(Module):
    def __init__(self,
                 model: EncoderHeadNet,
                 prototypes: dict[int, torch.Tensor],
                 device: torch.device):
        super().__init__()
        self.model: EncoderHeadNet = model
        self.prototypes: dict[int, torch.Tensor] = prototypes
        self.num_classes: int = len(prototypes)
        self.device: torch.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mse_loss = torch.nn.MSELoss()
        Z = self.model.forward_encoder(x)
        output = float('inf') * torch.ones(x.shape[0], self.num_classes).to(self.device)
        for i, r in enumerate(Z):
            for j, proto in self.prototypes.items():
                # CHECKME: is the following lines necessary?
                if proto is not None and type(proto) is not type([]):
                    output[i, j] = mse_loss(r, proto)
                else:
                    output[i, j] = float('inf')

        # Return the negative of the distance so
        # to compute the argmax to get the closest prototype
        return -output


class FedProtoClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int,
                 n_protos: int,
                 lam: float,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, model=model, train_set=train_set,
                         test_set=test_set, optimizer_cfg=optimizer_cfg, loss_fn=loss_fn,
                         local_epochs=local_epochs, **kwargs)
        self.hyper_params.update(
            n_protos=n_protos,
            lam=lam
        )
        self.model = self.personalized_model
        self.prototypes = {i: None for i in range(self.hyper_params.n_protos)}
        self.global_protos = None

    def receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.global_protos = msg.payload

    def send_model(self):
        self.channel.send(Message(self.prototypes, "model", self), self.server)

    def _update_protos(self, protos: Iterable[torch.Tensor]) -> None:
        for label, prts in protos.items():
            self.prototypes[label] = torch.sum(torch.vstack(prts), dim=0) / len(prts)

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        mse_loss = torch.nn.MSELoss()
        protos = defaultdict(list)
        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                Z = self.model.forward_encoder(X)
                y_hat = self.model.forward_head(Z)
                loss = self.hyper_params.loss_fn(y_hat, y)

                if self.server.rounds > 0:  # this is actually illegal in fluke :)
                    proto_new = deepcopy(Z.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if self.global_protos[y_c] is not None:
                            proto_new[i, :] = self.global_protos[y_c].data
                        else:
                            proto_new[i, :] = torch.zeros_like(proto_new[i, :])
                    loss += self.hyper_params.lam * mse_loss(proto_new, Z)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(Z[i, :].detach().data)

                # for label in range(self.hyper_params.n_protos):
                #     ids = y == label
                #     if ids.sum() > 0:
                #         protos[label].append(Z[ids, :].detach().data)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.model.to("cpu")
        clear_cache()
        self._update_protos(protos)
        return running_loss

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if test_set is not None and self.prototypes[0] is not None:
            model = FedProtoModel(self.model, self.prototypes, self.device)
            return evaluator.evaluate(self._last_round, model, test_set)
        return {}

    def finalize(self) -> None:
        self.fit()


class FedProtoServer(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[PFLClient],
                 weighted: bool = True,
                 n_protos: int = 10):
        super().__init__(model=None, test_set=None, clients=clients, weighted=weighted)
        self.hyper_params.update(n_protos=n_protos)
        self.prototypes = [None for _ in range(self.hyper_params.n_protos)]

    def broadcast_model(self, eligible: Iterable[PFLClient]) -> None:
        # This funciton broadcasts the prototypes to the clients
        self.channel.broadcast(Message(self.prototypes, "model", self), eligible)

    def get_client_models(self, eligible: Iterable[PFLClient], state_dict: bool = False):
        return [self.channel.receive(self, client, "model").payload for client in eligible]

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[PFLClient]) -> None:
        # Recieve models from clients, i.e., the prototypes
        clients_protos = self.get_client_models(eligible)

        # Group by label
        label_protos = {i: [protos[i] for protos in clients_protos if protos[i] is not None]
                        for i in range(self.hyper_params.n_protos)}

        # Aggregate prototypes
        for label, protos in label_protos.items():
            if protos:
                self.prototypes[label] = torch.sum(torch.stack(protos), dim=0) / len(protos)


class FedProto(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedProtoClient

    def get_server_class(self) -> Server:
        return FedProtoServer
