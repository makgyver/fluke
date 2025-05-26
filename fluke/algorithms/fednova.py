"""Implementation of the [FedNova21]_ algorithm.

References:
    .. [FedNova21] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H. Vincent Poor.
       Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization.
       In NeurIPS 2020. URL: https://arxiv.org/abs/2007.07481
"""

import sys
from typing import Collection, Sequence

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils.model import aggregate_models  # NOQA

__all__ = ["FedNovaClient", "FedNovaServer", "FedNova"]


class FedNovaClient(Client):

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: torch.nn.Module,
        local_epochs: int,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        **kwargs,
    ):
        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss_fn,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            **kwargs,
        )
        self.tau = 0

    def _get_momentum(self) -> float:
        if self.optimizer is None:
            if "momentum" in self._optimizer_cfg.optimizer_cfg:
                return self._optimizer_cfg.optimizer_cfg["momentum"]
            else:
                return 0
        else:
            return self.optimizer.param_groups[0]["momentum"]

    def fit(self, override_local_epochs: int = 0) -> float:
        loss = super().fit(override_local_epochs)
        self.tau = self.hyper_params.local_epochs * self.train_set.n_batches
        rho = self._get_momentum()
        self.a = (self.tau - rho * (1.0 - pow(rho, self.tau)) / (1.0 - rho)) / (1.0 - rho)
        self.channel.send(Message(self.a, "local_a", self.index, inmemory=True), "server")
        return loss


class FedNovaServer(Server):

    @torch.no_grad()
    def aggregate(self, eligible: Sequence[Client], client_models: Collection[Module]) -> None:
        weights = self._get_client_weights(eligible)
        a_i = [
            self.channel.receive("server", client.index, "local_a").payload for client in eligible
        ]
        coeff = sum([a_i[i] * weights[i] for i in range(len(eligible))])
        weights = torch.true_divide(torch.tensor(weights) * coeff, torch.tensor(a_i))
        aggregate_models(self.model, client_models, weights, self.hyper_params.lr)


class FedNova(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return FedNovaClient

    def get_server_class(self) -> type[Server]:
        return FedNovaServer
