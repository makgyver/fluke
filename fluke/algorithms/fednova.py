"""Implementation of the [FedNova21]_ algorithm.

References:
    .. [FedNova21] Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H. Vincent Poor.
       Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization.
       In NeurIPS 2020. URL: https://arxiv.org/abs/2007.07481
"""
import sys
from copy import deepcopy
from typing import Any, Iterable

import torch

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from ..utils.model import STATE_DICT_KEYS_TO_IGNORE  # NOQA

__all__ = [
    "FedNovaClient",
    "FedNovaServer",
    "FedNova"
]


class FedNovaClient(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.tau = 0

    def _get_momentum(self):
        if self.optimizer is None:
            if "momentum" in self.optimizer_cfg.optimizer_kwargs:
                return self.optimizer_cfg.optimizer_kwargs["momentum"]
            else:
                return 0
        else:
            return self.optimizer.param_groups[0]["momentum"]

    def fit(self, override_local_epochs: int = 0) -> float:
        super().fit(override_local_epochs)
        self.tau += self.hyper_params.local_epochs * self.train_set.n_batches
        rho = self._get_momentum()
        self.a = (self.tau - rho * (1.0 - pow(rho, self.tau)) / (1.0 - rho)) / (1.0 - rho)
        self.channel.send(Message(self.a, "local_a", self), self.server)


class FedNovaServer(Server):

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        clients_sd = self.get_client_models(eligible)
        weights = self._get_client_weights(eligible)
        a_i = [
            self.channel.receive(self, client, "local_a").payload
            for client in eligible
        ]

        coeff = sum([a_i[i] * weights[i] for i in range(len(eligible))])
        avg_model_sd = deepcopy(self.model.state_dict())
        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                # avg_model_sd[key] = clients_sd[0][key].clone()
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue

            for i, client_sd in enumerate(clients_sd):
                avg_model_sd[key] += coeff * weights[i] * \
                    torch.true_divide(client_sd[key] - avg_model_sd[key], a_i[i])

        self.model.load_state_dict(avg_model_sd)


class FedNova(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedNovaClient

    def get_server_class(self) -> Server:
        return FedNovaServer
