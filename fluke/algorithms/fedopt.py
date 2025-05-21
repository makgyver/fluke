"""Implementation of the [FedOpt21]_ algorithm.

References:
    .. [FedOpt21] Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
       Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan. Adaptive Federated Optimization.
       In ICLR (2021). URL: https://openreview.net/pdf?id=LkFG3lB13U5
"""
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Collection

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils.model import get_trainable_keys  # NOQA

__all__ = [
    "FedOptServer",
    "FedOpt"
]


class FedOptServer(Server):
    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Collection[Client],
                 mode: str = "adam",
                 lr: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 tau: float = 0.0001,
                 weighted: bool = True):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        assert mode in {"adam", "yogi", "adagrad"}, \
            "'mode' must be one of {'adam', 'yogi', 'adagrad'}"
        assert 0 <= beta1 < 1, "beta1 must be in [0, 1)"
        assert 0 <= beta2 < 1, "beta2 must be in [0, 1)"

        self.hyper_params.update(
            mode=mode,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            tau=tau
        )
        self._init_moments()

    def _init_moments(self) -> None:
        self.m_t = OrderedDict()
        self.v_t = OrderedDict()
        for key in self.model.state_dict().keys():
            if "num_batches_tracked" not in key:
                self.m_t[key] = torch.zeros_like(self.model.state_dict()[key])
                # This guarantees that the second moment is >= 0 and >= tau^2
                self.v_t[key] = torch.zeros_like(self.model.state_dict()[key])

    @torch.no_grad()
    def aggregate(self, eligible: Collection[Client], client_models: Collection[Module]) -> None:
        prev_model = deepcopy(self.model)
        super().aggregate(eligible, client_models)
        aggregated = self.model.state_dict()
        server_sd = prev_model.state_dict()
        b1, b2 = self.hyper_params.beta1, self.hyper_params.beta2
        eta, tau = self.hyper_params.lr, self.hyper_params.tau

        trainable_keys = get_trainable_keys(self.model)
        d_t = {k: aggregated[k] - server_sd[k] for k in trainable_keys}
        self.m_t = {k: b1 * self.m_t[k] + (1 - b1) * d_t[k] for k in trainable_keys}

        if self.hyper_params.mode == "adam":
            self.v_t = {k: b2 * self.v_t[k] + (1 - b2) * d_t[k] ** 2 for k in trainable_keys}
        elif self.hyper_params.mode == "yogi":
            self.v_t = {k: self.v_t[k] - (1 - b2) * (d_t[k] ** 2) *
                        torch.sign(self.v_t[k] - d_t[k] ** 2) for k in trainable_keys}
        elif self.hyper_params.mode == "adagrad":
            self.v_t = {k: self.v_t[k] + d_t[k] ** 2 for k in trainable_keys}
        else:
            raise ValueError(f"Unknown mode: {self.hyper_params.mode}")

        update = {k: eta * self.m_t[k] / (torch.sqrt(self.v_t[k]) + tau) for k in trainable_keys}
        agg_model_sd = {k: server_sd[k] + update[k] for k in trainable_keys}
        self.model.load_state_dict(agg_model_sd, strict=False)


class FedOpt(CentralizedFL):

    def get_server_class(self) -> type[Server]:
        return FedOptServer
