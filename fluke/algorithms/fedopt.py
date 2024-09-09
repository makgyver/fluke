"""Implementation of the [FedOpt21]_ algorithm.

References:
    .. [FedOpt21] Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush,
       Jakub Konečný, Sanjiv Kumar, H. Brendan McMahan. Adaptive Federated Optimization.
       In ICLR (2021). URL: https://openreview.net/pdf?id=LkFG3lB13U5
"""
import sys
from collections import OrderedDict
from typing import Iterable

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils.model import STATE_DICT_KEYS_TO_IGNORE  # NOQA

__all__ = [
    "FedOptServer",
    "FedOpt"
]


class FedOptServer(Server):
    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 mode: str = "fedadam",
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

    def _init_moments(self):
        self.m = OrderedDict()
        self.v = OrderedDict()
        for key in self.model.state_dict().keys():
            if "num_batches_tracked" not in key:
                self.m[key] = torch.zeros_like(self.model.state_dict()[key])
                # This guarantees that the second moment is >= 0 and <= tau^2
                self.v[key] = torch.rand_like(self.model.state_dict()[
                                              key]) * self.hyper_params.tau ** 2

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self.get_client_models(eligible)

        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                # avg_model_sd[key] = deepcopy(clients_sd[0][key])
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue

            den, diff = 0, 0
            for i, client_sd in enumerate(clients_sd):
                weight = 1 if not self.hyper_params.weighted else eligible[i].n_examples
                diff += weight * (client_sd[key] - self.model.state_dict()[key])
                den += weight
            diff /= den
            self.m[key] = self.hyper_params.beta1 * \
                self.m[key] + (1 - self.hyper_params.beta1) * diff

            diff_2 = diff ** 2
            if self.hyper_params.mode == "adam":
                self.v[key] = self.hyper_params.beta2 * self.v[key] + \
                    (1 - self.hyper_params.beta2) * diff_2
            elif self.hyper_params.mode == "yogi":
                self.v[key] -= (1 - self.hyper_params.beta2) * \
                    diff_2 * torch.sign(self.v[key] - diff_2)
            elif self.hyper_params.mode == "adagrad":
                self.v[key] += diff_2

            update = self.m[key] + self.hyper_params.lr * self.m[key] / \
                (torch.sqrt(self.v[key]) + self.hyper_params.tau)
            avg_model_sd[key] = self.model.state_dict()[key] + update

        self.model.load_state_dict(avg_model_sd)


class FedOpt(CentralizedFL):

    def get_server_class(self) -> Server:
        return FedOptServer
