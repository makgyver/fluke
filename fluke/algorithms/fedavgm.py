"""Implementation of the Federated Averaging with momentum [FedAVGM19]_ algorithm.

References:
    .. [FedAVGM19] Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown. Measuring the Effects of
       Non-Identical Data Distribution for Federated Visual Classification. In arXiv (2019).
       URL: https://arxiv.org/abs/1909.06335
"""
import sys
from copy import deepcopy
from typing import Any, Iterable

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils.model import state_dict_zero_like  # NOQA

__all__ = [
    "FedAVGMServer",
    "FedAVGM"
]


class FedAVGMServer(Server):
    """Server class for the FedAVGM algorithm.

    Args:
        model (Module): The model to be trained.
        test_set (FastDataLoader): The test data.
        clients (Iterable[Client]): The clients participating in the federated learning process.
        eval_every (int, optional): Evaluate the model every `eval_every` rounds. Defaults to 1.
        weighted (bool, optional): Use weighted averaging. Defaults to True.
        momentum (float, optional): The momentum hyper-parameter. Defaults to 0.9.
    """

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = True,
                 momentum: float = 0.9,
                 **kwargs: dict[str, Any]):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.hyper_params.update(momentum=momentum)
        self.momentum_vector = None

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        prev_model_sd = deepcopy(self.model.state_dict())
        super().aggregate(eligible)
        avg_model_sd = self.model.state_dict()

        if self.momentum_vector is None:
            self.momentum_vector = state_dict_zero_like(prev_model_sd)
        else:
            for key in prev_model_sd:
                self.momentum_vector[key].data = self.hyper_params.momentum * \
                    self.momentum_vector[key].data + \
                    prev_model_sd[key].data - avg_model_sd[key].data

        for key in prev_model_sd:
            avg_model_sd[key].data = prev_model_sd[key].data - self.momentum_vector[key].data

        self.model.load_state_dict(avg_model_sd)


class FedAVGM(CentralizedFL):

    def get_server_class(self) -> Server:
        return FedAVGMServer
