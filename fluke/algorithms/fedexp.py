"""Implementation of the FedExP [FedExP23]_ algorithm.

References:
    .. [FedExP23] Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi.
       FedExP: Speeding Up Federated Averaging via Extrapolation. In ICLR (2023).
       URL: https://arxiv.org/abs/2301.09604
"""
import sys
from typing import Collection

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..server import Server  # NOQA
from ..utils.model import flatten_parameters  # NOQA

__all__ = [
    "FedExPServer",
    "FedExP"
]


class FedExPServer(Server):

    @torch.no_grad()
    def aggregate(self, eligible: Collection[Client], client_models: Collection[Module]) -> None:
        W = flatten_parameters(self.model)
        client_models = list(client_models)
        Wi = [flatten_parameters(client_model) for client_model in client_models]
        eta = self._compute_eta(W, Wi)

        self.hyper_params.update(lr=eta)
        super().aggregate(eligible, client_models)

    def _compute_eta(self, W: torch.Tensor, Wi: list[torch.Tensor], eps: float = 1e-4) -> float:
        Delta_bar = torch.mean(W - torch.stack(Wi), dim=0)
        sum_norm_Delta_i = torch.sum(torch.norm(W - torch.stack(Wi))**2, dim=0)
        norm_Delta_bar = torch.norm(Delta_bar)**2
        eta = torch.max(sum_norm_Delta_i / (2 * len(Wi) * (norm_Delta_bar + eps)),
                        torch.FloatTensor([1.0]))
        return eta


class FedExP(CentralizedFL):

    def get_server_class(self) -> type[Server]:
        return FedExPServer
