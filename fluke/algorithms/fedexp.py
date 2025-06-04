"""Implementation of the FedExP [FedExP23]_ algorithm.

References:
    .. [FedExP23] Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi.
       FedExP: Speeding Up Federated Averaging via Extrapolation. In ICLR (2023).
       URL: https://arxiv.org/abs/2301.09604
"""

import sys
from typing import Collection, Sequence

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..server import Server  # NOQA
from ..utils.model import flatten_parameters  # NOQA

__all__ = ["FedExPServer", "FedExP"]


def _compute_eta(w: torch.Tensor, wi: list[torch.Tensor], eps: float = 1e-4) -> float:
    delta_bar = torch.mean(w - torch.stack(wi), dim=0)
    sum_norm_delta_i = torch.sum(torch.norm(w - torch.stack(wi)) ** 2, dim=0)
    norm_delta_bar = torch.norm(delta_bar) ** 2
    eta = torch.max(
        sum_norm_delta_i / (2 * len(wi) * (norm_delta_bar + eps)), torch.FloatTensor([1.0])
    )
    return eta.item()


class FedExPServer(Server):

    @torch.no_grad()
    def aggregate(self, eligible: Sequence[Client], client_models: Collection[Module]) -> None:
        w = flatten_parameters(self.model)
        client_models = list(client_models)
        wi = [flatten_parameters(client_model) for client_model in client_models]
        eta = _compute_eta(w, wi)

        self.hyper_params.update(lr=eta)
        super().aggregate(eligible, client_models)


class FedExP(CentralizedFL):

    def get_server_class(self) -> type[Server]:
        return FedExPServer
