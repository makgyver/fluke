"""Implementation of the FedExP [FedExP23]_ algorithm.

References:
    .. [FedExP23] Divyansh Jhunjhunwala, Shiqiang Wang, and Gauri Joshi.
       FedExP: Speeding Up Federated Averaging via Extrapolation. In ICLR (2023).
       URL: https://arxiv.org/abs/2301.09604
"""
import sys
from copy import deepcopy
from typing import Iterable

import torch

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..server import Server  # NOQA
from ..utils.model import STATE_DICT_KEYS_TO_IGNORE, flatten_parameters  # NOQA

__all__ = [
    "FedExPServer",
    "FedExP"
]


class FedExPServer(Server):

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        W = flatten_parameters(self.model)
        clients_model = self.get_client_models(eligible, state_dict=False)
        Wi = [flatten_parameters(client_model) for client_model in clients_model]
        eta = self._compute_eta(W, Wi)

        clients_sd = [client.model.state_dict() for client in eligible]
        avg_model_sd = deepcopy(self.model.state_dict())
        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                continue
            avg_model_sd[key] = avg_model_sd[key] - eta * torch.mean(
                torch.stack([avg_model_sd[key] - client_sd[key] for client_sd in clients_sd]),
                dim=0)
        self.model.load_state_dict(avg_model_sd)

    def _compute_eta(self, W: torch.Tensor, Wi: list[torch.Tensor], eps: float = 1e-4) -> float:
        Delta_bar = torch.mean(W - torch.stack(Wi), dim=0)
        sum_norm_Delta_i = torch.sum(torch.norm(W - torch.stack(Wi))**2, dim=0)
        norm_Delta_bar = torch.norm(Delta_bar)**2
        eta = torch.max(sum_norm_Delta_i / (2 * len(Wi) * (norm_Delta_bar + eps)),
                        torch.FloatTensor([1.0]))
        return eta


class FedExP(CentralizedFL):

    def get_server_class(self) -> Server:
        return FedExPServer
