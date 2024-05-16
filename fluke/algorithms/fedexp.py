from collections import OrderedDict
from typing import Iterable
import torch
import sys
sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..utils.model import diff_model, STATE_DICT_KEYS_TO_IGNORE  # NOQA
from ..server import Server  # NOQA
from ..client import Client  # NOQA


class FedExPServer(Server):

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        clients_sd = self.get_client_models(eligible)
        clients_diff = [diff_model(self.model.state_dict(), client_model)
                        for client_model in clients_sd]
        eta, mu_diff = self._compute_eta(clients_diff)

        avg_model_sd = OrderedDict()
        w = self.model.state_dict()
        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                # avg_model_sd[key] = clients_sd[0][key].clone()
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue
            avg_model_sd[key] = w[key] - eta[key] * mu_diff[key]
        self.model.load_state_dict(avg_model_sd)

    def _compute_eta(self, clients_diff: Iterable[dict], eps: float = 1e-4) -> float:
        num = {}
        den = {}
        mu_diff = {}
        eta = {}
        M = len(clients_diff)

        for key in clients_diff[0].keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                continue
            num[key] = torch.sum(torch.FloatTensor(
                [torch.norm(c[key])**2 for c in clients_diff]))
            mu_diff[key] = torch.mean(torch.stack([c[key] for c in clients_diff]), dim=0)
            den[key] = 2 * M * (torch.norm(mu_diff[key])**2 + eps)
            eta[key] = torch.max(num[key] / den[key], torch.FloatTensor([1.0]))

        return eta, mu_diff


class FedExP(CentralizedFL):

    def get_server_class(self) -> Server:
        return FedExPServer
