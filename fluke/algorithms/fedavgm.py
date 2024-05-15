from torch.nn import Module
import torch
from collections import OrderedDict
from typing import Iterable
import sys
sys.path.append(".")
sys.path.append("..")

from ..data import FastDataLoader  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..utils.model import diff_model, STATE_DICT_KEYS_TO_IGNORE  # NOQA
from ..server import Server  # NOQA
from ..client import Client  # NOQA


class FedAVGMServer(Server):
    def __init__(self,
                 model: Module,
                 test_data: FastDataLoader,
                 clients: Iterable[Client],
                 eval_every: int = 1,
                 weighted: bool = True,
                 momentum: float = 0.9):
        super().__init__(model, test_data, clients, eval_every, weighted)
        self.hyper_params.update(momentum=momentum)

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self.get_client_models(eligible)
        clients_diff = [diff_model(self.model.state_dict(), client_model)
                        for client_model in clients_sd]
        weights = self._get_client_weights(eligible)

        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue
            for i, client_diff in enumerate(clients_diff):
                if key not in avg_model_sd:
                    avg_model_sd[key] = weights[i] * client_diff[key]
                else:
                    avg_model_sd[key] += weights[i] * client_diff[key]

        for key, param in self.model.named_parameters():
            param.data = self.hyper_params.momentum * param.data - avg_model_sd[key].data


class FedAVGM(CentralizedFL):

    def get_server_class(self) -> Server:
        return FedAVGMServer
