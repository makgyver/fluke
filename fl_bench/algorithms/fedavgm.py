import sys; sys.path.append(".")

from copy import deepcopy
from typing import Any, Iterable
from collections import OrderedDict

import torch
from torch.nn import Module

from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench.utils import DDict, diff_model
from fl_bench.algorithms import CentralizedFL
from fl_bench.data import FastTensorDataLoader


class FedAVGMServer(Server):
    def __init__(self, 
                 model: Module,
                 test_data: FastTensorDataLoader,
                 clients: Iterable[Client],
                 momentum: float=0.9,
                 weighted: bool=True):
        super().__init__(model, test_data, clients, weighted)
        self.momentum = momentum
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr=1.0, 
                                         momentum=self.momentum, 
                                         nesterov=True)
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self._get_client_models(eligible)
        clients_diff = [diff_model(self.model.state_dict(), client_model) for client_model in clients_sd]

        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0][key])
                    continue
                den = 0
                for i, client_diff in enumerate(clients_diff):
                    weight = 1 if not self.weighted else eligible[i].n_examples
                    den += weight
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weight * client_diff[key]
                    else:
                        avg_model_sd[key] += weight * client_diff[key]
                avg_model_sd[key] /= den
        
        self.optimizer.zero_grad()
        for key, param in self.model.named_parameters():
            param.grad = avg_model_sd[key].data
        self.optimizer.step()
    
    def __str__(self) -> str:
        to_str = super().__str__()
        return f"{to_str[:-1]},momentum={self.momentum})"


class FedAVGM(CentralizedFL):

    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = FedAVGMServer(model, data, self.clients, **config)
    
