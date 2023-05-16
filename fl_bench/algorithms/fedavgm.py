from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Iterable
import torch

from torch.nn import Module

import sys
from fl_bench.client import Client; sys.path.append(".")
from fl_bench.server import Server
from fl_bench.utils import OptimizerConfigurator, diff_model
from fl_bench.algorithms import CentralizedFL

class FedAVGMServer(Server):
    def __init__(self, 
                 model: Module,
                 clients: Iterable[Client],
                 momentum: float=0.9,
                 eligibility_percentage: float=0.5, 
                 weighted: bool=True):
        super().__init__(model, clients, eligibility_percentage, weighted)
        self.momentum = momentum
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0, momentum=self.momentum, nesterov=True)
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = [self.receive(eligible[i], "model").payload.state_dict() for i in range(len(eligible))]
        clients_diff = [diff_model(self.model.state_dict(), client_model) for client_model in clients_sd]
        # clients_diff = clients_sd

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

class FedAVGM(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int, 
                 optimizer_cfg: OptimizerConfigurator, 
                 model: Module, 
                 server_momentum: float,
                 loss_fn: Callable, 
                 eligibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         eligibility_percentage)
        self.server_momentum = server_momentum

    def init_server(self, **kwargs):
        self.server = FedAVGMServer(self.model, 
                                    self.clients, 
                                    self.server_momentum, 
                                    self.eligibility_percentage, 
                                    weighted=True)
    
