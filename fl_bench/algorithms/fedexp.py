from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Iterable, Optional, Union

import torch
from torch.nn import Module

import sys; sys.path.append(".")
from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench.utils import OptimizerConfigurator, diff_model
from fl_bench.algorithms import CentralizedFL


class FedExPServer(Server):
    def __init__(self,
                 model: Module,
                 n_clients: int,
                 eligibility_percentage: float=0.5):
        super().__init__(model, n_clients, eligibility_percentage)
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        clients_sd = [eligible[i].send().state_dict() for i in range(len(eligible))]
        clients_diff = [diff_model(self.model.state_dict(), client_model) for client_model in clients_sd]
        eta, mu_diff = self._compute_eta(clients_diff)
        
        avg_model_sd = OrderedDict()
        w = self.model.state_dict()
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0][key])
                    continue
                avg_model_sd[key] = w[key] - eta[key] * mu_diff[key]
            self.model.load_state_dict(avg_model_sd)


    def _compute_eta(self, clients_diff: Iterable[dict], eps: float=1e-4) -> float:
        num = {}
        den = {}
        mu_diff = {}
        eta = {}
        M = len(clients_diff)

        for key in clients_diff[0].keys():
            num[key] = torch.sum(torch.FloatTensor([torch.norm(c[key])**2 for c in clients_diff]))
            mu_diff[key] = torch.mean(torch.stack([c[key] for c in clients_diff]), dim=0)
            den[key] = 2 * M *  (torch.norm(mu_diff[key])**2 + eps)
            eta[key] = torch.max(num[key] / den[key], torch.FloatTensor([1.0]))

        return eta, mu_diff

    

class FedExP(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 loss_fn: Callable,
                 eligibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         eligibility_percentage)
    
    def init_server(self, **kwargs):
        self.server = FedExPServer(self.model, self.clients, self.eligibility_percentage)