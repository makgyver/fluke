from enum import Enum
from copy import deepcopy
from collections import OrderedDict
from typing import Callable, Iterable

import torch
from torch.nn import Module

import sys; sys.path.append(".")
from fl_bench.data import DataSplitter
from fl_bench.server import Server
from fl_bench.client import Client
from fl_bench.utils import OptimizerConfigurator
from fl_bench.algorithms import CentralizedFL

class FedOptMode(Enum):
    FedAdam = 1
    FedYogi = 2
    FedAdagrad = 3


class FedOptServer(Server):
    def __init__(self,
                model: Module,
                clients: Iterable[Client], 
                mode: FedOptMode=FedOptMode.FedAdam,
                lr: float=0.001,
                beta1: float=0.9,
                beta2: float=0.999,
                tau: float=0.0001,
                elegibility_percentage: float=0.5, 
                weighted: bool=True):
        super().__init__(model, clients, elegibility_percentage, weighted)
        assert mode in FedOptMode, "mode must be one of FedOptMode"
        assert 0 <= beta1 < 1, "beta1 must be in [0, 1)"
        assert 0 <= beta2 < 1, "beta2 must be in [0, 1)"
        self.mode = mode
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self._init_moments()

    def _init_moments(self):
        self.m = OrderedDict()
        self.v = OrderedDict()
        for key in self.model.state_dict().keys():
            if not "num_batches_tracked" in key:
                self.m[key] = torch.zeros_like(self.model.state_dict()[key])
                # This guarantees that the second moment is >= 0 and <= tau^2
                self.v[key] = torch.rand_like(self.model.state_dict()[key]) * self.tau ** 2
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = [eligible[i].send().state_dict() for i in range(len(eligible))]
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0][key])
                    continue

                den, diff = 0, 0
                for i, client_sd in enumerate(clients_sd):
                    weight = 1 if not self.weighted else eligible[i].n_examples
                    diff += weight * (client_sd[key] - self.model.state_dict()[key])
                    den += weight
                diff /= den
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * diff

                diff_2 = diff ** 2
                if self.mode == FedOptMode.FedAdam:
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * diff_2
                elif self.mode == FedOptMode.FedYogi:
                    self.v[key] -= (1 - self.beta2) * diff_2 * torch.sign(self.v[key] - diff_2)
                elif self.mode == FedOptMode.FedAdagrad:
                    self.v[key] += diff_2
                    
                update = self.m[key] + self.lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
                avg_model_sd[key] = self.model.state_dict()[key] + update
            
            self.model.load_state_dict(avg_model_sd)


class FedOpt(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int, 
                 optimizer_cfg: OptimizerConfigurator, 
                 mode: FedOptMode,
                 server_lr: float,
                 beta1: float,
                 beta2: float,
                 tau: float,
                 model: Module, 
                 loss_fn: Callable, 
                 elegibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage)
        self.mode = mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.server_lr = server_lr
        self.tau = tau
    
    def init_parties(self, data_splitter: DataSplitter, callback: Callable=None):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.data_assignment = data_splitter.assignments
        self.clients = [Client(train_set=data_splitter.client_train_loader[i], 
                               optimizer_cfg=self.optimizer_cfg, 
                               loss_fn=self.loss_fn, 
                               validation_set=data_splitter.client_test_loader[i],
                               local_epochs=self.n_epochs) for i in range(self.n_clients)]

        self.server = FedOptServer(self.model, 
                                   self.clients, 
                                   mode=self.mode,
                                   lr=self.server_lr,
                                   beta1=self.beta1,
                                   beta2=self.beta2,
                                   tau=self.tau,
                                   elegibility_percentage=self.elegibility_percentage)

        self.server.register_callback(callback)

    def __str__(self) -> str:
        return f"{self.mode._name_}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
               f"\u03B7={self.server_lr},\u03B21={self.beta1},\u03B22={self.beta2}," + \
               f"\u03A4={self.tau},P={self.elegibility_percentage},{self.optimizer_cfg})"
    

