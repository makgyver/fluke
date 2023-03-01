from copy import deepcopy
from typing import Callable

import torch
from torch.nn import Module, CosineSimilarity

import sys
from fl_bench.client import Client
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench.server import Server; sys.path.append(".")
from fl_bench.utils import OptimizerConfigurator
from fl_bench.algorithms import CentralizedFL


class MOONClient(Client):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 mu: float,
                 tau: float,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):
        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.mu = mu
        self.tau = tau
        self.prev_model = None
        self.server_model = None

    def receive(self, model):
        if self.model is None:
            self.model = deepcopy(model)
            self.prev_model = deepcopy(model)
        else:
            self.prev_model.load_state_dict(self.model.state_dict())
            self.model.load_state_dict(model.state_dict())
        self.server_model = model

    def local_train(self, override_local_epochs: int=0):
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        cos = CosineSimilarity(dim=-1).to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat, z_local = self.model.forward_(X, -1)
                loss_sup = self.loss_fn(y_hat, y)

                self.prev_model.to(self.device)
                self.server_model.to(self.device)

                _, z_prev = self.prev_model.forward_(X, -1)
                _, z_global = self.server_model.forward_(X, -1)

                sim_lg = cos(z_local, z_global).reshape(-1, 1) / self.tau
                sim_lp = cos(z_local, z_prev).reshape(-1, 1) / self.tau
                loss_con = -torch.log(torch.exp(sim_lg) / (torch.exp(sim_lg) + torch.exp(sim_lp))).mean()

                loss = loss_sup + self.mu * loss_con
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        
        self.prev_model.to("cpu")
        self.server_model.to("cpu")
        
        return self.validate()

class MOON(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int, 
                 optimizer_cfg: OptimizerConfigurator, 
                 model: Module, 
                 client_mu: float,
                 client_tau: float,
                 loss_fn: Callable, 
                 eligibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         eligibility_percentage)
        self.client_mu = client_mu
        self.client_tau = client_tau
    
    def init_parties(self, data_splitter: DataSplitter, callback: Callable=None):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.clients = [ MOONClient(train_set=data_splitter.client_train_loader[i], 
                                    mu=self.client_mu,
                                    tau=self.client_tau,
                                    optimizer_cfg=self.optimizer_cfg, 
                                    loss_fn=self.loss_fn, 
                                    validation_set=data_splitter.client_test_loader[i],
                                    local_epochs=self.n_epochs) for i in range(self.n_clients) ]

        self.server = Server(self.model, self.clients, self.eligibility_percentage, weighted=True)
        self.server.register_callback(callback)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
               f"\u03BC={self.client_mu},\u03C4={self.client_tau},P={self.eligibility_percentage}," + \
               f"{self.optimizer_cfg})"
    
