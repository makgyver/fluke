from torch.optim import Optimizer

from copy import deepcopy
from typing import Callable

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from algorithms import CentralizedFL
from fl_bench.data import DataSplitter
from server import Server

import sys; sys.path.append(".")
from fl_bench.utils import OptimizerConfigurator
from fl_bench.client import Client

class FedProxClient(Client):
    def __init__(self,
                 dataset: DataLoader,
                 mu: float,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable, # CHECK ME
                 local_epochs: int=3,
                 seed: int=42):
        super().__init__(dataset, optimizer_cfg, loss_fn, local_epochs, seed)
        self.mu = mu
    
    def _proximal_loss(self, local_model, global_model):
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            proximal_term += torch.norm(w - w_t)**2
        return proximal_term

    def local_train(self, override_local_epochs: int=0, log_interval: int=0):
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        W = deepcopy(self.model)
        #total_step = len(self.dataset)
        self.model.train()
        if self.optimizer is None:
            self.optimizer = self.optimizer_cfg(self.model)
        for epoch in range(epochs):
            loss = None
            for i, (X, y) in enumerate(self.dataset):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y) + (self.mu / 2) * self._proximal_loss(self.model, W)
                loss.backward()
                self.optimizer.step()          
        return None # CHECK ME


class FedProx(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 client_mu: float,
                 loss_fn: Callable,
                 elegibility_percentage: float=0.5,
                 seed: int=42):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage,
                         seed)
        self.client_mu = client_mu
    
    def init_parties(self, data_splitter: DataSplitter, callback: Callable=None):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.clients = [FedProxClient(dataset=data_splitter.client_loader[i], 
                                      mu=self.client_mu,
                                      optimizer_cfg=self.optimizer_cfg, 
                                      loss_fn=self.loss_fn, 
                                      local_epochs=self.n_epochs,
                                      seed=self.seed) for i in range(self.n_clients)]

        self.server = Server(self.model, self.clients, self.elegibility_percentage, weighted=True, seed=self.seed)
        self.server.register_callback(callback)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
               f"\u03BC={self.client_mu},P={self.elegibility_percentage},seed={self.seed})"