from torch.optim import Optimizer

from abc import ABC
from copy import deepcopy
from typing import Callable

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from algorithms import CentralizedFL
from data import Datasets
from server import Server

from utils import OptimizerConfigurator
from client import Client

class FedProxClient(Client):
    def __init__(self,
                 dataset: DataLoader,
                 mu: float,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable, # CHECK ME
                 local_epochs: int=3,
                 device: torch.device=torch.device('cpu'),
                 seed: int=42):
        super().__init__(dataset, optimizer_cfg, loss_fn, local_epochs, device, seed)
        self.mu = mu
    
    def _proximal_loss(self, local_model, global_model):
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            proximal_term += (w - w_t).norm(2)
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
                 batch_size: int, 
                 train_set: Datasets,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 client_mu: float,
                 loss_fn: Callable,
                 elegibility_percentage: float=0.5,
                 device: torch.device=torch.device('cpu'),
                 seed: int=42):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         batch_size, 
                         train_set,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage,
                         device, 
                         seed)
        self.client_mu = client_mu
    
    def init_parties(self, callback: Callable=None):
        assert self.client_loader is not None, 'You must prepare data before initializing parties'
        self.clients = [FedProxClient(dataset=self.client_loader[i], 
                                      mu=self.client_mu,
                                      optimizer_cfg=self.optimizer_cfg, 
                                      loss_fn=self.loss_fn, 
                                      local_epochs=self.n_epochs,
                                      device=self.device,
                                      seed=self.seed) for i in range(self.n_clients)]

        self.server = Server(self.model.to(self.device), self.clients, self.elegibility_percentage, seed=self.seed)
        self.server.register_callback(callback)