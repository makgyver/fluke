from copy import deepcopy
from typing import Callable, Iterable, Union, Any, Optional

import torch
from torch.nn import Module
from algorithms import CentralizedFL
from server import Server

import sys; sys.path.append(".")
from fl_bench.utils import OptimizerConfigurator
from fl_bench.client import Client
from fl_bench.data import DataSplitter, FastTensorDataLoader


class FedProxClient(Client):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 mu: float,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):
        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.mu = mu
    
    def _proximal_loss(self, local_model, global_model):
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            proximal_term += torch.norm(w - w_t)**2
        return proximal_term

    def local_train(self, override_local_epochs: int=0):
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        W = deepcopy(self.model)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for i, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y) + (self.mu / 2) * self._proximal_loss(self.model, W)
                loss.backward()
                self.optimizer.step()          
            self.scheduler.step()
        return self.validate()


class FedProx(CentralizedFL):
    """FedProx federated learning algorithm.

    https://arxiv.org/pdf/1812.06127.pdf

    Parameters
    ----------
    n_clients : int
        The number of clients.
    n_rounds : int
        The number of rounds.
    n_epochs : int
        The number of local epochs.
    optimizer_cfg : OptimizerConfigurator
        The optimizer configurator for the clients.
    model : Module
        The model to be trained.
    client_mu : float
        The mu parameter for the FedProx algorithm.
    loss_fn : Callable
        The loss function.
    eligibility_percentage : float, optional
        The percentage of clients to be selected for training, by default 0.5.
    """
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 client_mu: float,
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
    
    def init_clients(self, data_splitter: DataSplitter, **kwargs):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.clients = [FedProxClient(train_set=data_splitter.client_train_loader[i], 
                                      mu=self.client_mu,
                                      optimizer_cfg=self.optimizer_cfg, 
                                      loss_fn=self.loss_fn, 
                                      validation_set=data_splitter.client_test_loader[i],
                                      local_epochs=self.n_epochs) for i in range(self.n_clients)]
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
               f"\u03BC={self.client_mu},P={self.eligibility_percentage},{self.optimizer_cfg})"