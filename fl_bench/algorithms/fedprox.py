import sys; sys.path.append(".")

import torch
from copy import deepcopy
from typing import Callable
from algorithms import CentralizedFL

from fl_bench import Message
from fl_bench.client import Client
from fl_bench.data import FastTensorDataLoader
from fl_bench.utils import DDict, OptimizerConfigurator, get_loss

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
        self._receive_model()
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
        self.channel.send(Message(deepcopy(self.model), "model", self), self.server)
    
    def __str__(self) -> str:
        to_str = super().__str__()
        return f"{to_str[:-1]},mu={self.mu})"


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
    eligible_perc : float, optional
        The percentage of clients to be selected for training, by default 0.5.
    """
    
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict):
        optimizer_cfg = OptimizerConfigurator(self.get_optimizer_class(), 
                                              lr=config.optimizer.lr, 
                                              scheduler_kwargs=config.optimizer.scheduler_kwargs)
        self.loss = get_loss(config.loss)
        self.clients = [FedProxClient(train_set=clients_tr_data[i], 
                                      mu=config.mu,
                                      optimizer_cfg=optimizer_cfg, 
                                      loss_fn=self.loss, 
                                      validation_set=clients_te_data[i],
                                      local_epochs=config.n_epochs) for i in range(self.n_clients)]
    