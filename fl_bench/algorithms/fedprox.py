import sys
sys.path.append(".")
sys.path.append("..")

import torch
from copy import deepcopy
from typing import Callable

from . import CentralizedFL
from ..client import Client
from ..data import FastTensorDataLoader
from ..utils import OptimizerConfigurator, clear_cache

class FedProxClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastTensorDataLoader,
                 validation_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 mu: float):
        super().__init__(index, train_set, validation_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update({
            "mu": mu
        })

    
    def _proximal_loss(self, local_model, global_model):
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            proximal_term += torch.norm(w - w_t)**2
        return proximal_term

    def local_train(self, override_local_epochs: int=0):
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self._receive_model()
        W = deepcopy(self.model)
        self.model.to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y) + (self.hyper_params.mu / 2) * self._proximal_loss(self.model, W)
                loss.backward()
                self.optimizer.step()          
            self.scheduler.step()

        self.model.to("cpu")
        clear_cache()
        self._send_model()


class FedProx(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedProxClient
    