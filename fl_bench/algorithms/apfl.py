import sys
sys.path.append(".")
sys.path.append("..")
from typing import Any, Callable, Iterable

import torch
from torch.nn import Module
from copy import deepcopy

from ..comm import Message
from ..server import Server
from ..client import Client, PFLClient
from ..algorithms import PersonalizedFL
from ..data import FastTensorDataLoader
from ..utils import OptimizerConfigurator, clear_cache
from ..utils.model import merge_models


# https://arxiv.org/pdf/2012.04221.pdf
class APFLClient(PFLClient):

    def __init__(self, 
                 index: int,
                 model: torch.nn.Module,
                 train_set: FastTensorDataLoader, 
                 test_set: FastTensorDataLoader, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable[..., Any], 
                 local_epochs: int = 3,
                 lam: float = 0.25):
        super().__init__(index, model, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.pers_optimizer = None
        self.pers_scheduler = None
        self.internal_model = deepcopy(model)
        self.hyper_params.update({
            "lam": lam
        })

    
    def local_train(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self._receive_model()

        self.model.train()
        self.personalized_model.train()
        
        self.model.to(self.device)
        self.personalized_model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        
        if self.pers_optimizer is None:
            self.pers_optimizer, self.pers_scheduler = self.optimizer_cfg(self.internal_model)

        for _ in range(epochs):
            loss = None
            local_loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)

                # Global
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()

                # Local
                self.pers_optimizer.zero_grad()
                y_hat = merge_models(self.model, self.internal_model, self.hyper_params.lam)(X)
                local_loss = self.hyper_params.loss_fn(y_hat, y)
                local_loss.backward()
                self.pers_optimizer.step()

            self.scheduler.step()
            self.pers_scheduler.step()
        
        self.model.to("cpu")
        self.internal_model.to("cpu")
        clear_cache()

        self.personalized_model = merge_models(self.model, self.internal_model, self.hyper_params.lam)

        # self._send_model()
    
    def get_model(self):
        return self._send_model()


class APFLServer(Server):

    def __init__(self, 
                 model: Module, 
                 test_data: FastTensorDataLoader, 
                 clients: Iterable[Client], 
                 weighted: bool = False,
                 tau: int = 3):
        super().__init__(model, test_data, clients, weighted)
        self.hyper_params.update({
            "tau": tau
        })
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        if self.rounds % self.hyper_params.tau == 0:
            for client in eligible:
                self.channel.send(Message((client.get_model, {}), "__action__", self), client)
            super().aggregate(eligible)


class APFL(PersonalizedFL):
    
    def get_client_class(self) -> PFLClient:
        return APFLClient

    def get_server_class(self) -> Server:
        return APFLServer
