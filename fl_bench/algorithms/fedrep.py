import sys; sys.path.append(".")
from typing import Any, Callable

import torch
from copy import deepcopy

from fl_bench import Message
from fl_bench.client import PFLClient
from fl_bench.data import FastTensorDataLoader
from fl_bench.utils import OptimizerConfigurator, clear_cache
from fl_bench.algorithms import PersonalizedFL

# https://arxiv.org/abs/1912.00818

class FedRepClient(PFLClient):

    def __init__(self, 
                 model: torch.nn.Module,
                 train_set: FastTensorDataLoader, 
                 validation_set: FastTensorDataLoader, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable[..., Any], 
                 local_epochs: int = 3,
                 tau: int = 3):
        super().__init__(model, train_set, validation_set, optimizer_cfg, loss_fn, local_epochs)
        self.pers_optimizer = None
        self.pers_scheduler = None
        self.hyper_params.update({
            "tau": tau
        })
    
    def local_train(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self._receive_model()
        self.model.train()
        self.model.to(self.device)

        # update downstream layers
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = ('downstream' in name)

        if self.pers_optimizer is None:
            self.pers_optimizer, self.pers_scheduler = self.optimizer_cfg(self.model.downstream)
        
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.pers_optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.pers_optimizer.step()
            self.pers_scheduler.step()
        
        # update downstream layers
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = ('downstream' not in name)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model.fed_E)
        
        for _ in range(self.hyper_params.tau):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.model.to("cpu")
        clear_cache()
        self._send_model()

    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.fed_E), "model", self), self.server)

    def _receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.fed_E.load_state_dict(msg.payload.state_dict())
    

class FedRep(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedRepClient
