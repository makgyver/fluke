import sys
from typing import Any, Callable

import torch
from fl_bench.data import FastTensorDataLoader
from fl_bench.evaluation import ClassificationEval

from fl_bench.utils import OptimizerConfigurator, clear_cache; sys.path.append(".")
from copy import deepcopy

from fl_bench import Message
from fl_bench.server import Server
from fl_bench.client import PFLClient
from fl_bench.algorithms import PersonalizedFL

# https://arxiv.org/pdf/2012.04221.pdf
class DittoClient(PFLClient):

    def __init__(self, 
                 model: torch.nn.Module,
                 train_set: FastTensorDataLoader, 
                 validation_set: FastTensorDataLoader, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable[..., Any], 
                 local_epochs: int = 3,
                 tau: int = 3,
                 lam: float = 0.1):
        super().__init__(model, train_set, validation_set, optimizer_cfg, loss_fn, local_epochs)
        self.local_optimizer = None
        self.local_scheduler = None
        self.hyper_params.update({
            "tau": tau,
            "lam": lam
        })

    def _proximal_loss(self, local_model, global_model):
        proximal_term = 0.0
        for name, param in local_model.named_parameters():
            if 'weight' not in name: continue
            proximal_term += (param - global_model.get_parameter(name)).norm(2)
        return proximal_term
    
    def local_train(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self._receive_model()

        w_prev = deepcopy(self.model)

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        
        for _ in range(epochs):
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

        self.private_model.train()
        self.private_model.to(self.device)
        w_prev.to(self.device)

        if self.local_optimizer is None:
            self.local_optimizer, self.local_scheduler = self.optimizer_cfg(self.private_model)
        
        for _ in range(self.hyper_params.tau):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.local_optimizer.zero_grad()
                y_hat = self.private_model(X)
                loss = self.hyper_params.loss_fn(y_hat, y) + self.hyper_params.lam * self._proximal_loss(self.private_model, w_prev)
                loss.backward()
                self.local_optimizer.step()
            self.local_scheduler.step()

        self.private_model.to("cpu")
        w_prev.to("cpu")
        clear_cache()
        self._send_model()


class Ditto(PersonalizedFL):
    
    def get_client_class(self) -> PFLClient:
        return DittoClient
