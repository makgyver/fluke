import sys
from typing import Any, Callable, Iterable

import torch
from torch.nn import Module
from fl_bench.data import FastTensorDataLoader
from fl_bench.evaluation import ClassificationEval

from fl_bench.utils import OptimizerConfigurator, clear_cache; sys.path.append(".")
from copy import deepcopy

from fl_bench import Message
from fl_bench.server import Server
from fl_bench.client import Client, PFLClient
from fl_bench.algorithms import PersonalizedFL

# https://arxiv.org/pdf/2012.04221.pdf
class APFLClient(PFLClient):

    def _merge_models(self, global_model, local_model):
        merged_model = deepcopy(global_model)
        for name, param in merged_model.named_parameters():
            param.data = (1 - self.hyper_params.lam) * global_model.get_parameter(name).data + \
                              self.hyper_params.lam  * local_model.get_parameter(name).data
        return merged_model
        

    def __init__(self, 
                 model: torch.nn.Module,
                 train_set: FastTensorDataLoader, 
                 validation_set: FastTensorDataLoader, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable[..., Any], 
                 local_epochs: int = 3,
                 lam: float = 0.25):
        super().__init__(model, train_set, validation_set, optimizer_cfg, loss_fn, local_epochs)
        self.local_optimizer = None
        self.local_scheduler = None
        self.internal_model = deepcopy(model)
        self.hyper_params.update({
            "lam": lam
        })

    
    def local_train(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self._receive_model()

        self.model.train()
        self.private_model.train()
        
        self.model.to(self.device)
        self.private_model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        
        if self.local_optimizer is None:
            self.local_optimizer, self.local_scheduler = self.optimizer_cfg(self.internal_model)

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
                self.local_optimizer.zero_grad()
                y_hat = self._merge_models(self.model, self.internal_model)(X)
                local_loss = self.hyper_params.loss_fn(y_hat, y)
                local_loss.backward()
                self.local_optimizer.step()

            self.scheduler.step()
            self.local_scheduler.step()
        
        self.model.to("cpu")
        self.internal_model.to("cpu")
        clear_cache()

        self.private_model = self._merge_models(self.model, self.internal_model)

        # self._send_model()
    
    def get_model(self):
        return self._send_model()


class APFLServer(Server):

    def __init__(self, 
                 model: Module, 
                 test_data: FastTensorDataLoader, 
                 clients: Iterable[Client], 
                 weighted: bool = False,
                 tau: int = 3,):
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
