import sys; sys.path.append(".")

from copy import deepcopy
from collections import OrderedDict
from typing import Callable, Iterable, List

import torch
from torch.nn import Module
from torch.optim import Optimizer

from fl_bench import Message
from fl_bench.client import Client, PFLClient
from fl_bench.server import Server
from fl_bench.algorithms import CentralizedFL
from fl_bench.data import FastTensorDataLoader
from fl_bench.utils import OptimizerConfigurator


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, local_parameters: List[torch.nn.Parameter]):
        group = None
        for group in self.param_groups:
            for param_p, param_l in zip(group["params"], local_parameters):
                param_p.data = param_p.data - group["lr"] * (
                    param_p.grad.data
                    + group["lamda"] * (param_p.data - param_l.data)
                    + group["mu"] * param_p.data
                )


class PFedMeClient(PFLClient):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 validation_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 lr: float,
                 k: int):
        
        super().__init__(None, train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.hyper_params.update({
            "lr": lr,
            "k": k
        })

    def _receive_model(self) -> None:
        model = self.channel.receive(self, self.server, msg_type="model").payload
        if self.private_model is None:
            self.private_model = deepcopy(model)
            self.model = deepcopy(self.private_model)
        else:
            self.private_model.load_state_dict(model.state_dict())

    def local_train(self, override_local_epochs: int=0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        self._receive_model()
        self.private_model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.private_model)

        lamda = self.optimizer.defaults["lamda"]
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                for _ in range(self.k):
                    self.optimizer.zero_grad()
                    y_hat = self.private_model(X)
                    loss = self.loss_fn(y_hat, y)
                    loss.backward()
                    self.optimizer.step(self.model.parameters())
                
                for param_p, param_l in zip(self.private_model.parameters(), self.model.parameters()):
                    param_l.data = param_l.data - lamda * self.lr * (param_l.data - param_p.data)
            
            self.scheduler.step()     
        self.private_model.load_state_dict(self.model.state_dict())
        self._send_model()
    
    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model), "model", self), self.server)
    

class PFedMeServer(Server):
    def __init__(self, 
                 model: Module,
                 test_data: FastTensorDataLoader,
                 clients: Iterable[Client],
                 weighted: bool=False,
                 beta: float=0.5):
        super().__init__(model, test_data, clients, weighted)
        self.hyper_params.update({
            "beta": beta
        })
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self._get_client_models(eligible)
        weights = self._get_client_weights(eligible)

        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = clients_sd[0][key].clone()
                    continue
                for i, client_sd in enumerate(clients_sd):
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weights[i] * client_sd[key]
                    else:
                        avg_model_sd[key] += weights[i] * client_sd[key]

        for key, param in self.model.named_parameters():
            param.data = (1 - self.hyper_params.beta) * param.data
            param.data += self.hyper_params.beta * avg_model_sd[key] 


class PFedMe(CentralizedFL):
    def get_optimizer_class(self) -> torch.optim.Optimizer:
        return pFedMeOptimizer
    
    def get_client_class(self) -> Client:
        return PFedMeClient

    def get_server_class(self) -> Server:
        return PFedMeServer
