import sys; sys.path.append(".")

from copy import deepcopy
from typing import Callable

import torch
from torch.nn import CosineSimilarity

from fl_bench.client import Client
from fl_bench.algorithms import CentralizedFL
from fl_bench.data import FastTensorDataLoader
from fl_bench.utils import OptimizerConfigurator, clear_cache


class MOONClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastTensorDataLoader,
                 validation_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 mu: float,
                 tau: float):
        super().__init__(index, train_set, validation_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update({
            "mu": mu,
            "tau": tau
        })
        self.prev_model = None
        self.server_model = None

    def _receive_model(self) -> None:
        model = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.model = deepcopy(model)
            self.prev_model = deepcopy(model)
        else:
            self.prev_model.load_state_dict(self.model.state_dict())
            self.model.load_state_dict(model.state_dict())
        self.server_model = model

    def local_train(self, override_local_epochs: int=0):
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self._receive_model()
        cos = CosineSimilarity(dim=-1).to(self.device)
        self.model.to(self.device)
        self.prev_model.to(self.device)
        self.server_model.to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                #FIXME
                y_hat = self.model(X)
                z_local = self.model.fed_E(X)#, -1)
                loss_sup = self.hyper_params.loss_fn(y_hat, y)

                z_prev = self.prev_model.fed_E(X)#, -1)
                z_global = self.server_model.fed_E(X)#, -1)

                sim_lg = cos(z_local, z_global).reshape(-1, 1) / self.hyper_params.tau
                sim_lp = cos(z_local, z_prev).reshape(-1, 1) / self.hyper_params.tau
                loss_con = -torch.log(torch.exp(sim_lg) / (torch.exp(sim_lg) + torch.exp(sim_lp))).mean()

                loss = loss_sup + self.hyper_params.mu * loss_con
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        
        self.prev_model.to("cpu")
        self.server_model.to("cpu")
        self.model.to("cpu")
        clear_cache()
        self._send_model()
    

class MOON(CentralizedFL):
    
    def get_client_class(self) -> Client:
        return MOONClient