from copy import deepcopy
from typing import Callable, Iterable, Any, Optional, Union

import torch
from torch.nn import Module, CosineSimilarity

import sys
from fl_bench import Message; sys.path.append(".")
from fl_bench.client import Client
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench.utils import DDict, OptimizerConfigurator, get_loss
from fl_bench.algorithms import CentralizedFL


class MOONClient(Client):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 mu: float,
                 tau: float,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):
        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.mu = mu
        self.tau = tau
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
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        self._receive_model()
        cos = CosineSimilarity(dim=-1).to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat, z_local = self.model.forward_(X, -1)
                loss_sup = self.loss_fn(y_hat, y)

                self.prev_model.to(self.device)
                self.server_model.to(self.device)

                _, z_prev = self.prev_model.forward_(X, -1)
                _, z_global = self.server_model.forward_(X, -1)

                sim_lg = cos(z_local, z_global).reshape(-1, 1) / self.tau
                sim_lp = cos(z_local, z_prev).reshape(-1, 1) / self.tau
                loss_con = -torch.log(torch.exp(sim_lg) / (torch.exp(sim_lg) + torch.exp(sim_lp))).mean()

                loss = loss_sup + self.mu * loss_con
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        
        self.prev_model.to("cpu")
        self.server_model.to("cpu")
        self.channel.send(Message(deepcopy(self.model), "model", self), self.server)

class MOON(CentralizedFL):
    
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict):
        optimizer_cfg=OptimizerConfigurator(self.get_optimizer_class(), 
                                            lr=config.optimizer.lr, 
                                            scheduler_kwargs=config.optimizer.scheduler_kwargs)
        self.loss = get_loss(config.loss)
        self.clients = [MOONClient(train_set=clients_tr_data[i],  
                                   mu=config.mu,
                                   tau=config.tau,
                                   optimizer_cfg=optimizer_cfg, 
                                   loss_fn=self.loss, 
                                   validation_set=clients_te_data[i],
                                   local_epochs=config.n_epochs) for i in range(self.n_clients)]