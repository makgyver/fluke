from typing import Callable, Iterable
from copy import deepcopy

import torch
from torch.nn import Module
from torch.optim import Optimizer

from client import Client
from server import Server


import sys; sys.path.append(".")
from fl_bench.algorithms import CentralizedFL
from fl_bench.utils import OptimizerConfigurator
from fl_bench.data import DataSplitter, FastTensorDataLoader


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr: float=0.001, weight_decay: float=0.01):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_controls, client_controls):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss


class ScaffoldClient(Client):
    def __init__(self,
                 dataset: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable, # CHECK ME
                 local_epochs: int=3,
                 seed: int=42):
        assert optimizer_cfg.optimizer == ScaffoldOptimizer, \
            "ScaffoldClient only supports ScaffoldOptimizer"
        super().__init__(dataset, optimizer_cfg, loss_fn, local_epochs, seed)
        self.control = None
        self.delta_c = None
        self.delta_y = None
        self.server_control = None

    def send(self):
        return self.delta_y, self.delta_c
    
    def receive(self, model, server_control):
        if self.model is None:
            self.model = deepcopy(model)
            self.control = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            self.delta_y = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            self.delta_c = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        else:
            self.model.load_state_dict(model.state_dict())
        self.server_control = server_control
    
    def local_train(self, override_local_epochs: int=0, log_interval: int=0):
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        server_model = deepcopy(self.model)
        self.model.train()
        if self.optimizer is None:
            self.optimizer = self.optimizer_cfg(self.model)
        for epoch in range(epochs):
            loss = None
            for i, (X, y) in enumerate(self.dataset):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step(self.server_control, self.control)          
            
                #if log_interval and (i+1) % log_interval == 0:
                #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                #        .format(epoch + 1, self.local_epochs, i + 1, total_step, loss.item()))
        

        for local_model, server_model, delta_y in zip(self.model.parameters(), server_model.parameters(), self.delta_y):
            delta_y.data = local_model.data.detach() - server_model.data.detach()
        
        new_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        coeff = 1. / (self.local_epochs * len(self.dataset) * self.optimizer_cfg.learning_rate())
        for local_control, server_control, new_control, delta_y in zip(self.control, self.server_control, new_controls, self.delta_y):
            new_control.data = local_control.data - server_control.data - delta_y.data * coeff

        for local_control, new_control, delta_c in zip(self.control, new_controls, self.delta_c):
            delta_c.data = new_control.data - local_control.data
            local_control.data = new_control.data
        
        # client_w = {}
        # for k, v in self.model.named_parameters():
        #     client_w[k] = v.data.clone()

        # client_c = {}
        # for k, v in self.control.items():
        #     client_c[k] = v.data.clone()

        # self.delta_y = {}
        # self.delta_c = {}
        # for k, server_w in server_model.named_parameters():
        #     local_steps = self.local_epochs * len(self.dataset)
        #     self.control[k] -= self.server_control[k] + (server_w.data - client_w[k]) / (local_steps * self.optimizer_cfg.learning_rate())
        #     self.delta_y[k] = client_w[k] - server_w.data
        #     self.delta_c[k] = self.control[k] - client_c[k]

        # del server_model
        # del client_c
        # del client_w
        # gc.collect()


class ScaffoldServer(Server):
    def __init__(self,
                 model: Module,
                 clients: Iterable[Client],
                 global_step: float=1.,
                 elegibility_percentage: float=0.5, 
                 seed: int=42):
        super().__init__(model, clients, elegibility_percentage, seed)
        self.control = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.global_step = global_step

    def broadcast(self, eligible: Iterable[Client]=None) -> Iterable[Client]:
        eligible = eligible if eligible is not None else self.clients
        for client in eligible:
            client.receive(deepcopy(self.model), self.control)
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        with torch.no_grad():
            delta_y = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            delta_c = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

            # tot_examples = sum([len(client.n_examples) for client in eligible])
            for client in eligible:
                cl_delta_y, cl_delta_c = client.send()
                for client_delta_c, client_delta_y, server_delta_c, server_delta_y in zip(cl_delta_c, cl_delta_y, delta_c, delta_y):
                    server_delta_y.data = server_delta_y.data + client_delta_y.data
                    server_delta_c.data = server_delta_c.data + client_delta_c.data
                
            for server_delta_c, server_delta_y in zip(delta_c, delta_y):
                server_delta_y.data = server_delta_y.data / len(eligible) #* (eligible[i].n_examples / tot_examples)
                server_delta_c.data = server_delta_c.data / self.n_clients

            for param, server_control, server_delta_y, server_delta_c in zip(self.model.parameters(), self.control, delta_y, delta_c):
                param.data = param.data + self.global_step * server_delta_y
                server_control.data = server_control.data + server_delta_c.data



class SCAFFOLD(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 loss_fn: Callable,
                 elegibility_percentage: float=0.5,
                 seed: int=42):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage,
                         seed)
    
    def init_parties(self, data_splitter: DataSplitter, global_step: float, callback: Callable=None):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.clients = [ScaffoldClient(dataset=data_splitter.client_loader[i], 
                                        optimizer_cfg=self.optimizer_cfg, 
                                        loss_fn=self.loss_fn, 
                                        local_epochs=self.n_epochs,
                                        seed=self.seed) for i in range(self.n_clients)]

        self.server = ScaffoldServer(self.model, 
                                     self.clients, 
                                     global_step, 
                                     self.elegibility_percentage, 
                                     seed=self.seed)
        self.server.register_callback(callback)
