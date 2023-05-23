import sys
from fl_bench import Message

from fl_bench.data import DataSplitter, FastTensorDataLoader; sys.path.append(".")

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Iterable, List
import torch

from torch.nn import Module
from torch.optim import Optimizer

from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench.utils import OptimizerConfigurator
from fl_bench.algorithms import CentralizedFL


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

class PFedMeClient(Client):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 lr: float,
                 k: int,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):
        
        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.k = k
        self.lr = lr
        self.shared_model = None

    def _receive_model(self) -> None:
        model = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.model = deepcopy(model)
            self.shared_model = deepcopy(self.model)
        else:
            self.model.load_state_dict(model.state_dict())

    def local_train(self, override_local_epochs: int=0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        self._receive_model()
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        lamda = self.optimizer.defaults["lamda"]
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                for _ in range(self.k):
                    self.optimizer.zero_grad()
                    y_hat = self.model(X)
                    loss = self.loss_fn(y_hat, y)
                    loss.backward()
                    self.optimizer.step(self.shared_model.parameters())
                
                for param_p, param_l in zip(self.model.parameters(), self.shared_model.parameters()):
                    param_l.data = param_l.data - lamda * self.lr * (param_l.data - param_p.data)
            
            self.scheduler.step()     
        self.model.load_state_dict(self.shared_model.state_dict())
        validation_results = self.validate()
        self.channel.send(Message(validation_results, "eval", self), self.server)
        self.channel.send(Message(deepcopy(self.shared_model), "model", self), self.server)
    

class PFedMeServer(Server):
    def __init__(self, 
                 model: Module,
                 clients: Iterable[Client],
                 beta: float=0.5,
                 eligibility_percentage: float=0.5, 
                 weighted: bool=False):
        super().__init__(model, clients, eligibility_percentage, weighted)
        self.beta = beta
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self._get_client_models(eligible)

        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0][key])
                    continue
                den = 0
                for i, client_sd in enumerate(clients_sd):
                    weight = 1 if not self.weighted else eligible[i].n_examples
                    den += weight
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weight * client_sd[key]
                    else:
                        avg_model_sd[key] += weight * client_sd[key]
                avg_model_sd[key] /= den
        
        for key, param in self.model.named_parameters():
            param.data = (1 - self.beta) * param.data
            param.data += self.beta * avg_model_sd[key] 

class PFedMe(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int, 
                 optimizer_cfg: OptimizerConfigurator, 
                 model: Module, 
                 beta: float,
                 k: int,
                 client_lr: float,
                 loss_fn: Callable, 
                 eligibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         eligibility_percentage)
        self.beta = beta
        self.k = k
        self.client_lr = client_lr

    def init_server(self, **kwargs):
        self.server = PFedMeServer(self.model, 
                                   self.clients, 
                                   self.beta, 
                                   self.eligibility_percentage, 
                                   weighted=True)
    
    def init_clients(self, data_splitter: DataSplitter, **kwargs):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.clients = [PFedMeClient(train_set=data_splitter.client_train_loader[i], 
                                     k=self.k,
                                     lr=self.client_lr,
                                     optimizer_cfg=self.optimizer_cfg, 
                                     loss_fn=self.loss_fn, 
                                     validation_set=data_splitter.client_test_loader[i],
                                     local_epochs=self.n_epochs) for i in range(self.n_clients)]
