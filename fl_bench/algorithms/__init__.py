from abc import ABC
from typing import Callable

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from client import Client
from server import Server
from utils import OptimizerConfigurator

__all__ = [
    'CentralizedFL',
    'FLEnvironment',
    'fedavg',
    'fedsgd',
    'scaffold'
    'fedprox',
    'flhalf'
]

class FLEnvironment(ABC):
    pass

class CentralizedFL(FLEnvironment):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int, 
                 batch_size: int, 
                 train_set: Dataset,
                 model: Module, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable,
                 elegibility_percentage: float=0.5,
                 device: torch.device=torch.device('cpu'),
                 seed: int=42):
        
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.train_set = train_set
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.elegibility_percentage = elegibility_percentage
        self.loss_fn = loss_fn
        self.device = device
        self.client_loader = None
        self.test_loader = None

    def prepare_data(self, dataclass: type[Dataset], **kwargs):
        ex_client = self.train_set.data.shape[0] // self.n_clients
        # client_data = [self.train_set.data[ex_client*n: ex_client*(n+1)] for n in range(self.n_clients)]
        # client_label = [self.train_set.targets[ex_client*n: ex_client*(n+1)] for n in range(self.n_clients)]

        # self.client_loader = [DataLoader(dataclass(client_data[c], client_label[c], **kwargs), 
        #                                      batch_size=self.batch_size, 
        #                                      shuffle=True, 
        #                                      num_workers=0) for c in range(self.n_clients)]

        self.client_loader = [DataLoader(dataclass(self.train_set, ex_client*c, ex_client*(c+1), **kwargs), 
                                             batch_size=100, 
                                             shuffle=True, 
                                             num_workers=0) for c in range(self.n_clients)]
    
    def init_parties(self, callback: Callable=None):
        assert self.client_loader is not None, 'You must prepare data before initializing clients'
        self.clients = [Client(dataset=self.client_loader[i], 
                               optimizer_cfg=self.optimizer_cfg, 
                               loss_fn=self.loss_fn, 
                               local_epochs=self.n_epochs,
                               device=self.device,
                               seed=self.seed) for i in range(self.n_clients)]

        self.server = Server(self.model.to(self.device), self.clients, self.elegibility_percentage, seed=self.seed)
        self.server.register_callback(callback)
        
    def run(self, log_interval=0):
        self.server.init()
        self.server.fit(n_rounds=self.n_rounds, log_interval=log_interval)

