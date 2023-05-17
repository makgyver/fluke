from abc import ABC
from typing import Callable, Optional, Union, Any, Iterable

from torch.nn import Module

from client import Client
from server import Server

from fl_bench.data import DataSplitter
from fl_bench.utils import OptimizerConfigurator

class FLEnvironment(ABC):

    def init_parties(self, callback: Callable=None, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError


class CentralizedFL(FLEnvironment):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int,
                 model: Module, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable,
                 eligibility_percentage: float=0.5):
        
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.eligibility_percentage = eligibility_percentage
        self.loss_fn = loss_fn
        self.data_assignment = None
    
    def init_clients(self, data_splitter: DataSplitter, **kwargs):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.data_assignment = data_splitter.assignments
        self.clients = [Client(train_set=data_splitter.client_train_loader[i],  
                               optimizer_cfg=self.optimizer_cfg, 
                               loss_fn=self.loss_fn, 
                               validation_set=data_splitter.client_test_loader[i],
                               local_epochs=self.n_epochs) for i in range(self.n_clients)]

    def init_server(self, **kwargs):
        self.server = Server(self.model, self.clients, self.eligibility_percentage, weighted=True)
        

    def init_parties(self, 
                     data_splitter: DataSplitter, 
                     callbacks: Optional[Union[Any, Iterable[Any]]]=None):
        
        self.init_clients(data_splitter)
        self.init_server()
        self.server.attach(callbacks)
        
    def run(self):
        self.server.init()
        self.server.fit(n_rounds=self.n_rounds)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
               f"P={self.eligibility_percentage},{self.optimizer_cfg})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def activate_checkpoint(self, path: str):
        self.server.checkpoint_path = path
    
    def load_checkpoint(self, path: str):
        self.server.load(path)


from .fedavg import FedAVG
from .fedavgm import FedAVGM
from .fedsgd import FedSGD
from .fedprox import FedProx
from .scaffold import SCAFFOLD, ScaffoldOptimizer
from .flhalf import FLHalf
from .fedbn import FedBN
from .fedopt import FedOpt
from .moon import MOON
from .fednova import FedNova, FedNovaOptimizer
from .fedexp import FedExP
from .pfedme import PFedMe, pFedMeOptimizer

from .adaboostf import AdaboostF
from .adaboostf2 import AdaboostF2
from .distboostf import DistboostF
# from .distboostf2 import DistboostF2
from .preweakf import PreweakF
# from .preweakf2 import PreweakF2


from enum import Enum

import torch

class FedAlgorithmsEnum(Enum):
    FEDAVG = 'fedavg'
    FEDAGM = 'fedavgm'
    FEDSGD = 'fedsgd'
    FEDPROX = 'fedprox'
    SCAFFOLD = 'scaffold'
    FLHALF = 'flhalf'
    FEDBN = 'fedbn'
    FEDOPT = 'fedopt'
    MOON = 'moon'
    FEDNOVA = 'fednova'
    FEDEXP = 'fedexp'
    PEFEDME = 'pfedme'

    def optimizer(self) -> torch.optim.Optimizer:
        if self.value == "scaffold":
            return ScaffoldOptimizer
        elif self.value == "fednova":
            return FedNovaOptimizer
        elif self.value == "pfedme":
            return pFedMeOptimizer
        else:
            return torch.optim.SGD

    def algorithm(self):
        algos = {
            'fedavg': FedAVG,
            'fedavgm': FedAVGM,
            'fedsgd': FedSGD,
            'fedprox': FedProx,
            'scaffold': SCAFFOLD,
            'flhalf': FLHalf,
            'fedbn': FedBN,
            'fedopt': FedOpt,
            'moon': MOON,
            'fednova': FedNova,
            'fedexp': FedExP,
            'pfedme': PFedMe
        }

        return algos[self.value]
    
class FedAdaboostAlgorithmsEnum(Enum):
    ADABOOSTF = 'adaboostf'
    ADABOOSTF2 = 'adaboostf2'
    DISTBOOSTF = 'distboostf'
    DISTBOOSTF2 = 'distboostf2'
    PREWEAKF = 'preweakf'
    PREWEAKF2 = 'preweakf2'

    def algorithm(self):
        algos = {
            'adaboostf': AdaboostF,
            'adaboostf2': AdaboostF2,
            'distboostf': DistboostF,
            # 'distboostf2': DistboostF2,
            'preweakf': PreweakF,
            # 'preweakf2': PreweakF2
        }

        return algos[self.value]