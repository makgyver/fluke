from typing import Callable

import torch
from torch.nn import Module
from torch.utils.data import Dataset

from client import Client
from server import Server
from utils import OptimizerConfigurator

from . import CentralizedFL


class FedSGD(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 train_set: Dataset,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 loss_fn: Callable,
                 elegibility_percentage: float=0.5,
                 device: torch.device=torch.device('cpu'),
                 seed: int=42):
        
        super().__init__(n_clients,
                         n_rounds,
                         1,
                         1,
                         train_set,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage,
                         device, 
                         seed)
    
