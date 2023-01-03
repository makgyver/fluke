from typing import Callable

from torch.nn import Module

import sys; sys.path.append(".")
from fl_bench.utils import OptimizerConfigurator
from fl_bench.algorithms import CentralizedFL


class FedAVG(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int, 
                 optimizer_cfg: OptimizerConfigurator, 
                 model: Module, 
                 loss_fn: Callable, 
                 elegibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage)
    
