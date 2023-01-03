from typing import Callable

from torch.nn import Module

import sys; sys.path.append(".")
from fl_bench.data import DataSplitter
from fl_bench.algorithms import CentralizedFL
from fl_bench.utils import OptimizerConfigurator


class FedSGD(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 loss_fn: Callable,
                 elegibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         1,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage)
    
    def init_parties(self, data_splitter: DataSplitter, callback: Callable = None):
        assert data_splitter.batch_size == 0, \
               "Batch size must be 0 (i.e., the full local dataset is the batch) for FedSGD"
        return super().init_parties(data_splitter, callback)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds}," + \
               f"P={self.elegibility_percentage})"