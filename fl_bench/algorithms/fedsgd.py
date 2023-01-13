from typing import Callable

from torch.nn import Module

import sys; sys.path.append(".")
from fl_bench.data import DataSplitter
from fl_bench.algorithms import CentralizedFL
from fl_bench.utils import OptimizerConfigurator


class FedSGD(CentralizedFL):
    """Federated SGD algorithm.

    Parameters
    ----------
    n_clients : int
        Number of clients.
    n_rounds : int
        Number of communication rounds.
    optimizer_cfg : OptimizerConfigurator
        Optimizer configurator for the clients.
    model : Module
        Model to be trained.
    loss_fn : Callable
        Loss function.
    eligibility_percentage : float, optional
        Percentage of clients to be selected for each communication round, by default 0.5.
    """
    def __init__(self,
                 n_clients: int,
                 n_rounds: int,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 loss_fn: Callable,
                 eligibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         1,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         eligibility_percentage)
    
    def init_parties(self, data_splitter: DataSplitter, callback: Callable = None) -> None:
        assert data_splitter.batch_size == 0, \
               "Batch size must be 0 (i.e., the full local dataset is the batch) for FedSGD"
        return super().init_parties(data_splitter, callback)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds}," + \
               f"P={self.eligibility_percentage},{self.optimizer_cfg})"