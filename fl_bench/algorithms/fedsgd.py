from typing import Callable

from torch.nn import Module

import sys; sys.path.append(".")
from fl_bench.data import DataSplitter
from fl_bench.algorithms import CentralizedFL
from fl_bench.utils import OptimizerConfigurator


class FedSGD(CentralizedFL):
    def init_parties(self, 
                     n_clients: int,
                     data_splitter: DataSplitter, 
                     hyperparameters: dict):
        # Force single epoch for each client
        hyperparameters.client.n_epochs = 1
        # Force batch size to 0 == full batch
        hyperparameters.client.batch_size = 0
        super().init_parties(n_clients, data_splitter, hyperparameters)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds}," + \
               f"P={self.eligible_perc},{self.optimizer_cfg})"