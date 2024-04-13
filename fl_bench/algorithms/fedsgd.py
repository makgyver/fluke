from ..algorithms import CentralizedFL
from ..data import DataSplitter
import sys
sys.path.append(".")
sys.path.append("..")


class FedSGD(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 data_splitter: DataSplitter,
                 hyperparameters: dict):
        # Force single epoch for each client
        hyperparameters.client.local_epochs = 1
        # Force batch size to 0 == full batch
        hyperparameters.client.batch_size = 0
        super().__init__(n_clients, data_splitter, hyperparameters)
