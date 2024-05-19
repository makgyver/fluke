from fluke.algorithms import CentralizedFL
from typing import Sequence
from fluke.client import Client
from fluke.server import Server
import numpy as np


class MyServer(Server):

    # we override the aggregate method to implement our aggregation strategy
    def aggregate(self, eligible: Sequence[Client]) -> None:
        # eligible is a list of clients that participated in the last round
        # here we randomly select only two of them

        selected = np.random.choice(eligible, 2, replace=False)

        # we call the parent class method to aggregate the selected clients
        return super().aggregate(selected)


class MyClient(Client):

    # we override the fit method to implement our training "strategy"
    def fit(self, override_local_epochs: int = 0) -> None:
        # we can override the number of local epochs and call the parent class method
        new_local_epochs = np.random.randint(1, self.hyper_params.local_epochs + 1)
        return super().fit(new_local_epochs)


class MyFLAlgorithm(CentralizedFL):

    def get_client_class(self) -> Client:
        return MyClient

    def get_server_class(self) -> Server:
        return MyServer
