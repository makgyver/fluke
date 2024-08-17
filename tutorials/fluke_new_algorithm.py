from fluke.algorithms import CentralizedFL
from typing import Iterable
from fluke.client import Client
from fluke.server import Server
from fluke.data import DataSplitter
from fluke.data.datasets import Datasets
from fluke import DDict
from fluke.utils.log import Log
import numpy as np


class MyServer(Server):

    # we override the aggregate method to implement our aggregation strategy
    def aggregate(self, eligible: Iterable[Client]) -> None:
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


dataset = Datasets.get("mnist", path="./data")
splitter = DataSplitter(dataset=dataset,
                        distribution="iid")

client_hp = DDict(
    batch_size=10,
    local_epochs=5,
    loss="CrossEntropyLoss",
    optimizer=DDict(
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001),
    scheduler=DDict(
        gamma=1,
        step_size=1)
)

# we put together the hyperparameters for the algorithm
hyperparams = DDict(client=client_hp,
                    server=DDict(weighted=True),
                    model="MNIST_2NN")

algorithm = MyFLAlgorithm(n_clients=10,  # 10 clients in the federation
                          data_splitter=splitter,
                          hyper_params=hyperparams)

algorithm.set_callbacks(Log())
algorithm.run(n_rounds=10, eligible_perc=0.5)
