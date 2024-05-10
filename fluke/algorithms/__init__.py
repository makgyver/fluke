from __future__ import annotations
import torch
from typing import Callable, Union, Any, Iterable
from copy import deepcopy
import sys
sys.path.append(".")
sys.path.append("..")

from .. import DDict  # NOQA
from ..utils import OptimizerConfigurator, get_loss, get_model  # NOQA
from ..data import DataSplitter, FastTensorDataLoader  # NOQA
from ..server import Server  # NOQA
from ..client import Client, PFLClient  # NOQA

__all__ = [
    'apfl',
    'ccvr',
    'ditto',
    'fedamp',
    'fedavg',
    'fedavgm',
    'fedbabu',
    'fedbn',
    'feddyn',
    'fedexp',
    'fedhyperproto',
    'fedlc',
    'fednh',
    'fednova',
    'fedopt',
    'fedper',
    'fedproto',
    'fedprox',
    'fedrep',
    'fedsgd',
    'lg_fedavg',
    'moon',
    'per_fedavg',
    'pfedme',
    'scaffold',
    'superfed'
]


class CentralizedFL():
    """Generic Centralized Federated Learning algorithm.

    This class is a generic implementation of a centralized federated learning algorithm.

    Args:
        n_clients (int): Number of clients.
        data_splitter (DataSplitter): Data splitter object.
        hyperparameters (DDict): Hyperparameters of the algorithm.
    """

    def __init__(self,
                 n_clients: int,
                 data_splitter: DataSplitter,
                 hyperparameters: DDict):
        self.hyperparameters = hyperparameters
        self.n_clients = n_clients
        (clients_tr_data, clients_te_data), server_data = \
            data_splitter.assign(n_clients, hyperparameters.client.batch_size)
        # Federated model
        model = get_model(
            mname=hyperparameters.model
            # **hyperparameters.net_args if 'net_args' in hyperparameters else {}
        ) if isinstance(hyperparameters.model, str) else hyperparameters.model

        self.init_clients(clients_tr_data, clients_te_data, hyperparameters.client)
        self.init_server(model, server_data, hyperparameters.server)

    def get_optimizer_class(self) -> torch.optim.Optimizer:
        """Get the optimizer class.

        Returns:
            torch.optim.Optimizer: Optimizer class.
        """
        return torch.optim.SGD

    def get_client_class(self) -> Client:
        """Get the client class.

        This method should be overriden by the subclasses when a different client class is defined.
        This allows to reuse all the logic of the algorithm and only change the client class.

        Returns:
            Client: Client class.
        """
        return Client

    def get_server_class(self) -> Server:
        """Get the server class.

        This method should be overriden by the subclasses when a different server class is defined.
        This allows to reuse all the logic of the algorithm and only change the server class.

        Returns:
            Server: Server class.
        """
        return Server

    def init_clients(self,
                     clients_tr_data: list[FastTensorDataLoader],
                     clients_te_data: list[FastTensorDataLoader],
                     config: DDict) -> None:
        """Initialize the clients.

        Args:
            clients_tr_data (list[FastTensorDataLoader]): List of training data loaders, one for
               each client.
            clients_te_data (list[FastTensorDataLoader]): List of test data loaders, one for
               each client.
            config (DDict): Configuration of the clients.
        """

        if "name" not in config.optimizer:
            config.optimizer.name = self.get_optimizer_class()
        optimizer_cfg = OptimizerConfigurator(optimizer_cfg=config.optimizer,
                                              scheduler_cfg=config.scheduler)
        self.loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss
        self.clients = [
            self.get_client_class()(
                index=i,
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=self.loss,
                **config.exclude('optimizer', 'loss', 'batch_size', 'scheduler')
            )
            for i in range(self.n_clients)]

    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        """Initailize the server.

        Args:
            model (Any): The global model.
            data (FastTensorDataLoader): The server-side test set.
            config (DDict): Configuration of the server.
        """
        self.server = self.get_server_class()(model, data, self.clients, **config)

    def set_callbacks(self, callbacks: Union[Callable, Iterable[Callable]]):
        """Set the callbacks.

        Args:
            callbacks (Union[Callable, Iterable[Callable]]): Callbacks to attach to the algorithm.
        """
        self.server.attach(callbacks)
        self.server.channel.attach(callbacks)

    def run(self, n_rounds: int, eligible_perc: float):
        self.server.fit(n_rounds=n_rounds, eligible_perc=eligible_perc)

    def __str__(self) -> str:
        algo_hp = ",\n\t".join(
            [f"{h}={str(v)}"
             for h, v in self.hyperparameters.items() if h not in ['client', 'server']]
        )
        algo_hp = f"\n\t{algo_hp}," if algo_hp else ""
        client_str = str(self.clients[0]).replace("[0]", f"[0-{self.n_clients-1}]")
        return f"{self.__class__.__name__}({algo_hp}\n\t{client_str},\n\t{self.server}\n)"

    def __repr__(self) -> str:
        return self.__str__()


class PersonalizedFL(CentralizedFL):

    def get_client_class(self) -> PFLClient:
        return PFLClient

    def init_clients(self,
                     clients_tr_data: list[FastTensorDataLoader],
                     clients_te_data: list[FastTensorDataLoader],
                     config: DDict) -> None:

        model = get_model(mname=config.model) if isinstance(config.model, str) else config.model
        if "name" not in config.optimizer:
            config.optimizer.name = self.get_optimizer_class()
        optimizer_cfg = OptimizerConfigurator(optimizer_cfg=config.optimizer,
                                              scheduler_cfg=config.scheduler)
        self.loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss
        self.clients = [
            self.get_client_class()(
                index=i,
                model=deepcopy(model),
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=self.loss,
                **config.exclude('optimizer', 'loss', 'batch_size', 'model', 'scheduler')
            )
            for i in range(self.n_clients)]
