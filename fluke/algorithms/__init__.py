"""This module contains (as submodules) the implementation of several the federated learning
algorithms."""
from __future__ import annotations

import os
import sys
import warnings
from copy import deepcopy
from typing import Any, Iterable, Union
import uuid

import torch

sys.path.append(".")
sys.path.append("..")

from .. import DDict, GlobalSettings  # NOQA
from ..client import Client, PFLClient  # NOQA
from ..data import DataSplitter, FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, get_loss, get_model, ServerObserver  # NOQA

__all__ = [
    'CentralizedFL',
    'PersonalizedFL',
    'apfl',
    'ccvr',
    'ditto',
    'dpfedavg',
    'fedala',
    'fedamp',
    'fedavg',
    'fedavgm',
    'fedaws',
    'fedbabu',
    'fedbn',
    'feddyn',
    'fedexp',
    'fedhp',
    'fedlc',
    'fednh',
    'fednova',
    'fedopt',
    'fedper',
    'fedproto',
    'fedprox',
    'fedrep',
    'fedrod',
    'fedrs',
    'fedsam',
    'fedsgd',
    'kafe',
    'lg_fedavg',
    'moon',
    'per_fedavg',
    'pfedme',
    'scaffold',
    'superfed'
]


class CentralizedFL(ServerObserver):
    """Centralized Federated Learning algorithm.
    This class is a generic implementation of a centralized federated learning algorithm that
    follows the Federated Averaging workflow. This class represents the entry point to the
    federated learning algorithm.
    Each new algorithm should inherit from this class and implement the specific logic of the
    algorithm. The main components of the algorithm are:

    - ``Clients``: Each client should implement :class:`fluke.client.Client` class and the specific
      specialization must be defined in the :meth:`get_client_class` method. The initialization of
      the clients is done in the :meth:`init_clients` method.
    - ``Server``: The server is the entity that coordinates the training process. It should
      implement the :class:`fluke.server.Server` class and the specific specialization must be
      defined in the :meth:`get_server_class` method. The initialization of the server is done in
      the :meth:`init_server` method.
    - ``Optimizer``: The optimizer used by the clients. The default optimizer class is defined in
      the :meth:`get_optimizer_class` method.

    To run the algorithm, the :meth:`run` method should be called with the number of rounds and the
    percentage of eligible clients. This method will call the :meth:`Server.fit` method which will
    orchestrate the training process.

    Args:
        n_clients (int): Number of clients.
        data_splitter (DataSplitter): Data splitter object.
        hyper_params (DDict): Hyperparameters of the algorithm. This set of hyperparameteers should
          be divided in two parts: the client hyperparameters and the server hyperparameters.

    See Also:
        - :class:`fluke.client.Client`
        - :class:`fluke.server.Server`
        - :class:`PersonalizedFL`
    """

    def __init__(self,
                 n_clients: int,
                 data_splitter: DataSplitter,
                 hyper_params: DDict | dict[str, Any],
                 **kwargs: dict[str, Any]):
        self._id = uuid.uuid4()
        self.hyper_params = hyper_params if isinstance(hyper_params, DDict) else DDict(hyper_params)
        self.n_clients = n_clients
        (clients_tr_data, clients_te_data), server_data = \
            data_splitter.assign(n_clients, hyper_params.client.batch_size)
        # Federated model
        model = get_model(mname=hyper_params.model,
                          **hyper_params.net_args if 'net_args' in hyper_params else {}
                          ) if isinstance(hyper_params.model, str) else hyper_params.model
        self.clients = None
        self.server = None
        self.init_clients(clients_tr_data, clients_te_data, hyper_params.client)
        self.init_server(model, server_data, hyper_params.server)

        for client in self.clients:
            client.set_server(self.server)

    @property
    def id(self) -> str:
        """Get the unique identifier of this instance of the algorithm.

        Returns:
            str: Unique identifier of the instance of the algorithm.
        """
        return str(self._id)

    def can_override_optimizer(self) -> bool:
        """Return whether the optimizer can be changed user-side.
        Generally, the optimizer can be configured by the user. However, in some cases, the
        algorithm may require a specific optimizer and the user should not be able to change it.

        Returns:
            bool: Whether the optimizer can be changed user-side.
        """
        return True

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
                     clients_tr_data: list[FastDataLoader],
                     clients_te_data: list[FastDataLoader],
                     config: DDict) -> None:
        """Initialize the clients.

        Args:
            clients_tr_data (list[FastDataLoader]): List of training data loaders, one for
               each client.
            clients_te_data (list[FastDataLoader]): List of test data loaders, one for
               each client. The test data loaders can be ``None``.
            config (DDict): Configuration of the clients.

        Important:
            For more deatils about the configuration of the clients, see the
            :ref:`configuration <configuration>` page.

        See Also:
            :class:`fluke.client.Client`
        """

        if "name" not in config.optimizer or not self.can_override_optimizer():
            config.optimizer.name = self.get_optimizer_class()

        if not self.can_override_optimizer() and \
                config.optimizer.name != self.get_optimizer_class().__name__:
            warnings.warn(f"The algorithm does not support the optimizer {config.optimizer.name}. "
                          f"Using {self.get_optimizer_class().__name__} instead.")

        optimizer_cfg = OptimizerConfigurator(optimizer_cfg=config.optimizer,
                                              scheduler_cfg=config.scheduler)
        self.loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss()
        self.clients = [
            self.get_client_class()(
                index=i,
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=deepcopy(self.loss),
                **config.exclude('optimizer', 'loss', 'batch_size', 'scheduler')
            )
            for i in range(self.n_clients)]

    def init_server(self, model: Any, data: FastDataLoader, config: DDict):
        """Initailize the server.

        Args:
            model (Any): The global model.
            data (FastDataLoader): The server-side test set.
            config (DDict): Configuration of the server.
        """
        self.server: Server = self.get_server_class()(model, data, self.clients, **config)
        if GlobalSettings().get_save_options()[0] is not None:
            self.server.attach(self)

    def set_callbacks(self, callbacks: Union[callable, Iterable[callable]]):
        """Set the callbacks.

        Args:
            callbacks (Union[callable, Iterable[callable]]): Callbacks to attach to the algorithm.
        """
        self.server.attach(callbacks)
        self.server.channel.attach(callbacks)
        for client in self.clients:
            client.attach(callbacks)

    def run(self,
            n_rounds: int,
            eligible_perc: float,
            finalize: bool = True,
            **kwargs: dict[str, Any]):
        """Run the federated algorithm.
        This method will call the :meth:`Server.fit` method which will orchestrate the training
        process.

        Args:
            n_rounds (int): Number of rounds.
            eligible_perc (float): Percentage of eligible clients.
            finalize (bool, optional): Whether to finalize the training process.
              Defaults to ``True``.
            **kwargs (dict[str, Any]): Additional keyword arguments.
        """
        self.server.fit(n_rounds=n_rounds, eligible_perc=eligible_perc, finalize=finalize, **kwargs)

    def __str__(self) -> str:
        algo_hp = ",\n\t".join(
            [f"{h}={str(v)}"
             for h, v in self.hyper_params.items() if h not in ['client', 'server']]
        )
        algo_hp = f"\n\t{algo_hp}," if algo_hp else ""

        if self.clients is None:
            client_str = "Client?"
        else:
            client_str = str(self.clients[0]).replace("[0]", f"[0-{self.n_clients-1}]")

        if self.server is None:
            server_str = "Server?"
        else:
            server_str = str(self.server)

        return f"{self.__class__.__name__}[{self._id}]" + \
            f"({algo_hp}\n\t{client_str},\n\t{server_str}\n)"

    def __repr__(self) -> str:
        return str(self)

    def save(self, path: str, global_only: bool = False, round: int | None = None) -> str:
        """Save the algorithm state into files in the specified directory.

        Note:
            To avoid overwriting previous saved states, the folder name will be suffixed with the
            unique (randomly generated) identifier of the algorithm.

        Args:
            path (str): Path to the folder where the algorithm state will be saved.
            global_only (bool, optional): Whether to save only the global model. Defaults to
                ``False``.
            round (int, optional): Round number. Defaults to ``None``.

        Returns:
            str: Path to the folder where the algorithm state was saved.
        """
        path = f"{path}_{self._id}"
        if not os.path.exists(path):
            os.makedirs(path)

        prefix = f"r{str(round).zfill(4)}_" if round is not None else ""
        self.server.save(os.path.join(path, f"{prefix}server.pth"))
        if not global_only:
            for i, client in enumerate(self.clients):
                client.save(os.path.join(path, f"{prefix}client_{i}.pth"))

        return path

    def load(self, path: str, round: int | None = None) -> None:
        """Load the algorithm state from the specified folder

        Args:
            path (str): Path to the folder where the algorithm state is saved.
            round (int, optional): Round number. Defaults to ``None``.
        """
        if round is not None:
            prefix = f"r{str(round).zfill(4)}_"
        else:
            prefix = ""
            # search in path for the last round
            max_round = -1
            for f in os.listdir(path):
                if f.startswith("r") and f.endswith("_server.pth"):
                    round = int(f[1:5])
                    max_round = max(max_round, round)
            if max_round != -1:
                prefix = f"r{str(max_round).zfill(4)}_"

        self.server.load(os.path.join(path, f"{prefix}server.pth"))
        for i, client in enumerate(self.clients):
            client.model = deepcopy(self.server.model)
            client.load(os.path.join(path, f"{prefix}client_{i}.pth"))

    # ServerObserver methods
    def end_round(self, round: int) -> None:
        path, freq, g_only = GlobalSettings().get_save_options()
        if round % freq == 0:
            self.save(path, g_only, round)

    # ServerObserver methods
    def finished(self, round: int) -> None:
        path, freq, g_only = GlobalSettings().get_save_options()
        if freq == -1:
            self.save(path, g_only, round-1)


class PersonalizedFL(CentralizedFL):
    """Personalized Federated Learning algorithm. This class is a simple extension of the
    :class:`CentralizedFL` class where the clients are expected to implement the
    :class:`fluke.client.PFLClient` class (see :meth:`get_client_class`). The main difference with
    respect to the :class:`CentralizedFL` class is that the clients initialization requires a model
    that is the personalized model of the client.

    Important:
        Differently from :class:`CentralizedFL`, which is actually the FedAvg algorithm, the
        :class:`PersonalizedFL` class must not be used as is because it is a generic implementation
        of a personalized federated learning algorithm. The subclasses of this class should
        implement the specific logic of the personalized federated learning algorithm.

    See Also:
        - :class:`CentralizedFL`
        - :class:`fluke.client.PFLClient`
    """

    def get_client_class(self) -> PFLClient:
        return PFLClient

    def init_clients(self,
                     clients_tr_data: list[FastDataLoader],
                     clients_te_data: list[FastDataLoader],
                     config: DDict) -> None:

        if isinstance(config.model, str):
            model = get_model(mname=config.model, **config.net_args if 'net_args' in config else {})
        elif isinstance(config.model, torch.nn.Module):
            model = config.model
        else:
            raise ValueError("Invalid model configuration. \
                             It should be a string or a torch.nn.Module")

        if not self.can_override_optimizer() and \
                config.optimizer.name != self.get_optimizer_class().__name__:
            opt_name = "SGD" if config.optimizer.name is None else config.optimizer.name
            warnings.warn(f"The algorithm does not support the optimizer {opt_name}. "
                          f"Using {self.get_optimizer_class().__name__} instead.")

        if "name" not in config.optimizer or not self.can_override_optimizer():
            config.optimizer.name = self.get_optimizer_class()

        optimizer_cfg = OptimizerConfigurator(optimizer_cfg=config.optimizer,
                                              scheduler_cfg=config.scheduler)
        self.loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss()
        self.clients = [
            self.get_client_class()(
                index=i,
                model=deepcopy(model),
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=deepcopy(self.loss),
                **config.exclude('optimizer', 'loss', 'batch_size', 'model', 'scheduler')
            )
            for i in range(self.n_clients)]
