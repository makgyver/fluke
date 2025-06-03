"""This module contains classes and functions for parallel federated learning using multiple
GPUs."""

__all__ = ["client", "server", "ParallelAlgorithm"]


from copy import deepcopy
from typing import Any, Sequence

from .. import DDict  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import DataSplitter  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils import get_loss  # NOQA
from .client import ParallelClient  # NOQA
from .server import ParallelServer  # NOQA
from .utils import ModelBuilder  # NOQA


class ParallelAlgorithm(CentralizedFL):
    """
    ParallelAlgorithm is a class for parallel federated learning algorithms that use multiple
    GPUs. It inherits from CentralizedFL and initializes clients and server for parallel execution.
    Clients training runs on different GPUs, and the server aggregates their updates.
    """

    def __init__(
        self,
        n_clients: int,
        data_splitter: DataSplitter,
        hyper_params: DDict | dict[str, Any],
        clients: list[ParallelClient] = None,
        server: ParallelServer = None,
        **kwargs,
    ):
        self._builder = ModelBuilder(
            hyper_params.model, **hyper_params.model_args if "model_args" in hyper_params else {}
        )
        super().__init__(
            n_clients=n_clients,
            data_splitter=data_splitter,
            hyper_params=hyper_params,
            clients=clients,
            server=server,
            **kwargs,
        )

    def init_clients(
        self,
        clients_tr_data: list[FastDataLoader],
        clients_te_data: list[FastDataLoader],
        config: DDict,
    ) -> Sequence[ParallelClient]:

        self._fix_opt_cfg(config.optimizer)
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=config.optimizer, scheduler_cfg=config.scheduler
        )
        loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss()
        clients = [
            self.get_client_class()(
                builder=self._builder,
                index=i,
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=deepcopy(loss),
                **config.exclude("optimizer", "loss", "batch_size", "scheduler"),
            )
            for i in range(self.n_clients)
        ]
        return clients

    def get_server_class(self) -> type[ParallelServer]:
        return ParallelServer

    def get_client_class(self) -> type[ParallelClient]:
        return ParallelClient
