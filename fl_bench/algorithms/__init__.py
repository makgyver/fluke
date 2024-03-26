from __future__ import annotations
import sys
sys.path.append(".")
sys.path.append("..")
from typing import Callable, Union, Any, Iterable

import torch
from enum import Enum

from ..client import Client
from ..server import Server
from ..data import DataSplitter, FastTensorDataLoader
from ..utils import DDict, OptimizerConfigurator, get_loss, get_model

__all__ = [
    'apfl',
    'ditto',
    'fedavg',
    'fedavgm',
    'fedbn',
    'feddisel',
    'feddyn',
    'fedexp',
    'fednova',
    'fedopt',
    'fedper',
    'fedprox',
    'fedrep',
    'fedsgd',
    'flhalf',
    'lg_fedavg',
    'moon',
    'pfedme',
    'scaffold',
    'superfed',
    'per_fedavg'
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
                mname=hyperparameters.model,
                **hyperparameters.net_args if 'net_args' in hyperparameters else {}
            )

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

        optimizer_cfg = OptimizerConfigurator(self.get_optimizer_class(), 
                                              **config.optimizer,
                                              scheduler_kwargs=config.scheduler)
        self.loss = get_loss(config.loss)
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
        self.server = self.get_server_class()(model, data, self.clients, **config)
    
    def set_callbacks(self, callbacks: Union[Callable, Iterable[Callable]]):
        self.server.attach(callbacks)
        self.server.channel.attach(callbacks)
        
    def run(self, n_rounds: int, eligible_perc: float):
        self.server.fit(n_rounds=n_rounds, eligible_perc=eligible_perc)
    
    def __str__(self) -> str:
        algo_hp = ",\n\t".join([f"{h}={str(v)}" for h,v in self.hyperparameters.items() if h not in ['client', 'server']])
        algo_hp = f"\n\t{algo_hp}," if algo_hp else ""
        client_str = str(self.clients[0]).replace("[0]", f"[0-{self.n_clients-1}]")
        return f"{self.__class__.__name__}({algo_hp}\n\t{client_str},\n\t{self.server}\n)"
    
    def __repr__(self) -> str:
        return self.__str__()


class PersonalizedFL(CentralizedFL):
    
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict) -> None:

        model = get_model(mname=config.model)
        optimizer_cfg = OptimizerConfigurator(self.get_optimizer_class(), 
                                              **config.optimizer,
                                              scheduler_kwargs=config.scheduler)
        self.loss = get_loss(config.loss)
        self.clients = [
            self.get_client_class()(
                index=i,
                model=model,
                train_set=clients_tr_data[i],  
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg, 
                loss_fn=self.loss, 
                **config.exclude('optimizer', 'loss', 'batch_size', 'model', 'scheduler')
            ) 
            for i in range(self.n_clients)]


# FEDERATED LEARNING ALGORITHMS
from .fedavg import FedAVG
from .fedavgm import FedAVGM
from .fedsgd import FedSGD
from .fedprox import FedProx
from .scaffold import SCAFFOLD
from .flhalf import FLHalf
from .fedbn import FedBN
from .fedopt import FedOpt
from .moon import MOON
from .fednova import FedNova
from .fedexp import FedExP
from .pfedme import PFedMe
from .feddisel import FedDisel
from .feddyn import FedDyn
from .fedper import FedPer
from .fedrep import FedRep
from .ditto import Ditto
from .apfl import APFL
from .superfed import SuPerFed
from .lg_fedavg import LGFedAVG
from .per_fedavg import PerFedAVG

class FedAlgorithmsEnum(Enum):
    FEDAVG = 'fedavg'
    FEDAGM = 'fedavgm'
    FEDSGD = 'fedsgd'
    FEDPROX = 'fedprox'
    SCAFFOLD = 'scaffold'
    FLHALF = 'flhalf'
    FEDBN = 'fedbn'
    FEDOPT = 'fedopt'
    MOON = 'moon'
    FEDNOVA = 'fednova'
    FEDEXP = 'fedexp'
    PEFEDME = 'pfedme'
    FEDDYN = 'feddyn'
    FEDDISEL = 'feddisel'
    FEDPER = 'fedper'
    FEDREP = 'fedrep'
    DITTO = 'ditto'
    APFL = 'apfl'
    SUPERFED = 'superfed'
    LGFEDAVG = 'lg_fedavg'
    PERFEDAVG = 'per_fedavg'

    @classmethod
    def contains(cls, member: object) -> bool:
        if isinstance(member, str):
            return member in cls._value2member_map_.keys()
        elif isinstance(member, FedAlgorithmsEnum):
            return member.value in cls._member_names_
        
    def algorithm(self):
        algos = {
            'fedavg': FedAVG,
            'fedavgm': FedAVGM,
            'fedsgd': FedSGD,
            'fedprox': FedProx,
            'scaffold': SCAFFOLD,
            'flhalf': FLHalf,
            'fedbn': FedBN,
            'fedopt': FedOpt,
            'moon': MOON,
            'fednova': FedNova,
            'fedexp': FedExP,
            'pfedme': PFedMe,
            'feddyn': FedDyn,
            'feddisel': FedDisel,
            'fedper': FedPer,
            'fedrep': FedRep,
            'ditto': Ditto,
            'apfl': APFL,
            'superfed': SuPerFed,
            'lg_fedavg': LGFedAVG,
            'per_fedavg': PerFedAVG,
        }

        return algos[self.value]
