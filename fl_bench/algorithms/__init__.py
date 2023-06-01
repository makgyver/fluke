from typing import Callable, Union, Any, Iterable

import torch
from enum import Enum

from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench import GlobalSettings
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench.utils import DDict, OptimizerConfigurator, get_loss, get_model


class CentralizedFL():
    def __init__(self, 
                 n_clients: int,
                 data_splitter: DataSplitter, 
                 hyperparameters: DDict):
        self.hyperparameters = hyperparameters
        self.n_clients = n_clients
        (clients_tr_data, clients_te_data), server_data = data_splitter.assign(n_clients, 
                                                                               hyperparameters.client.batch_size)
        model = get_model(
                mname=hyperparameters.model,
                input_size=data_splitter.num_features(), 
                num_classes=data_splitter.num_classes()
            ).to(GlobalSettings().get_device())
        self.init_clients(clients_tr_data, clients_te_data, hyperparameters.client)
        self.init_server(model, server_data, hyperparameters.server)

    def get_optimizer_class(self) -> torch.optim.Optimizer:
        return torch.optim.SGD
    
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict):
        optimizer_cfg = OptimizerConfigurator(self.get_optimizer_class(), 
                                              lr=config.optimizer.lr, 
                                              scheduler_kwargs=config.optimizer.scheduler_kwargs)
        self.loss = get_loss(config.loss)
        self.clients = [Client(train_set=clients_tr_data[i],  
                               optimizer_cfg=optimizer_cfg, 
                               loss_fn=self.loss, 
                               validation_set=clients_te_data[i],
                               local_epochs=config.n_epochs) for i in range(self.n_clients)]

    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = Server(model, data, self.clients, **config)
    
    def set_callbacks(self, callbacks: Union[Callable, Iterable[Callable]]):
        self.server.attach(callbacks)
        self.server.channel.attach(callbacks)
        
    def run(self, n_rounds: int, eligible_perc: float):
        self.server.init()
        self.server.fit(n_rounds=n_rounds, eligible_perc=eligible_perc)
    
    def __str__(self) -> str:
        algo_hp = ",\n\t".join([f"{h}={str(v)}" for h,v in self.hyperparameters.items() if h not in ['client', 'server']])
        return f"{self.__class__.__name__}(\n\t{algo_hp},\n\t{self.clients[0]},\n\t{self.server}\n)"
    
    def __repr__(self) -> str:
        return self.__str__()

    def activate_checkpoint(self, path: str):
        self.server.checkpoint_path = path
    
    def load_checkpoint(self, path: str):
        self.server.load(path)


from .fedavg import FedAVG
from .fedavgm import FedAVGM
from .fedsgd import FedSGD
from .fedprox import FedProx
from .scaffold import SCAFFOLD
# from .flhalf import FLHalf
from .fedbn import FedBN
from .fedopt import FedOpt
from .moon import MOON
from .fednova import FedNova
from .fedexp import FedExP
from .pfedme import PFedMe

# BOOSTING ALGORITHMS
from .adaboostf import AdaboostF
from .adaboostf2 import AdaboostF2
from .distboostf import DistboostF
from .preweakf import PreweakF
from .preweakf2 import PreweakF2

class FedAlgorithmsEnum(Enum):
    FEDAVG = 'fedavg'
    FEDAGM = 'fedavgm'
    FEDSGD = 'fedsgd'
    FEDPROX = 'fedprox'
    SCAFFOLD = 'scaffold'
    # FLHALF = 'flhalf'
    FEDBN = 'fedbn'
    FEDOPT = 'fedopt'
    MOON = 'moon'
    FEDNOVA = 'fednova'
    FEDEXP = 'fedexp'
    PEFEDME = 'pfedme'

    def algorithm(self):
        algos = {
            'fedavg': FedAVG,
            'fedavgm': FedAVGM,
            'fedsgd': FedSGD,
            'fedprox': FedProx,
            'scaffold': SCAFFOLD,
            # 'flhalf': FLHalf,
            'fedbn': FedBN,
            'fedopt': FedOpt,
            'moon': MOON,
            'fednova': FedNova,
            'fedexp': FedExP,
            'pfedme': PFedMe
        }

        return algos[self.value]
    
class FedAdaboostAlgorithmsEnum(Enum):
    ADABOOSTF = 'adaboostf'
    ADABOOSTF2 = 'adaboostf2'
    DISTBOOSTF = 'distboostf'
    PREWEAKF = 'preweakf'
    PREWEAKF2 = 'preweakf2'

    def algorithm(self):
        algos = {
            'adaboostf': AdaboostF,
            'adaboostf2': AdaboostF2,
            'distboostf': DistboostF,
            'preweakf': PreweakF,
            'preweakf2': PreweakF2
        }

        return algos[self.value]