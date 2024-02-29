from collections import OrderedDict
from copy import deepcopy
import random
from typing import Callable, Iterable, Sequence, Union, Optional, Any

import torch
from torch.nn import Module, MSELoss

import sys

from fl_bench import GlobalSettings, Message
from fl_bench.evaluation import ClassificationEval; sys.path.append(".")
from rich.progress import Progress
import torch.nn.functional as F
from torch import nn

from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench.utils import DDict, OptimizerConfigurator, get_loss, get_model
from fl_bench.algorithms import CentralizedFL
from fl_bench.net import EDModule

# FEDerated DISEntanglment Learning
class FedDiselClient(Client):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 model: nn.Module,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):
        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.model = model

    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.fed_E), "model", self), self.server)

    def _receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.fed_E.load_state_dict(msg.payload.state_dict())


class FedDisel(CentralizedFL):
    
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict):
        optimizer_cfg = OptimizerConfigurator(self.get_optimizer_class(), 
                                              lr=config.optimizer.lr, 
                                              scheduler_kwargs=config.optimizer.scheduler_kwargs)
        self.loss = get_loss(config.loss)
        model = get_model(mname=config.model).to(GlobalSettings().get_device())
        
        self.clients = [FedDiselClient(train_set=clients_tr_data[i],    
                                       model=model,                    
                                        optimizer_cfg=optimizer_cfg, 
                                        loss_fn=self.loss, 
                                        validation_set=clients_te_data[i],
                                        local_epochs=config.n_epochs) for i in range(self.n_clients)]

    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = Server(model, None, self.clients, **config)