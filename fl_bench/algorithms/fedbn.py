import sys; sys.path.append(".")

import torch
from copy import deepcopy
from typing import Iterable, Any
from collections import OrderedDict

from fl_bench import Message
from fl_bench.server import Server
from fl_bench.client import Client
from fl_bench.algorithms import CentralizedFL
from fl_bench.data import FastTensorDataLoader
from fl_bench.utils import DDict, OptimizerConfigurator, get_loss


class FedBNClient(Client):

    def receive(self, message: Message) -> None:
        if message.msg_type == "model":
            model = message.payload
            if self.model is None:
                self.model = deepcopy(model)
            else:
                with torch.no_grad():
                    for key in model.state_dict().keys():
                        if not key.startswith("bn"):
                            self.model.state_dict()[key].data.copy_(model.state_dict()[key])


class FedBN(CentralizedFL):
    
    def get_client_class(self) -> Client:
        return FedBNClient