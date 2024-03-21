import sys
sys.path.append(".")
sys.path.append("..")
from typing import Any, Callable, Sequence

from torch.nn.modules import Module
from torch.nn import CrossEntropyLoss
from copy import deepcopy

from .. import Message
from ..server import Server
from ..algorithms import PersonalizedFL
from ..client import PFLClient
from ..data import FastTensorDataLoader
from ..utils import OptimizerConfigurator


# The implementation is almost identical to FedPerClient
# The difference lies in the part of the network that is local and the part that is global.
# One is the inverse of the other.
class LGFedAVGClient(PFLClient):

    def __init__(self, 
                 index: int,
                 model: Module, 
                 train_set: FastTensorDataLoader, 
                 test_set: FastTensorDataLoader, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable[..., Any], # This is ignored!
                 local_epochs: int = 3):
        super().__init__(index, model, train_set, test_set, optimizer_cfg, CrossEntropyLoss(), local_epochs)

    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.get_global()), "model", self), self.server)

    def _receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model # personalized_model and model are the same
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.get_global().load_state_dict(msg.payload.state_dict())


class LGFedAVGServer(Server):

    def __init__(self,
                 model: Module,
                 test_data: FastTensorDataLoader,
                 clients: Sequence[PFLClient], 
                 weighted: bool=False):
        super().__init__(model, None, clients, weighted)


class LGFedAVG(PersonalizedFL):
    
    def get_client_class(self) -> PFLClient:
        return LGFedAVGClient

    def get_server_class(self) -> Server:
        return LGFedAVGServer