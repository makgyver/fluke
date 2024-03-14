import sys;  sys.path.append(".")
from typing import Any, Callable

from torch.nn.modules import Module
from torch.nn import CrossEntropyLoss

from fl_bench.client import PFLClient
from fl_bench.data import FastTensorDataLoader
from fl_bench.utils import OptimizerConfigurator
from copy import deepcopy

from fl_bench import Message
from fl_bench.algorithms import PersonalizedFL


# The implementation is almost identical to FedPerClient
# The difference lies in the part of the network that is local and the part that is global.
# One is the inverse of the other.
class LGFedAVGClient(PFLClient):

    def __init__(self, 
                 model: Module, 
                 train_set: FastTensorDataLoader, 
                 validation_set: FastTensorDataLoader, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable[..., Any], # This is ignored!
                 local_epochs: int = 3):
        super().__init__(model, train_set, validation_set, optimizer_cfg, CrossEntropyLoss, local_epochs)

    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.get_global()), "model", self), self.server)

    def _receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.get_global().load_state_dict(msg.payload.state_dict())

    

class LGFedAVG(PersonalizedFL):
    
    def get_client_class(self) -> PFLClient:
        return LGFedAVGClient
