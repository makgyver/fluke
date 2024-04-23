from copy import deepcopy
from torch.nn import CrossEntropyLoss
from torch.nn.modules import Module
from typing import Any, Callable, Sequence
import sys
sys.path.append(".")
sys.path.append("..")

from ..utils import OptimizerConfigurator  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from ..data import FastTensorDataLoader  # NOQA
from ..client import PFLClient  # NOQA
from ..algorithms import PersonalizedFL  # NOQA
from ..server import Server  # NOQA
from ..comm import Message  # NOQA


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
                 loss_fn: Callable[..., Any],  # This is ignored!
                 local_epochs: int = 3):
        loss_fn = CrossEntropyLoss()
        super().__init__(index, model, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)

    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.get_global()), "model", self), self.server)

    def _receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model  # personalized_model and model are the same
        msg = self.channel.receive(self, self.server, msg_type="model")
        safe_load_state_dict(self.model.get_global(), msg.payload.state_dict())


class LGFedAVGServer(Server):

    def __init__(self,
                 model: Module,
                 test_data: FastTensorDataLoader,
                 clients: Sequence[PFLClient],
                 eval_every: int = 1,
                 weighted: bool = False):
        super().__init__(model, None, clients, eval_every, weighted)


class LGFedAVG(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return LGFedAVGClient

    def get_server_class(self) -> Server:
        return LGFedAVGServer
