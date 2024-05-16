from torch.nn import CrossEntropyLoss
from torch.nn.modules import Module
from typing import Any, Callable, Sequence
import sys
sys.path.append(".")
sys.path.append("..")

from ..utils import OptimizerConfigurator  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from ..data import FastDataLoader  # NOQA
from ..client import PFLClient  # NOQA
from ..algorithms import PersonalizedFL  # NOQA
from ..server import Server  # NOQA
from ..comm import Message  # NOQA
from ..nets import EncoderHeadNet, HeadGlobalEncoderLocalNet  # NOQA


# The implementation is almost identical to FedPerClient
# The difference lies in the part of the network that is local and the part that is global.
# One is the inverse of the other.
class LGFedAVGClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable[..., Any],  # not used because fixed to CrossEntropyLoss
                 local_epochs: int = 3):
        super().__init__(index, HeadGlobalEncoderLocalNet(model),
                         train_set, test_set, optimizer_cfg, CrossEntropyLoss(), local_epochs)

    def send_model(self):
        self.channel.send(Message(self.model.get_global(), "model", self), self.server)

    def receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model  # personalized_model and model are the same
        msg = self.channel.receive(self, self.server, msg_type="model")
        safe_load_state_dict(self.model.get_global(), msg.payload.state_dict())


class LGFedAVGServer(Server):

    def __init__(self,
                 model: Module,
                 test_data: FastDataLoader,
                 clients: Sequence[PFLClient],
                 eval_every: int = 1,
                 weighted: bool = False):
        super().__init__(model, None, clients, eval_every, weighted)


class LGFedAVG(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return LGFedAVGClient

    def get_server_class(self) -> Server:
        return LGFedAVGServer
