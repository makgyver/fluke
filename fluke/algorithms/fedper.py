from typing import Sequence, Callable
from copy import deepcopy
import torch
import sys
sys.path.append(".")
sys.path.append("..")

from ..algorithms import PersonalizedFL  # NOQA
from ..server import Server  # NOQA
from ..client import PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from ..nets import EncoderHeadNet, EncoderGlobalHeadLocalNet  # NOQA


# https://arxiv.org/abs/1912.00818
class FedPerClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int = 3):
        super().__init__(index, EncoderGlobalHeadLocalNet(model),
                         train_set, test_set, optimizer_cfg, loss_fn, local_epochs)

    def send_model(self):
        self.channel.send(Message(deepcopy(self.model.get_global()), "model", self), self.server)

    def receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model
        msg = self.channel.receive(self, self.server, msg_type="model")
        safe_load_state_dict(self.model.get_global(), msg.payload.state_dict())


class FedPerServer(Server):

    def __init__(self,
                 model: torch.nn.Module,
                 test_data: FastDataLoader,  # test_data is not used
                 clients: Sequence[PFLClient],
                 eval_every: int = 1,
                 weighted: bool = False):
        super().__init__(model, None, clients, eval_every, weighted)


class FedPer(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedPerClient

    def get_server_class(self) -> Server:
        return FedPerServer
