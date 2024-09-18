"""Implementation of the [FedPer19]_ algorithm.

References:
    .. [FedPer19] Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and
       Sunav Choudhary. Federated learning with personalization layers.
       In arXiv (2019). URL:https://arxiv.org/abs/1912.00818
"""
import sys
from copy import deepcopy
from typing import Any, Iterable

import torch

sys.path.append(".")
sys.path.append("..")

from ..algorithms import PersonalizedFL  # NOQA
from ..client import PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..nets import EncoderGlobalHeadLocalNet, EncoderHeadNet  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA


# https://arxiv.org/abs/1912.00818
class FedPerClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, model=EncoderGlobalHeadLocalNet(model),
                         train_set=train_set, test_set=test_set, optimizer_cfg=optimizer_cfg,
                         loss_fn=loss_fn, local_epochs=local_epochs, **kwargs)

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
                 test_set: FastDataLoader,  # not used
                 clients: Iterable[PFLClient],
                 weighted: bool = False):
        super().__init__(model=model, test_set=None, clients=clients, weighted=weighted)


class FedPer(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedPerClient

    def get_server_class(self) -> Server:
        return FedPerServer
