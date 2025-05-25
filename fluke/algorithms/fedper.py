"""Implementation of the [FedPer19]_ algorithm.

References:
    .. [FedPer19] Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and
       Sunav Choudhary. Federated learning with personalization layers.
       In arXiv (2019). URL:https://arxiv.org/abs/1912.00818
"""

import sys
from copy import deepcopy
from typing import Sequence

import torch

sys.path.append(".")
sys.path.append("..")

from ..algorithms import PersonalizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..nets import EncoderGlobalHeadLocalNet, EncoderHeadNet  # NOQA
from ..server import Server  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA


# https://arxiv.org/abs/1912.00818
class FedPerClient(Client):

    def __init__(
        self,
        index: int,
        model: EncoderHeadNet,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: torch.nn.Module,
        local_epochs: int = 3,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        **kwargs,
    ):
        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss_fn,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            **kwargs,
        )
        self.model = EncoderGlobalHeadLocalNet(model)
        self._save_to_cache()

    def send_model(self) -> None:
        self.channel.send(
            Message(deepcopy(self.model.get_global()), "model", self.index, inmemory=True),
            "server",
        )

    def receive_model(self) -> None:
        msg = self.channel.receive(self.index, "server", msg_type="model")
        safe_load_state_dict(self.model.get_global(), msg.payload.state_dict())


class FedPerServer(Server):

    def __init__(
        self,
        model: torch.nn.Module,
        test_set: FastDataLoader,  # not used
        clients: Sequence[Client],
        weighted: bool = False,
    ):
        super().__init__(model=model, test_set=None, clients=clients, weighted=weighted)


class FedPer(PersonalizedFL):

    def get_client_class(self) -> type[Client]:
        return FedPerClient

    def get_server_class(self) -> type[Server]:
        return FedPerServer
