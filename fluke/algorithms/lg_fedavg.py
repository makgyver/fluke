"""Implementation of the [LG-FedAVG20]_ algorithm.

References:
    .. [LG-FedAVG20] Paul Pu Liang, Terrance Liu, Liu Ziyin, Nicholas B. Allen, Randy P. Auerbach,
       David Brent, Ruslan Salakhutdinov, Louis-Philippe Morency. Think Locally, Act Globally:
       Federated Learning with Local and Global Representations. In arXiv (2020).
       URL: https://arxiv.org/abs/2001.01523
"""

import sys
from typing import Sequence

# from torch.nn import CrossEntropyLoss
from torch.nn.modules import Module

sys.path.append(".")
sys.path.append("..")

from ..algorithms import PersonalizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..nets import EncoderHeadNet, HeadGlobalEncoderLocalNet  # NOQA
from ..server import Server  # NOQA
from ..utils import get_model  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA

__all__ = ["LGFedAVGClient", "LGFedAVGServer", "LGFedAVG"]

# The implementation is almost identical to FedPerClient
# The difference lies in the part of the network that is local and the part that is global.
# One is the inverse of the other.


class LGFedAVGClient(Client):

    def __init__(
        self,
        index: int,
        model: EncoderHeadNet,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,  # In the paper it is fixed to CrossEntropyLoss
        local_epochs: int = 3,
        fine_tuning_epochs: int = 0,
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
            **kwargs,
        )
        self.model = HeadGlobalEncoderLocalNet(model)
        self._save_to_cache()

    def send_model(self) -> None:
        self.channel.send(
            Message(self.model.get_global(), "model", self.index, inmemory=True),
            "server",
        )

    def receive_model(self) -> None:
        msg = self.channel.receive(self.index, "server", msg_type="model")
        safe_load_state_dict(self.model.get_global(), msg.payload.state_dict())


class LGFedAVGServer(Server):

    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader,  # not used
        clients: Sequence[Client],
        weighted: bool = False,
    ):
        super().__init__(model=model, test_set=None, clients=clients, weighted=weighted)


class LGFedAVG(PersonalizedFL):

    def get_client_class(self) -> type[Client]:
        return LGFedAVGClient

    def get_server_class(self) -> type[Server]:
        return LGFedAVGServer
