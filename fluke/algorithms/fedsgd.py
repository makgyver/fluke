"""Implementation of the [FedSGD17]_ algorithm.

References:
    .. [FedSGD17] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera
       y Arcas. Communication-efficient learning of deep networks from decentralized data.
       In AISTATS (2017). URL: https://arxiv.org/abs/1602.05629
"""
import sys

import torch

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA

__all__ = [
    "FedSGDClient",
    "FedSGD"
]


class FedSGDClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 fine_tuning_epochs: int = 0,
                 **kwargs):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, **kwargs)
        self.train_set.single_batch = True
        self.train_set.shuffle = True
        self.hyper_params.local_epochs = 1


class FedSGD(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return FedSGDClient
