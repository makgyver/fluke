"""Implementation of the DPFedAVG: Differential Privacy Federated Averaging [DPFedAVG2017]_
algorithm.

Note:
    This implementation does not exactly follow the original paper, but it is a simplified version
    that uses the Opacus library to provide differential privacy guarantees.

References:
    .. [DPFedAVG2017] Robin C. Geyer, Tassilo Klein, Moin Nabi.
       Differentially Private Federated Learning: A Client Level Perspective
       In ArXiv (2017). URL: https://arxiv.org/abs/1712.07557
"""
import sys
from typing import Iterable, Any

import torch
from opacus import PrivacyEngine
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from . import CentralizedFL  # NOQA

__all__ = [
    "DPFedAVG",
    "DPFedAVGClient",
    "DPFedAVGServer"
]


class _OpacusModelAdapter(Module):
    """Adapt a model to be compatible with Opacus.
    Opacus encapsultes the model in a way that the parameter names have a prefix
    named "_module". This class is a simple adapter to make the model compatible
    with Opacus.

    Args:
        model (Module): The model to be adapted.
    """

    def __init__(self, model: Module):
        super().__init__()
        self._module = model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._module(*args, **kwargs)


class DPFedAVGClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Module,
                 local_epochs: int = 3,
                 fine_tuning_epochs: int = 0,
                 noise_mul: float = 1.1,
                 max_grad_norm: float = 1.0,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, **kwargs)
        self.hyper_params.update(noise_mul=noise_mul, max_grad_norm=max_grad_norm)

    def _init_private_engine(self) -> None:
        self.privacy_engine = PrivacyEngine()
        self.model.train()
        self.model, self.optimizer, self.train_set = self.privacy_engine.make_private(
            module=self.model._module,
            optimizer=self.optimizer,
            data_loader=self.train_set.asDataLoader(),
            noise_multiplier=self.hyper_params.noise_mul,
            max_grad_norm=self.hyper_params.max_grad_norm,

        )

    def receive_model(self) -> None:
        if self.model is None:
            super().receive_model()
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
            self._init_private_engine()
        else:
            return super().receive_model()


class DPFedAVGServer(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False,
                 lr: float = 1.0,
                 **kwargs: dict[str, Any]):
        super().__init__(model=_OpacusModelAdapter(model),
                         test_set=test_set, clients=clients, weighted=weighted, lr=lr, **kwargs)


class DPFedAVG(CentralizedFL):

    def get_server_class(self) -> Server:
        return DPFedAVGServer

    def get_client_class(self) -> Client:
        return DPFedAVGClient
