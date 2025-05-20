"""Implementation of the [FedRep21]_ algorithm.

References:
    .. [FedRep21] Liam Collins, Hamed Hassani, Aryan Mokhtari, and Sanjay Shakkottai.
       Exploiting shared representations for personalized federated learning. In ICML (2021).
       URL: https://arxiv.org/abs/2102.07078
"""
import sys
from dataclasses import dataclass
from typing import Collection

import torch

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..nets import EncoderGlobalHeadLocalNet, EncoderHeadNet  # NOQA
from ..server import Server  # NOQA
from ..utils import clear_cuda_cache, get_model  # NOQA
from ..utils.model import ModOpt, safe_load_state_dict, unwrap  # NOQA

__all__ = [
    "FedRepClient",
    "FedRepServer",
    "FedRep"
]

# https://arxiv.org/abs/2102.07078


@dataclass
class _ModOpt2(ModOpt):

    pers_optimizer: torch.optim.Optimizer = None
    pers_scheduler: torch.optim.lr_scheduler = None


class FedRepClient(Client):

    def __init__(self,
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 0,
                 tau: int = 3,
                 **kwargs):
        super().__init__(index=index, train_set=train_set,
                         test_set=test_set, optimizer_cfg=optimizer_cfg, loss_fn=loss_fn,
                         local_epochs=local_epochs, fine_tuning_epochs=fine_tuning_epochs,
                         clipping=clipping, **kwargs)
        if isinstance(model, str):
            model = get_model(model)
        self._modopt = _ModOpt2(model=EncoderGlobalHeadLocalNet(model))
        self.hyper_params.update(tau=tau)
        self._save_to_cache()

    @property
    def pers_optimizer(self) -> torch.optim.Optimizer:
        """Optimizers for the personalized part of the model.

        Returns:
            torch.optim.Optimizer: Optimizer for the personalized part of the model.
        """
        return self._modopt.pers_optimizer

    @pers_optimizer.setter
    def pers_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self._modopt.pers_optimizer = optimizer

    @property
    def pers_scheduler(self) -> torch.optim.lr_scheduler:
        """Learning rate scheduler for the personalized part of the model.

        Returns:
            torch.optim.lr_scheduler: Learning rate scheduler for the personalized part of the
                model.
        """
        return self._modopt.pers_scheduler

    @pers_scheduler.setter
    def pers_scheduler(self, scheduler: torch.optim.lr_scheduler) -> None:
        self._modopt.pers_scheduler = scheduler

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (override_local_epochs if override_local_epochs > 0
                       else self.hyper_params.local_epochs)
        self.model.train()
        self.model.to(self.device)

        # update head layers
        for parameter in unwrap(self.model).get_local().parameters():
            parameter.requires_grad = True
        for parameter in unwrap(self.model).get_global().parameters():
            parameter.requires_grad = False

        if self.pers_optimizer is None:
            self.pers_optimizer, self.pers_scheduler = \
                self._optimizer_cfg(unwrap(self.model).get_local())

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.pers_optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self._clip_grads(self.model)
                self.pers_optimizer.step()
                running_loss += loss.item()
            self.pers_scheduler.step()

        # update encoder layers
        for parameter in unwrap(self.model).get_local().parameters():
            parameter.requires_grad = False
        for parameter in unwrap(self.model).get_global().parameters():
            parameter.requires_grad = True

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(unwrap(self.model).get_global())

        for _ in range(self.hyper_params.tau):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.model.cpu()
        clear_cuda_cache()
        return running_loss

    def send_model(self) -> None:
        self.channel.send(Message(self.model.get_global(), "model", self.index, inmemory=True),
                          "server")

    def receive_model(self) -> None:
        server_model = self.channel.receive(self.index, "server", msg_type="model").payload
        if self.model is None:
            self.model = EncoderGlobalHeadLocalNet(server_model)
        safe_load_state_dict(self.model.get_global(), server_model.state_dict())


class FedRepServer(Server):

    def __init__(self,
                 model: torch.nn.Module,
                 test_set: FastDataLoader,  # test_set is not used
                 clients: Collection[Client],
                 weighted: bool = False):
        super().__init__(model=model, test_set=None, clients=clients, weighted=weighted)


class FedRep(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return FedRepClient

    def get_server_class(self) -> type[Server]:
        return FedRepServer
