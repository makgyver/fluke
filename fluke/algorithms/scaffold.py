"""Implementation of the [SCAFFOLD20]_ algorithm.

References:
    .. [SCAFFOLD20] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi,
       Sebastian U. Stich, Ananda Theertha Suresh. SCAFFOLD: Stochastic Controlled Averaging for
       Federated Learning. In ICML (2020). URL: https://arxiv.org/abs/1910.06378
"""
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Iterable

import torch
# from torch.optim import Optimizer
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from .. import GlobalSettings  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import safe_load_state_dict, state_dict_zero_like  # NOQA

__all__ = [
    "SCAFFOLDClient",
    "SCAFFOLDServer",
    "SCAFFOLD"
]


class SCAFFOLDClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.control: OrderedDict = None
        self.delta_control: OrderedDict = None
        self.server_control: OrderedDict = None

    def receive_model(self) -> None:
        model = self.channel.receive(self, self.server, msg_type="model").payload
        self.server_control = self.channel.receive(self, self.server, msg_type="control").payload
        if self.model is None:
            self.model = model
            self.control = state_dict_zero_like(model.state_dict())
        else:
            safe_load_state_dict(self.model, model.state_dict())
        self.server_model = deepcopy(model.state_dict())

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.model.to(self.device)
        self.model.train()

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        K = 0
        running_loss = 0.0
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                K += 1
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()

                for n, p in self.model.named_parameters():
                    p.grad.data = p.grad.data + (self.server_control[n].to(self.device) -
                                                 self.control[n].to(self.device))
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        self.model.to("cpu")
        # This happens on CPU!

        with torch.no_grad():
            c_plus = state_dict_zero_like(self.control)
            c_delta = state_dict_zero_like(self.control)
            model_params = self.model.state_dict()
            for key in model_params:
                c_plus[key] = self.control[key] - self.server_control[key] + \
                    (self.server_model[key] - model_params[key]) / \
                    (K * self.scheduler.get_last_lr()[0])

            for key in model_params:
                c_delta[key] = c_plus[key] - self.control[key]

            self.control = deepcopy(c_plus)
            self.delta_control = c_delta

        running_loss /= (epochs * len(self.train_set))
        clear_cache()
        return running_loss

    def send_model(self):
        self.channel.send(Message(self.model, "model", self), self.server)
        self.channel.send(Message(self.delta_control, "control", self), self.server)


class SCAFFOLDServer(Server):
    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = True,
                 global_step: float = 1.,
                 **kwargs: dict[str, Any]):
        super().__init__(model=model,
                         test_set=test_set,
                         clients=clients,
                         weighted=weighted,
                         **kwargs)
        self.device = GlobalSettings().get_device()
        self.control = state_dict_zero_like(self.model.state_dict())
        self.hyper_params.update(global_step=global_step)

    def broadcast_model(self, eligible: Iterable[Client]) -> None:
        self.channel.broadcast(Message(self.model, "model", self), eligible)
        self.channel.broadcast(Message(self.control, "control", self), eligible)

    def _get_client_weights(self, eligible: Iterable[Client]):
        weights = super()._get_client_weights(eligible)
        if self.hyper_params.global_step != 1:
            for c in range(len(eligible)):
                weights[c] = self.hyper_params.global_step * weights[c]
        return weights

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        self.model.to(self.device)

        total_delta = state_dict_zero_like(self.model.state_dict())
        delta_params = [self.channel.receive(self, client, "control").payload
                        for client in eligible]

        for key in total_delta:
            for c_delta_params in delta_params:
                total_delta[key] = total_delta[key] + c_delta_params[key].to(self.device)
            total_delta[key] = total_delta[key] / self.n_clients
            self.control[key] = self.control[key] + total_delta[key].to("cpu")

        return super().aggregate(eligible)


class SCAFFOLD(CentralizedFL):

    def get_client_class(self) -> Client:
        return SCAFFOLDClient

    def get_server_class(self) -> Server:
        return SCAFFOLDServer
