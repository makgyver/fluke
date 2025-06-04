"""Implementation of the [SCAFFOLD20]_ algorithm.

References:
    .. [SCAFFOLD20] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi,
       Sebastian U. Stich, Ananda Theertha Suresh. SCAFFOLD: Stochastic Controlled Averaging for
       Federated Learning. In ICML (2020). URL: https://arxiv.org/abs/1910.06378
"""

import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Collection, Sequence

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from .. import FlukeENV  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from ..utils.model import safe_load_state_dict, state_dict_zero_like, unwrap  # NOQA

__all__ = ["SCAFFOLDClient", "SCAFFOLDServer", "SCAFFOLD"]


class SCAFFOLDClient(Client):
    def __init__(
        self,
        index: int,
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
        self.control: OrderedDict | None = None
        self.delta_control: OrderedDict | None = None
        self.server_control: OrderedDict | None = None
        self._attr_to_cache.extend(["control", "delta_control", "server_control"])

    def receive_model(self) -> None:
        model = self.channel.receive(self.index, "server", msg_type="model").payload
        self.server_control = self.channel.receive(self.index, "server", msg_type="control").payload
        if self.model is None:
            self.model = model
            if self.control is None:
                self.control = state_dict_zero_like(model.state_dict())
        else:
            safe_load_state_dict(self.model, model.state_dict())
        self.server_model = deepcopy(model.state_dict())

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )
        self.model.to(self.device)
        self.model.train()

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

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

                for n, p in unwrap(self.model).named_parameters():
                    p.grad.data = p.grad.data + (
                        self.server_control[n].to(self.device) - self.control[n].to(self.device)
                    )

                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        self.model.cpu()
        # This happens on CPU!

        with torch.no_grad():
            c_plus = state_dict_zero_like(self.control)
            c_delta = state_dict_zero_like(self.control)
            model_params = unwrap(self.model).state_dict()
            for key in model_params:
                c_plus[key] = (
                    self.control[key]
                    - self.server_control[key]
                    + (self.server_model[key] - model_params[key])
                    / (K * self.scheduler.get_last_lr()[0])
                )

            for key in model_params:
                c_delta[key] = c_plus[key] - self.control[key]

            self.control = deepcopy(c_plus)
            self.delta_control = c_delta

        running_loss /= epochs * len(self.train_set)
        clear_cuda_cache()
        return running_loss

    def send_model(self) -> None:
        self.channel.send(Message(self.model, "model", self.index, inmemory=True), "server")
        self.channel.send(
            Message(self.delta_control, "control", self.index, inmemory=True), "server"
        )


class SCAFFOLDServer(Server):
    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader,
        clients: Sequence[Client],
        weighted: bool = True,
        global_step: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            model=model, test_set=test_set, clients=clients, weighted=weighted, **kwargs
        )
        self.device = FlukeENV().get_device()
        self.control = state_dict_zero_like(self.model.state_dict())
        self.hyper_params.update(global_step=global_step)

    def broadcast_model(self, eligible: Sequence[Client]) -> None:
        self.channel.broadcast(Message(self.model, "model", "server"), [c.index for c in eligible])
        self.channel.broadcast(
            Message(self.control, "control", "server"), [c.index for c in eligible]
        )

    def _get_client_weights(self, eligible: Sequence[Client]) -> list[float]:
        weights = super()._get_client_weights(eligible)
        if self.hyper_params.global_step != 1:
            for c in range(len(eligible)):
                weights[c] = self.hyper_params.global_step * weights[c]
        return weights

    @torch.no_grad()
    def aggregate(self, eligible: Sequence[Client], client_models: Collection[Module]) -> None:
        self.model.to(self.device)

        total_delta = state_dict_zero_like(self.model.state_dict())
        for client in eligible:
            c_delta_params = self.channel.receive("server", client.index, "control").payload
            for key in total_delta:
                total_delta[key] = total_delta[key] + c_delta_params[key].to(self.device)

        for key in total_delta:
            total_delta[key] = total_delta[key] / self.n_clients
            self.control[key] = self.control[key] + total_delta[key].to("cpu")

        self.model.cpu()
        return super().aggregate(eligible, client_models)


class SCAFFOLD(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return SCAFFOLDClient

    def get_server_class(self) -> type[Server]:
        return SCAFFOLDServer
