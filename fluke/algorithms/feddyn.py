"""Implementation of the FedDyn [FedDyn21]_ algorithm.

References:
    .. [FedDyn21] Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina, Paul Whatmough,
       and Venkatesh Saligrama. Federated Learning with Dynamic Regularization.
       In ICLR (2021). URL: https://openreview.net/pdf?id=B7v4QMR6Z9w
"""
import sys
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from .. import FlukeENV  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cuda_cache  # NOQA
from ..utils.model import aggregate_models, safe_load_state_dict  # NOQA
from . import CentralizedFL  # NOQA

__all__ = [
    "FedDynClient",
    "FedDynServer",
    "FedDyn"
]


def get_all_params_of(model: torch.Tensor, copy: bool = True) -> torch.Tensor:
    result = None
    for param in model.parameters():
        if not param.requires_grad:
            continue

        if result is None:
            result = param.clone().detach().reshape(-1) if copy else param.reshape(-1)
        else:
            result = torch.cat((result, param.clone().detach().reshape(-1)),
                               0) if copy else torch.cat((result, param.reshape(-1)), 0)
    return result


def load_all_params(device: torch.device, model: torch.nn.Module, params: torch.Tensor) -> None:
    dict_param = deepcopy(dict(model.named_parameters()))
    idx = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(
            params[idx:idx+length].clone().detach().reshape(weights.shape).to(device))
        idx += length

    model.load_state_dict(dict_param, strict=False)


class FedDynClient(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int,
                 alpha: float,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 5,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, clipping=clipping, **kwargs)

        self.hyper_params.update(alpha=alpha)
        self.weight = None
        self.weight_decay = self._optimizer_cfg.optimizer_cfg[
            "weight_decay"] if "weight_decay" in self._optimizer_cfg.optimizer_cfg else 0
        self.prev_grads = None
        self._attr_to_cache.extend(["prev_grads", "weight"])

    def receive_model(self) -> None:
        model, cld_mdl = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.model = model
            self.prev_grads = torch.zeros_like(get_all_params_of(self.model), device=self.device)
        else:
            safe_load_state_dict(self.model, cld_mdl.state_dict())

    def _receive_weights(self) -> None:
        self.weight = self.channel.receive(self, self.server, msg_type="weight").payload

    def _send_weight(self) -> None:
        self.channel.send(Message(self.train_set.tensors[0].shape[0],
                                  "weight",
                                  self,
                                  inmemory=True), self.server)

    def send_model(self) -> None:
        self.channel.send(Message(self.model, "model", self, inmemory=True), self.server)
        self.channel.send(Message(self.prev_grads, "grads", self, inmemory=True), self.server)

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (override_local_epochs if override_local_epochs > 0
                       else self.hyper_params.local_epochs)

        self.model.train()
        self.model.to(self.device)

        alpha_coef_adpt = self.hyper_params.alpha / self.weight
        server_params = get_all_params_of(self.model).to(self.device)

        for params in self.model.parameters():
            params.requires_grad = True

        if not self.optimizer:
            w_dec = alpha_coef_adpt + self.weight_decay
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model,
                                                                 # this override the weight_decay
                                                                 # in the optimizer_cfg
                                                                 weight_decay=w_dec)
        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)

                # Dynamic regularization
                curr_params = get_all_params_of(self.model, False).to(self.device)
                # penalty = -torch.sum(curr_params * self.prev_grads)
                # penalty += 0.5 * alpha_coef_adpt * torch.sum((curr_params - server_params) ** 2)
                penalty = alpha_coef_adpt * \
                    torch.sum(curr_params * (-server_params + self.prev_grads))
                loss = loss + penalty

                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()

            self.scheduler.step()

        # update the previous gradients
        curr_params = get_all_params_of(self.model).to(self.device)
        self.prev_grads += alpha_coef_adpt * (server_params - curr_params)

        running_loss /= (epochs * len(self.train_set))
        self.model.cpu()
        clear_cuda_cache()
        return running_loss


class FedDynServer(Server):
    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = True,
                 alpha: float = 0.01):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.alpha = alpha
        self.device = FlukeENV().get_device()
        self.cld_mdl = deepcopy(self.model).to(self.device)

    def broadcast_model(self, eligible: Iterable[Client]) -> None:
        self.channel.broadcast(Message((self.model, self.cld_mdl), "model", self), eligible)

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1, finalize: bool = True) -> None:

        # Weight computation
        for client in self.clients:
            client._send_weight()

        weights = np.array([self.channel.receive(
            self, client, msg_type="weight").payload for client in self.clients])
        weights = weights / np.sum(weights) * self.n_clients

        for i, client in enumerate(self.clients):
            self.channel.send(Message(weights[i], "weight", self), client)

        for client in self.clients:
            client._receive_weights()

        return super().fit(n_rounds, eligible_perc)

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client], client_models: Iterable[Module]) -> None:
        weights = self._get_client_weights(eligible)
        agg_model_sd = aggregate_models(self.model,
                                        client_models,
                                        weights,
                                        eta=self.hyper_params.lr,
                                        inplace=False).state_dict()

        avg_grad = None
        grad_count = 0
        for client in eligible:
            pg = self.channel.receive(self, client, msg_type="grads").payload
            if pg is not None:
                grad_count += 1
                if avg_grad is None:
                    avg_grad = pg.clone().detach()
                else:
                    avg_grad += pg.clone().detach()

        if grad_count > 0:
            avg_grad /= grad_count

        self.model.load_state_dict(agg_model_sd)
        load_all_params(self.device,
                        self.cld_mdl,
                        get_all_params_of(self.model).to(self.device) + avg_grad.to(self.device))


class FedDyn(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedDynClient

    def get_server_class(self) -> Server:
        return FedDynServer
