"""Implementation of the [pFedMe20]_ algorithm.

References:
    .. [pFedMe20] Canh T. Dinh, Nguyen H. Tran, and Tuan Dung Nguyen. Personalized Federated
       Learning with Moreau Envelopes. In NeurIPS (2020). URL: https://arxiv.org/abs/2006.08848
"""
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Iterable, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client, PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from ..utils.model import (STATE_DICT_KEYS_TO_IGNORE,  # NOQA
                           safe_load_state_dict)

__all__ = [
    "PFedMeOptimizer",
    "PFedMeClient",
    "PFedMeServer",
    "PFedMe"
]


class PFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(PFedMeOptimizer, self).__init__(params, defaults)

    def step(self,
             local_parameters: list[torch.nn.Parameter],
             closure: callable = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure
        for group in self.param_groups:
            for param_p, param_l in zip(group["params"], local_parameters):
                param_p.data = param_p.data - group["lr"] * (
                    param_p.grad.data
                    + group["lamda"] * (param_p.data - param_l.data)
                    + group["mu"] * param_p.data
                )
        return loss


class PFedMeClient(PFLClient):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int,
                 k: int,
                 **kwargs: dict[str, Any]):

        super().__init__(index=index, model=None, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.hyper_params.update(k=k)

    def receive_model(self) -> None:
        model = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.personalized_model = model
            self.model = deepcopy(self.personalized_model)
        else:
            safe_load_state_dict(self.personalized_model, model.state_dict())

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.personalized_model.train()
        self.personalized_model.to(self.device)
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.personalized_model)

        lamda = self.optimizer.defaults["lamda"]
        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                for _ in range(self.hyper_params.k):
                    self.optimizer.zero_grad()
                    y_hat = self.personalized_model(X)
                    loss = self.hyper_params.loss_fn(y_hat, y)
                    loss.backward()
                    self.optimizer.step(self.model.parameters())

                lr = self.optimizer.param_groups[0]["lr"]
                params = zip(self.personalized_model.parameters(), self.model.parameters())
                for param_p, param_l in params:
                    param_l.data = param_l.data - lamda * lr * (param_l.data - param_p.data)
                running_loss += loss.item()
            self.scheduler.step()
        running_loss /= (epochs * len(self.train_set))
        self.model.load_state_dict(self.personalized_model.state_dict())
        self.model.to("cpu")
        self.personalized_model.to("cpu")
        return running_loss

    def send_model(self):
        self.channel.send(Message(self.model, "model", self), self.server)


class PFedMeServer(Server):
    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False,
                 beta: float = 0.5,
                 **kwargs: dict[str, Any]):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.hyper_params.update(beta=beta)

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self.get_client_models(eligible)
        weights = self._get_client_weights(eligible)
        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                # avg_model_sd[key] = clients_sd[0][key].clone()
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue
            for i, client_sd in enumerate(clients_sd):
                if key not in avg_model_sd:
                    avg_model_sd[key] = weights[i] * client_sd[key]
                else:
                    avg_model_sd[key] += weights[i] * client_sd[key]

        for key, param in self.model.named_parameters():
            param.data = (1 - self.hyper_params.beta) * param.data
            param.data += self.hyper_params.beta * avg_model_sd[key]


class PFedMe(CentralizedFL):

    def can_override_optimizer(self) -> bool:
        return False

    def get_optimizer_class(self) -> torch.optim.Optimizer:
        return PFedMeOptimizer

    def get_client_class(self) -> Client:
        return PFedMeClient

    def get_server_class(self) -> Server:
        return PFedMeServer
