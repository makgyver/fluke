"""Implementation of the [FedProx18]_ algorithm.

References:
    .. [FedProx18] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar,
       and Virginia Smith. Federated Optimization in Heterogeneous Networks. Adaptive & Multitask
       Learning Workshop. URL: https://openreview.net/pdf?id=SkgwE5Ss3N
"""
import sys
from copy import deepcopy
from typing import Any

import torch

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from . import CentralizedFL  # NOQA


__all__ = [
    "FedProxClient",
    "FedProx"
]


class FedProxClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int,
                 mu: float,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.hyper_params.update(mu=mu)

    def _proximal_loss(self, local_model, global_model):
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            if w_t.requires_grad:
                proximal_term += (w - w_t).norm(2) ** 2
        return proximal_term

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        W = deepcopy(self.model).to(self.device)
        self.model.to(self.device)
        self.model.train()

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y) + \
                    (self.hyper_params.mu / 2) * self._proximal_loss(self.model, W)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.model.to("cpu")
        W.to("cpu")
        clear_cache()
        return running_loss


class FedProx(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedProxClient
