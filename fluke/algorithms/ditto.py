from copy import deepcopy
import torch
from typing import Any, Callable
import sys
sys.path.append(".")
sys.path.append("..")

from . import PersonalizedFL  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..data import FastDataLoader  # NOQA
from ..client import PFLClient  # NOQA


# https://arxiv.org/pdf/2012.04221.pdf
class DittoClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: torch.nn.Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable[..., Any],
                 local_epochs: int = 3,
                 tau: int = 3,
                 lam: float = 0.1):
        super().__init__(index, model, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.pers_optimizer = None
        self.pers_scheduler = None
        self.hyper_params.update(
            tau=tau,
            lam=lam
        )

    def _proximal_loss(self, local_model, global_model):
        proximal_term = 0.0
        for name, param in local_model.named_parameters():
            if 'weight' not in name:
                continue
            proximal_term += (param - global_model.get_parameter(name)).norm(2)
        return proximal_term

    def fit(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()

        w_prev = deepcopy(self.model)

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.model.to("cpu")
        clear_cache()

        self.personalized_model.train()
        self.personalized_model.to(self.device)
        w_prev.to(self.device)

        if self.pers_optimizer is None:
            self.pers_optimizer, self.pers_scheduler = self.optimizer_cfg(self.personalized_model)

        for _ in range(self.hyper_params.tau):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.pers_optimizer.zero_grad()
                y_hat = self.personalized_model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss += self.hyper_params.lam * self._proximal_loss(self.personalized_model, w_prev)
                loss.backward()
                self.pers_optimizer.step()
            self.pers_scheduler.step()

        self.personalized_model.to("cpu")
        w_prev.to("cpu")
        clear_cache()
        self.send_model()


class Ditto(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return DittoClient
