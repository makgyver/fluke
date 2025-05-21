"""Implementation of the DITTO [Ditto21]_ algorithm.

References:
    .. [Ditto21] Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. Ditto: Fair and Robust
       Federated Learning Through Personalization. In ICML (2021).
       URL: https://arxiv.org/abs/2012.04221
"""
import sys
from copy import deepcopy
from typing import Iterator

import torch
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

sys.path.append(".")
sys.path.append("..")

from ..client import PFLClient  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from . import PersonalizedFL  # NOQA

__all__ = [
    "PerturbedGradientDescent",
    "DittoClient",
    "Ditto"
]


class PerturbedGradientDescent(Optimizer):
    def __init__(self,
                 params: Iterator[Parameter],
                 lr: float = 0.01,
                 lam: float = 0.0,
                 **kwargs):
        default = dict(lr=lr, lam=lam)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                d_p = p.grad.data + group['lam'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


# https://arxiv.org/pdf/2012.04221.pdf
class DittoClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: torch.nn.Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 0,
                 tau: int = 3,
                 lam: float = 0.1,
                 **kwargs):
        super().__init__(index=index, model=model, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, clipping=clipping, **kwargs)
        self.pers_optimizer = None
        self.pers_scheduler = None
        self.hyper_params.update(
            tau=tau,
            lam=lam
        )
        self._save_to_cache()

    def _proximal_loss(self,
                       local_model: torch.nn.Module,
                       global_model: torch.nn.Module) -> torch.Tensor:
        proximal_term = 0.0
        for name, param in local_model.named_parameters():
            # if 'weight' not in name:
            #     continue
            proximal_term += (param - global_model.get_parameter(name))  # .norm(2)
        return proximal_term

    def fit(self, override_local_epochs: int = 0) -> dict:
        epochs: int = (override_local_epochs if override_local_epochs > 0
                       else self.hyper_params.local_epochs)

        w_prev = deepcopy(self.model)

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
            self.scheduler.step()

        self.model.cpu()
        clear_cuda_cache()

        self.personalized_model.train()
        self.personalized_model.to(self.device)
        w_prev.to(self.device)

        if self.pers_optimizer is None:
            self.pers_optimizer = PerturbedGradientDescent(self.personalized_model.parameters(),
                                                           lam=self.hyper_params.lam,
                                                           **self._optimizer_cfg.optimizer_cfg)
            self.pers_scheduler = self._optimizer_cfg.scheduler(
                self.pers_optimizer,
                **self._optimizer_cfg.scheduler_cfg
            )

        running_loss = 0.0
        for _ in range(self.hyper_params.tau):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.pers_optimizer.zero_grad()
                y_hat = self.personalized_model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.pers_optimizer.step(w_prev.parameters())
                running_loss += loss.item()
            self.pers_scheduler.step()

        running_loss /= (self.hyper_params.tau * len(self.train_set))
        self.personalized_model.cpu()
        w_prev.cpu()
        clear_cuda_cache()

        return running_loss


class Ditto(PersonalizedFL):

    def get_client_class(self) -> type[PFLClient]:
        return DittoClient
