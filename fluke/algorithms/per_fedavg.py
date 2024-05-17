from torch.optim import Optimizer
import torch
from typing import Callable, Union
from collections import OrderedDict
from copy import deepcopy
import sys

sys.path.append(".")
sys.path.append("..")

from ..utils import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA


class PerFedAVGOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerFedAVGOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, beta=0, grads=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            if grads is None:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    p.data.sub_(d_p, alpha=beta if (beta != 0) else group['lr'])
            else:
                for p, g1, g2 in zip(group['params'], grads[0], grads[1]):
                    if p.grad is None:
                        continue
                    p.data.sub_(beta * g1 - beta * group['lr'] * g2)
        return loss


class PerFedAVGClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 mode: str,
                 beta: float):

        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update(
            mode=mode,
            beta=beta
        )
        self.train_iterator = iter(self.train_set)

    def _get_next_batch(self):
        try:
            X, y = next(self.train_set)
        except StopIteration:
            self.train_iterator = iter(self.train_set)
            X, y = next(self.train_iterator)

        return X, y

    def _compute_grad(self,
                      model: torch.nn.Module,
                      data_batch: tuple[torch.Tensor, torch.Tensor],
                      v: Union[tuple[torch.Tensor, ...], None] = None) -> list[torch.Tensor]:
        X, y = data_batch
        if v is not None:
            frz_model_params = deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(X)
            loss_1 = self.hyper_params.loss_fn(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(X)
            loss_2 = self.hyper_params.loss_fn(logit_2, y)
            grads_2 = torch.autograd.grad(loss_2, model.parameters())

            model.load_state_dict(frz_model_params)

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads
        else:
            logit = model(X)
            loss = self.hyper_params.loss_fn(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads

    def fit(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()
        self.model.train()
        if self.optimizer is None:
            self.optimizer, _ = self.optimizer_cfg(self.model)

        for _ in range(epochs):
            loss = None

            batch_1 = self._get_next_batch()
            batch_2 = self._get_next_batch()

            X, y = batch_1
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(X)
            loss = self.hyper_params.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step(self.model.parameters())

            if self.hyper_params.mode == "FO":
                X, y = batch_2
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step(self.model.parameters(), beta=self.hyper_params.beta)

            elif self.hyper_params.mode == "HF":
                batch_3 = self._get_next_batch()

                temp_model = deepcopy(self.model)
                grads_1st = self._compute_grad(temp_model, batch_2)
                grads_2nd = self._compute_grad(self.model, batch_3, v=grads_1st)
                self.optimizer.step(self.model.parameters(),
                                    beta=self.hyper_params.beta,
                                    grads=(grads_1st, grads_2nd))

            else:
                raise ValueError(f"Invalid mode: {self.hyper_params.mode}")

        self.send_model()


class PerFedAVG(CentralizedFL):

    def get_client_class(self) -> Client:
        return PerFedAVGClient

    def get_optimizer_class(self) -> Optimizer:
        return PerFedAVGOptimizer
