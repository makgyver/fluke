"""Implementation of the [FedSAM22]_ algorithm.

References:
    .. [FedSAM22] Caldarola, D., Caputo, B., & Ciccone, M. Improving Generalization in Federated
       Learning by Seeking Flat Minima. In ECCV (2022).
       URL: https://arxiv.org/abs/2203.11834
"""

import sys
from typing import Any, Collection, Union

import torch
from torch.optim import Optimizer

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from . import CentralizedFL  # NOQA

# This implementation is based on
# https://github.com/bytedance/FedDecorr/blob/master/approach/fedsam.py

__all__ = ["SAMOptimizer", "FedSAMClient", "FedSAM"]


class SAMOptimizer(torch.optim.Optimizer):

    def __init__(
        self,
        params: Union[Collection[torch.Tensor], Collection[dict[str, Any]]],
        base_optimizer: Optimizer = torch.optim.SGD,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.defaults = defaults
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Union[None, callable] = None) -> float:
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, \
            but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        loss1 = closure()
        self.first_step(zero_grad=True)
        loss2 = closure()
        self.second_step()

        return (loss1.item() + loss2.item()) / 2

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["base_optimizer"] = self.base_optimizer
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.base_optimizer = state["base_optimizer"]


class FedSAMClient(Client):

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: torch.nn.Module,  # ignored
        local_epochs: int,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        rho: float = 0.05,
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
        self.hyper_params.update(rho=rho)

    def _get_closure(self, X: torch.Tensor, y: torch.Tensor) -> callable:
        def closure():
            y_hat = self.model(X)
            loss = self.hyper_params.loss_fn(y_hat, y)
            loss.backward()
            self._clip_grads(self.model)
            return loss

        return closure

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )
        self.model.to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(
                self.model, rho=self.hyper_params.rho
            )
        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                running_loss += self.optimizer.step(self._get_closure(X, y))

            self.scheduler.step()

        running_loss /= epochs * len(self.train_set)
        self.model.cpu()
        clear_cuda_cache()
        return running_loss


class FedSAM(CentralizedFL):

    def can_override_optimizer(self) -> bool:
        return False

    def get_optimizer_class(self) -> type[Optimizer]:
        return SAMOptimizer

    def get_client_class(self) -> type[Client]:
        return FedSAMClient
