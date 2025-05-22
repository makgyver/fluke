"""Implementation of the GEAR [GEAR22]_ algorithm.

References:
    .. [GEAR22] Chen Chen, Jie Zhang, Lingjuan Lyu. GEAR: A Margin-based Federated Adversarial
       Training Approach. In FL@AAAI (2022).
       URL: https://federated-learning.org/fl-aaai-2022/Papers/FL-AAAI-22_paper_34.pdf

"""

import sys
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from . import CentralizedFL  # NOQA

__all__ = ["GEARClient", "GEAR", "MarginBasedCrossEntropyLoss"]


class MarginBasedCrossEntropyLoss(nn.Module):
    """Margin-based cross-entropy loss."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        delta: float,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super(MarginBasedCrossEntropyLoss, self).__init__()
        self.class_counts = class_counts.float()
        self.delta = delta
        self.margins = self._compute_margins()
        self.reduction = reduction

    def _compute_margins(self) -> torch.Tensor:
        min_sqrt_n = torch.min(self.class_counts.pow(0.25))
        return self.delta * (min_sqrt_n / self.class_counts.pow(0.25))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        margins = self.margins.to(logits.device)
        target_margins = margins[targets]
        logits_adjusted = logits.clone()
        logits_adjusted[torch.arange(logits.shape[0]), targets] -= target_margins
        return F.cross_entropy(logits_adjusted, targets, reduction=self.reduction)

    def __str__(self, indent: int = 0) -> str:
        indent_str = " " * indent
        return (
            f"{indent_str}MarginBasedCrossEntropyLoss(delta={self.delta},"
            + f"reduction={self.reduction})"
        )

    def __repr__(self, indent: int = 0) -> str:
        return self.__str__(indent=indent)


class GEARClient(Client):

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,  # not used
        local_epochs: int,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        eps: float = 0.1,
        delta: float = 0.01,
        alpha: float = 2.0 / 255,
        adv_iters: int = 10,
        **kwargs,
    ):
        self.sample_per_class = torch.zeros(train_set.num_labels)
        uniq_val, uniq_count = np.unique(train_set.tensors[1], return_counts=True)
        for i, c in enumerate(uniq_val.tolist()):
            self.sample_per_class[c] = uniq_count[i]

        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            loss_fn=MarginBasedCrossEntropyLoss(self.sample_per_class, delta),
            optimizer_cfg=optimizer_cfg,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            **kwargs,
        )
        self.hyper_params.update(eps=eps, delta=delta, alpha=alpha, adv_iters=adv_iters)

    def generate_adversarial(
        self,
        model: Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 0.1,
        alpha: float = 2.0 / 255,
        iters: int = 10,
    ) -> torch.Tensor:
        """
        Generates adversarial examples using Projected Gradient Descent (PGD).

        Args:
            model (nn.Module): The neural network model.
            inputs (Tensor): Input samples.
            targets (Tensor): True labels for the inputs.
            alpha (float): Step size for each iteration.
            iters (int): Number of attack iterations.

        Returns:
            Tensor: Adversarial examples.
        """
        adv_inputs = inputs.clone().detach().requires_grad_(True)
        for _ in range(iters):
            outputs = model(adv_inputs)
            loss = F.cross_entropy(outputs, targets)
            model.zero_grad()
            loss.backward()
            perturbation = alpha * adv_inputs.grad.sign()
            adv_inputs = adv_inputs + perturbation
            adv_inputs = torch.min(torch.max(adv_inputs, inputs - eps), inputs + eps)
            adv_inputs = torch.clamp(adv_inputs, 0, 1).detach().requires_grad_(True)
        return adv_inputs.requires_grad_(False)

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                X_adv = self.generate_adversarial(self.model, X, y)
                self.optimizer.zero_grad()
                y_hat = self.model(X_adv)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= epochs * len(self.train_set)
        self.model.cpu()
        clear_cuda_cache()
        return running_loss


class GEAR(CentralizedFL):

    def get_client_class(self):
        return GEARClient
