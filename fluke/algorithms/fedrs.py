"""Implementation of the [FedRS21]_ algorithm.

References:
    .. [FedRS21] Xin-Chun Li and De-Chuan Zhan. FedRS: Federated Learning with Restricted
       Softmax for Label Distribution Non-IID Data. In KDD (2021).
       URL: https://doi.org/10.1145/3447548.3467254
"""

import sys
from typing import Literal

import numpy as np
import torch

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from . import CentralizedFL  # NOQA

__all__ = ["RSLoss", "FedRSClient", "FedRS"]


class RSLoss(torch.nn.Module):
    """Restricted Softmax Loss function.

    Args:
        class_scaling (torch.Tensor): Class scaling factor.
        reduction (Literal["mean", "sum"]): Specifies the reduction to apply to the output.

    See Also:
        This loss function is very similar to the one used in ``FedLC``, i.e.,
        :class:`fluke.algorithms.fedlc.CalibratedLoss`.
    """

    def __init__(self, class_scaling: torch.Tensor, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        self.class_scaling = class_scaling
        self.reduction = reduction

    def forward(self, logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = logit * self.class_scaling.to(logit.device)
        return torch.nn.functional.cross_entropy(out, y, reduction=self.reduction)

    def __str__(self, indent: int = 0) -> str:
        indent_str = " " * indent
        return f"{indent_str}RSLoss()"

    def __repr__(self, indent: int = 0) -> str:
        return self.__str__(indent=indent)


class FedRSClient(Client):

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: torch.nn.Module,  # ignored
        local_epochs: int,
        alpha: float,
        count_as_missing: int = 2,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        **kwargs,
    ):
        uniq_val, uniq_cnt = np.unique(train_set.tensors[1], return_counts=True)
        class_scaling = torch.ones(train_set.num_labels) * alpha
        for i, c in enumerate(uniq_val):
            if uniq_cnt[i] > count_as_missing:
                class_scaling[c] = 1.0
        class_scaling = class_scaling.unsqueeze(dim=0)
        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=RSLoss(class_scaling),
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            **kwargs,
        )
        self.hyper_params.update(alpha=alpha, count_as_missing=count_as_missing)


class FedRS(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return FedRSClient
