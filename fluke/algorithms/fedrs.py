"""Implementation of the [FedRS21]_ algorithm.

References:
    .. [FedRS21] Xin-Chun Li and De-Chuan Zhan. FedRS: Federated Learning with Restricted
       Softmax for Label Distribution Non-IID Data. In KDD (2021).
       URL: https://doi.org/10.1145/3447548.3467254
"""
import sys
from typing import Any

import numpy as np
import torch

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from . import CentralizedFL  # NOQA

__all__ = [
    "RSLoss",
    "FedRSClient",
    "FedRS"
]


class RSLoss(torch.nn.Module):
    """Restricted Softmax Loss function.

    Args:
        class_scaling (torch.Tensor): Class scaling factor.

    See Also:
        This loss function is very similar to the one used in ``FedLC``, i.e.,
        :class:`fluke.algorithms.fedlc.CalibratedLoss`.
    """

    def __init__(self, class_scaling: torch.Tensor):
        super().__init__()
        self.class_scaling = class_scaling

    def forward(self, logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = logit * self.class_scaling
        return torch.nn.functional.cross_entropy(out, y)

    def __str__(self):
        return "RSLoss()"

    def __repr__(self):
        return str(self)


class FedRSClient(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,  # ignored
                 local_epochs: int,
                 alpha: float,
                 count_as_missing: int = 2,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=None, local_epochs=local_epochs,
                         **kwargs)
        self.hyper_params.update(alpha=alpha, count_as_missing=count_as_missing)
        uniq_val, uniq_cnt = np.unique(self.train_set.tensors[1], return_counts=True)
        class_scaling = torch.ones(self.train_set.num_labels) * \
            self.hyper_params.alpha
        for i, c in enumerate(uniq_val):
            if uniq_cnt[i] > self.hyper_params.count_as_missing:
                class_scaling[c] = 1.0
        class_scaling = class_scaling.unsqueeze(dim=0).to(self.device)
        self.hyper_params.loss_fn = RSLoss(class_scaling)


class FedRS(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedRSClient
