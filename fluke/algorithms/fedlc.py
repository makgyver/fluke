"""Implementation of the [FedLC22]_ algorithm.

References:
    .. [FedLC22] Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Chao Wu.
       Federated Learning with Label Distribution Skew via Logits Calibration. In ICML (2022).
       URL: https://arxiv.org/abs/2209.00189
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

__all__ = [
    "CalibratedLoss",
    "FedLCClient",
    "FedLC"
]


class CalibratedLoss(torch.nn.Module):
    """Calibrated Loss function.

    Args:
        tau (float): calibration parameter.
        label_distrib (torch.Tensor): Label distribution.
    """

    def __init__(self,
                 tau: float,
                 label_distrib: torch.Tensor,
                 reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        self.tau = tau
        self.label_distrib = label_distrib
        self.reduction = reduction

    def forward(self, logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = logit - self.tau * self.label_distrib.to(logit.device)**(-0.25)
        return torch.nn.functional.cross_entropy(out, y, reduction=self.reduction)

    def __str__(self, indent: int = 0) -> str:
        indent_str = " " * indent
        return f"{indent_str}CalibratedLoss(tau={self.tau}, reduction={self.reduction})"

    def __repr__(self, indent: int = 0) -> str:
        return self.__str__(indent=indent)


class FedLCClient(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,  # not used
                 local_epochs: int,
                 tau: float,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 0,
                 **kwargs):
        label_counter = torch.zeros(train_set.num_labels)
        uniq_val, uniq_count = np.unique(train_set.tensors[1], return_counts=True)
        for i, c in enumerate(uniq_val.tolist()):
            label_counter[c] = max(1e-8, uniq_count[i])
        label_counter = label_counter.unsqueeze(dim=0)  # .to(self.device)
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=CalibratedLoss(tau, label_counter),
                         local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, clipping=clipping, **kwargs)
        self.hyper_params.update(tau=tau)


class FedLC(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return FedLCClient
