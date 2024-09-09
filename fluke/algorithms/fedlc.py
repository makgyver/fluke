"""Implementation of the [FedLC22]_ algorithm.

References:
    .. [FedLC22] Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Chao Wu.
       Federated Learning with Label Distribution Skew via Logits Calibration. In ICML (2022).
       URL: https://arxiv.org/abs/2209.00189
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

    def __init__(self, tau: float, label_distrib: torch.Tensor):
        super().__init__()
        self.tau = tau
        self.label_distrib = label_distrib

    def forward(self, logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = logit - self.tau * self.label_distrib**(-0.25)
        return torch.nn.functional.cross_entropy(out, y)

    def __str__(self):
        return f"CalibratedLoss(tau={self.tau})"

    def __repr__(self):
        return str(self)


class FedLCClient(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,  # not used
                 local_epochs: int,
                 tau: float,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=None, local_epochs=local_epochs,
                         **kwargs)
        self.hyper_params.update(tau=tau)
        label_counter = torch.zeros(self.train_set.num_labels)
        uniq_val, uniq_count = np.unique(self.train_set.tensors[1], return_counts=True)
        for i, c in enumerate(uniq_val.tolist()):
            label_counter[c] = max(1e-8, uniq_count[i])
        label_counter = label_counter.unsqueeze(dim=0).to(self.device)
        self.hyper_params.loss_fn = CalibratedLoss(tau, label_counter)


class FedLC(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedLCClient
