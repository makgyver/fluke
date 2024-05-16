from collections import Counter
from typing import Callable
import torch
import sys
sys.path.append(".")
sys.path.append("..")

from ..utils import OptimizerConfigurator  # NOQA
from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from . import CentralizedFL  # NOQA


class FedLCClient(Client):

    def __calibrated_loss(self, logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cal_logit = torch.exp(
            logit
            - (
                self.hyper_params.tau
                * torch.pow(self.label_distrib, -1 / 4)
                .unsqueeze(0)
                .expand((logit.shape[0], -1))
            )
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.sum() / logit.shape[0]

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,  # ignored
                 local_epochs: int,
                 tau: float):
        super().__init__(index, train_set, test_set, optimizer_cfg, None, local_epochs)
        self.hyper_params.update(tau=tau)
        all_labels = self.train_set.tensors[1].tolist()
        label_counter = Counter(all_labels)
        self.label_distrib = torch.zeros(self.train_set.num_labels, device=self.device)
        for cls, count in label_counter.items():
            self.label_distrib[cls] = max(1e-8, count)

        self.hyper_params.loss_fn = self.__calibrated_loss


class FedLC(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedLCClient
