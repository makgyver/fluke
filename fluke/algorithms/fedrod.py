"""Implementation of the FedROD [FedROD22]_ algorithm.

References:
    .. [FedROD22] Hong-You Chen and Wei-Lun Chao. On Bridging Generic and Personalized Federated
       Learning for Image Classification. In ICLR (2022).
       URL: https://openreview.net/pdf?id=I1hQbx10Kxn
"""

import sys
from copy import deepcopy
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

sys.path.append(".")
sys.path.append("..")

from .. import FlukeENV  # NOQA
from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..nets import EncoderHeadNet  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from ..utils.model import ModOpt, unwrap  # NOQA
from . import CentralizedFL  # NOQA

__all__ = ["RODModel", "BalancedSoftmaxLoss", "FedRODClient", "FedROD"]


class RODModel(torch.nn.Module):
    """Model that combines a global model and a local head.
    During the forward pass, the global model, formed by an encoder and a head, is used to extract
    the representation of the input (using the encoder). The representation is then passed to
    the local head and the global head. The output of the local head is added to the output of the
    global head and returned as the final output.

    Args:
        global_model (EncoderHeadNet): Global model.
        local_head (EncoderHeadNet): Local head.
    """

    def __init__(self, global_model: EncoderHeadNet, local_head: EncoderHeadNet):
        super().__init__()
        self.local_head = local_head
        self.global_model = global_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = unwrap(self.global_model).encoder(x)
        out_g = unwrap(self.global_model).head(rep)
        out_p = self.local_head(rep.detach())
        output = out_g.detach() + out_p
        return output


class BalancedSoftmaxLoss(torch.nn.Module):
    """Compute the Balanced Softmax Loss.

    Args:
        sample_per_class (torch.Tensor): Number of samples per class.
    """

    def __init__(self, sample_per_class: torch.Tensor, reduction: Literal["mean", "sum"] = "mean"):
        super().__init__()
        self.sample_per_class = sample_per_class
        self.reduction = reduction

    def forward(self, y: torch.LongTensor, logits: torch.FloatTensor) -> torch.Tensor:
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=y, reduction=self.reduction)
        return loss


class FedRODClient(Client):

    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: torch.nn.Module,
        local_epochs: int,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
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

        self.sample_per_class: torch.Tensor = torch.zeros(self.train_set.num_labels)
        uniq_val, uniq_count = np.unique(self.train_set.tensors[1], return_counts=True)
        for i, c in enumerate(uniq_val.tolist()):
            self.sample_per_class[c] = uniq_count[i]
        self._inner_modopt: ModOpt = ModOpt()

    @property
    def inner_model(self) -> torch.nn.Module:
        return self._inner_modopt.model

    @inner_model.setter
    def inner_model(self, model: torch.nn.Module) -> None:
        self._inner_modopt.model = model

    @property
    def optimizer_head(self) -> Optimizer:
        return self._inner_modopt.optimizer

    @optimizer_head.setter
    def optimizer_head(self, optimizer: Optimizer) -> None:
        self._inner_modopt.optimizer = optimizer

    @property
    def scheduler_head(self) -> LRScheduler:
        return self._inner_modopt.scheduler

    @scheduler_head.setter
    def scheduler_head(self, scheduler: LRScheduler) -> None:
        self._inner_modopt.scheduler = scheduler

    def _model_to_dataparallel(self):
        super()._model_to_dataparallel()
        self.inner_model = torch.nn.DataParallel(
            self.inner_model, device_ids=FlukeENV().get_device_ids()
        )

    def _dataparallel_to_model(self):
        super()._dataparallel_to_model()
        self.inner_model = self.inner_model.module

    def receive_model(self) -> None:
        super().receive_model()
        if self.inner_model is None:
            self.inner_model = deepcopy(self.model.head)

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (
            override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        )
        self.model.train()
        self.inner_model.train()
        self.model.to(self.device)
        self.inner_model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)
            self.optimizer_head, self.scheduler_head = self._optimizer_cfg(self.inner_model)

        bsm_loss = BalancedSoftmaxLoss(self.sample_per_class)
        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)

                rep = unwrap(self.model).encoder(X)
                out_g = unwrap(self.model).head(rep)
                loss = bsm_loss(y, out_g)
                self.optimizer.zero_grad()
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()

                out_p = self.inner_model(rep.detach())
                loss = self.hyper_params.loss_fn(out_g.detach() + out_p, y)
                self.optimizer_head.zero_grad()
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer_head.step()

            self.scheduler.step()
            self.scheduler_head.step()

        running_loss /= epochs * len(self.train_set)
        self.model.cpu()
        self.inner_model.cpu()
        clear_cuda_cache()
        return running_loss

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if test_set is not None and self.model is not None and self.inner_model is not None:
            return evaluator.evaluate(
                self._last_round,
                RODModel(self.model, self.inner_model),
                test_set,
                device=self.device,
                loss_fn=None
            )
        return {}


class FedROD(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return FedRODClient
