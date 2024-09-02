import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss  # , CosineSimilarity
# from copy import deepcopy

from ..client import Client  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..comm import Message  # NOQA
from ..nets import EncoderHeadNet  # NOQA

from ..algorithms import CentralizedFL  # NOQA
from .fedavgm import FedAVGM  # NOQA
from .fedexp import FedExP  # NOQA
from .fedopt import FedOpt  # NOQA
from .scaffold import SCAFFOLDClient, SCAFFOLD  # NOQA
from .fedlc import FedLCClient, FedLC  # NOQA
from .fednova import FedNovaClient, FedNova  # NOQA
from .fedprox import FedProxClient, FedProx  # NOQA
from .moon import MOONClient   # NOQA
from .lg_fedavg import LGFedAVGClient, LGFedAVG  # NOQA


class LargeMarginLoss(torch.nn.Module):
    """
    Large Margin Loss as proposed in the paper [MARGIN2015]_.
    The same concept of margin is also discussed in [AAAI2016]_.

    Args:
        base_loss (torch.nn.Module): Base loss function.
        margin_lam (float): Margin factor.
        reduce (str): Reduction method for the loss. Default is "mean". The other option is "max".
            When "max" is selected, the loss is calculated as the score difference between the
            correct class and the incorrect class with the maximum score.
        num_labels (int): Number of classes.

    References:
        .. [MARGIN2015] Shizhao Sun, et al. Large Margin Deep Neural Networks: Theory and
           Algorithms. In ArXiV 2015. URL: https://arxiv.org/pdf/1506.05232v1
        .. [AAAI2016] Shizhao Sun, et al. On the Depth of Deep Neural Networks: A Theoretical View.
           In AAAI 2016. URL: https://cdn.aaai.org/ojs/10243/10243-13-13771-1-2-20201228.pdf
    """

    def __init__(self,
                 base_loss: torch.nn.Module = CrossEntropyLoss(),
                 margin_lam: float = 0.2,
                 reduce: str = "mean",  # "max"
                 num_labels: int = 10):
        super().__init__()
        self.base_loss = base_loss
        self.margin_lam = margin_lam
        self.num_labels = num_labels
        self.reduce = reduce

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        logits = F.softmax(y_pred, dim=1)
        y_unsqueezed = y_true.view(-1, 1)
        logits_y = torch.gather(logits, 1, y_unsqueezed).view(-1)
        if self.reduce == "mean":
            logits_y = logits_y.repeat(self.num_labels, 1).t()
            diff_y_k = torch.square(1 - (logits_y - logits))
            loss = (diff_y_k.sum() - 1) / (self.num_labels - 1)
        else:
            logits_noy = logits.clone()
            logits_noy[np.arange(len(y_true)), y_true] = -np.inf
            loss = torch.square(1 - logits_y + torch.max(logits_noy, dim=1).values)
            loss = loss.mean()
        return self.base_loss(y_pred, y_true) + self.margin_lam * loss

    def __str__(self):
        return f"LargeMarginLoss(base_loss={self.base_loss}, margin_lam={self.margin_lam},\
            reduce={self.reduce})"

    def __repr__(self) -> str:
        return str(self)


class FedMarginClient(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 margin_lam: float = 0.2,
                 **kwargs):
        super().__init__(index, train_set, test_set, optimizer_cfg,
                         LargeMarginLoss(base_loss=loss_fn,
                                         margin_lam=margin_lam,
                                         num_labels=train_set.num_labels),
                         local_epochs)
        self.hyper_params.update(margin_lam=margin_lam)


# FedAVG + FedMargin
class FedAVGMargin(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedMarginClient


# FedAVGM + FedMargin
class FedAVGMMargin(FedAVGM, FedAVGMargin):
    pass


# FedExP + FedMargin
class FedExPMargin(FedExP, FedAVGMargin):
    pass


# FedOpt + FedMargin
class FedOptMargin(FedOpt, FedAVGMargin):
    pass


# SCAFFOLD + FedMargin
class SCAFFOLDMarginClient(SCAFFOLDClient, FedMarginClient):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 margin_lam: float = 0.2,
                 **kwargs):
        super().__init__(index=index,
                         train_set=train_set,
                         test_set=test_set,
                         optimizer_cfg=optimizer_cfg,
                         loss_fn=LargeMarginLoss(base_loss=loss_fn,
                                                 margin_lam=margin_lam,
                                                 num_labels=train_set.num_labels),
                         local_epochs=local_epochs,
                         margin_lam=margin_lam,
                         ** kwargs)


class SCAFFOLDMargin(SCAFFOLD):

    def get_client_class(self) -> Client:
        return SCAFFOLDMarginClient


class LGFedAVGMarginClient(LGFedAVGClient, FedMarginClient):
    def __init__(self,
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,  # ignored
                 local_epochs: int,
                 margin_lam: float = 0.2,
                 **kwargs):
        super().__init__(index=index,
                         model=model,
                         train_set=train_set,
                         test_set=test_set,
                         optimizer_cfg=optimizer_cfg,
                         loss_fn=LargeMarginLoss(base_loss=loss_fn,
                                                 margin_lam=margin_lam,
                                                 num_labels=train_set.num_labels),
                         local_epochs=local_epochs,
                         margin_lam=margin_lam)


class LGFedAVGMargin(LGFedAVG):

    def get_client_class(self) -> Client:
        return LGFedAVGMarginClient


# FedLC + FedMargin
class FedLCMarginClient(FedLCClient, FedMarginClient):
    pass


class FedLCMargin(FedLC):

    def get_client_class(self) -> Client:
        return FedLCMarginClient


# FedNova + FedMargin
class FedNovaMarginClient(FedNovaClient, FedMarginClient):

    def fit(self, override_local_epochs: int = 0) -> float:
        FedMarginClient.fit(self, override_local_epochs)
        self.tau += self.hyper_params.local_epochs * self.train_set.n_batches
        rho = self._get_momentum()
        self.a = (self.tau - rho * (1.0 - pow(rho, self.tau)) / (1.0 - rho)) / (1.0 - rho)
        self.channel.send(Message(self.a, "local_a", self), self.server)


class FedNovaMargin(FedNova):

    def get_client_class(self) -> Client:
        return FedNovaMarginClient


# FedProx + FedMargin
class FedProxMarginClient(FedProxClient, FedMarginClient):
    pass


class FedProxMargin(FedProx):

    def get_client_class(self) -> Client:
        return FedProxMarginClient


# MOON + FedMargin
class MOONMarginClient(MOONClient, FedMarginClient):
    pass


class MOONMargin(CentralizedFL):

    def get_client_class(self) -> Client:
        return MOONMarginClient
