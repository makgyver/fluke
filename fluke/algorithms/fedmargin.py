import numpy as np
from typing import Callable, Literal
import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from copy import deepcopy

from ..evaluation import ClassificationEval  # NOQA
from ..client import Client  # NOQA
from ..utils import clear_cache, OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..comm import Message  # NOQA

from ..algorithms import CentralizedFL  # NOQA
from .fedavgm import FedAVGM  # NOQA
from .fedexp import FedExP  # NOQA
from .fedopt import FedOpt  # NOQA
from .scaffold import SCAFFOLDClient, SCAFFOLD  # NOQA
from .fedlc import FedLCClient  # NOQA
from .fednova import FedNovaClient  # NOQA
from .fedprox import FedProxClient  # NOQA
from .moon import MOONClient   # NOQA


def _max_with_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + F.relu(b - a)


def _get_grad(out_: torch.Tensor, in_: torch.Tensor) -> torch.Tensor:
    grad, *_ = torch.autograd.grad(out_, in_,
                                   grad_outputs=torch.ones_like(out_, dtype=torch.float32),
                                   retain_graph=True)
    return grad.view(in_.shape[0], -1)


class LargeMarginLoss:
    """Large Margin Loss as proposed in the paper [MARGIN2018]_.
    This implementation follows the one in (
    https://github.com/zsef123/Large_Margin_Loss_PyTorch) which is based on
    the official repo (
    https://github.com/google-research/google-research/tree/master/large_margin).

    Args:
        gamma (float): Desired margin, and distance to boundary above the margin will be clipped.
        alpha_factor (float): Factor to determine the lower bound of margin.
            Both gamma and alpha_factor determine points to include in training
            the margin these points lie with distance to boundary of [gamma * (1 - alpha), gamma]
        top_k (int): Number of top classes to include in the margin loss.
        dist_norm (1, 2, np.inf): Distance to boundary defined on norm
        epslion (float): Small number to avoid division by 0.
        use_approximation (bool):
        agg_fun ("min", "avg", "all"):  If 'min'
            only consider the minimum distance to boundary of the top_k classes. If
            'avg' consider average distance to boundary. If 'all'
            consider all top_k. When top_k = 1, these choices are equivalent.

    References:
        .. [MARGIN2018] Gamaleldin F. Elsayed, Dilip Krishnan, Hossein Mobahi, Kevin Regan, Samy
           Bengio. Large Margin Loss for Multi-class Classification. in: NeurIPS 2018.
           URL: https://arxiv.org/pdf/1803.05598
    """

    def __init__(self,
                 gamma: float = 10000.0,
                 alpha_factor: float = 4.0,
                 top_k: int = 1,
                 dist_norm: Literal[1, 2, "inf"] = 2,
                 epsilon: float = 1e-8,
                 agg_fun: str = "avg"):

        assert dist_norm in [1, 2, np.inf, "inf"], "dist_norm must be 1, 2, or np.inf"
        assert agg_fun in ["min", "avg", "all"], "agg_fun must be 'min', 'avg', or 'all'"
        assert top_k > 0, "top_k must be a positive integer"

        self.dist_upper = gamma
        self.dist_lower = gamma * (1.0 - alpha_factor)

        self.alpha = alpha_factor
        self.top_k = top_k
        self.dual_norm = {1: np.inf, 2: 2, np.inf: 1, "inf": 1}[dist_norm]
        self.eps = epsilon

        self.agg_fun = agg_fun

    def __call__(self,
                 logits: torch.Tensor,
                 onehot_labels: torch.Tensor,
                 feature_maps: list[torch.Tensor]):
        """Compute the Large Margin loss.

        Args:
            logits (Tensor): output of network *before* softmax
            onehot_labels (Tensor): One-hot encoded label.
            feature_maps (list of Tensor): Target feature maps (i.e., output of a layer of the
                model) want to enforcing by Large Margin.

        Returns:
            loss:  Large Margin loss
        """
        prob = F.softmax(logits, dim=1)
        correct_prob = prob * onehot_labels

        correct_prob = torch.sum(correct_prob, dim=1, keepdim=True)
        other_prob = prob * (1.0 - onehot_labels)

        if self.top_k > 1:
            topk_prob, _ = other_prob.topk(self.top_k, dim=1)
        else:
            topk_prob, _ = other_prob.max(dim=1, keepdim=True)

        diff_prob = correct_prob - topk_prob

        loss = torch.empty(0, device=logits.device)
        for feature_map in feature_maps:
            diff_grad = torch.stack([_get_grad(diff_prob[:, i], feature_map)
                                     for i in range(self.top_k)], dim=1)
            diff_gradnorm = torch.norm(diff_grad, p=self.dual_norm, dim=2)
            diff_gradnorm.detach_()

            dist_to_boundary = diff_prob / (diff_gradnorm + self.eps)

            if self.agg_fun == "min":
                dist_to_boundary, _ = dist_to_boundary.min(dim=1)
            elif self.agg_fun == "avg":
                dist_to_boundary = dist_to_boundary.mean(dim=1)
            # else "all"

            loss_layer = _max_with_relu(dist_to_boundary, self.dist_lower)
            loss_layer = _max_with_relu(0, self.dist_upper - loss_layer) - self.dist_upper
            loss = torch.cat([loss, loss_layer])
        return loss.mean()

    def __str__(self):
        return f"LargeMarginLoss(gamma={self.dist_upper}, alpha={self.alpha}, top_k={self.top_k}, "\
            f"dist_norm={self.dual_norm}, epsilon={self.eps}, agg_fun={self.agg_fun})"

    def __repr__(self) -> str:
        return self.__str__()


class FedMarginClient(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int = 3,
                 margin_lam: float = 0.2,
                 **kwargs):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update(margin_lam=margin_lam)

    def fit(self, override_local_epochs: int = 0) -> None:
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)
        self.receive_model()
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                one_hot_y = torch.zeros(len(y),
                                        self.train_set.num_labels,
                                        device=self.device).scatter_(1,
                                                                     y.unsqueeze(1),
                                                                     1.).float()
                # one_hot_y = one_hot_y.to(self.device)
                self.optimizer.zero_grad()
                feature_maps = self.model.encoder(X)
                y_hat = self.model.head(feature_maps)
                lam = self.hyper_params.margin_lam
                loss = (1. - lam) * self.hyper_params.loss_fn(y_hat, y) + \
                    lam * LargeMarginLoss()(y_hat, one_hot_y, [feature_maps])
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.model.to("cpu")
        clear_cache()
        self.send_model()

    def evaluate(self) -> dict[str, float]:
        if self.test_set is not None and self.model is not None:
            return ClassificationEval(None,  # self.hyper_params.loss_fn,
                                      #   self.model.output_size,
                                      self.train_set.num_labels,
                                      self.device).evaluate(self.model,
                                                            self.test_set)
        return {}


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

    def fit(self, override_local_epochs: int = 0):
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()
        server_model = deepcopy(self.model)
        self.model.to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X = X.to(self.device)
                one_hot_y = torch.zeros(len(y),
                                        self.train_set.num_labels).scatter_(1,
                                                                            y.unsqueeze(1),
                                                                            1.).float()
                y = one_hot_y.to(self.device)
                self.optimizer.zero_grad()
                feature_maps = self.model.encoder(X)
                y_hat = self.model.head(feature_maps)
                lam = self.hyper_params.margin_lam
                loss = (1. - lam) * self.hyper_params.loss_fn(y_hat, y) + \
                    lam * LargeMarginLoss()(y_hat, y, [feature_maps])
                loss.backward()
                self.optimizer.step(self.server_control, self.control)
            self.scheduler.step()

        params = zip(self.model.parameters(), server_model.parameters(), self.delta_y)
        for local_model, server_model, delta_y in params:
            delta_y.data = local_model.data.detach() - server_model.data.detach()

        new_controls = [torch.zeros_like(p.data)
                        for p in self.model.parameters() if p.requires_grad]
        coeff = 1. / (self.hyper_params.local_epochs * len(self.train_set)
                      * self.scheduler.get_last_lr()[0])
        params = zip(self.control, self.server_control, new_controls, self.delta_y)
        for local_control, server_control, new_control, delta_y in params:
            new_control.data = local_control.data - server_control.data - delta_y.data * coeff

        for local_control, new_control, delta_c in zip(self.control, new_controls, self.delta_c):
            delta_c.data = new_control.data - local_control.data
            local_control.data = new_control.data

        self.model.to("cpu")
        clear_cache()
        self.send_model()


class SCAFFOLDMargin(SCAFFOLD):

    def get_client_class(self) -> Client:
        return SCAFFOLDMarginClient


# FedLC + FedMargin
class FedLCMarginClient(FedLCClient, FedMarginClient):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,  # ignored
                 local_epochs: int,
                 tau: float,
                 margin_lam: float):
        super().__init__(index,
                         train_set,
                         test_set,
                         optimizer_cfg,
                         None,
                         local_epochs,
                         tau=tau,
                         margin_lam=margin_lam)


class FedLCMargin(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedLCMarginClient


# FedNova + FedMargin
class FedNovaMarginClient(FedNovaClient, FedMarginClient):

    def fit(self, override_local_epochs: int = 0) -> None:
        FedMarginClient.fit(self, override_local_epochs)
        self.tau += self.hyper_params.local_epochs * self.train_set.n_batches
        rho = self._get_momentum()
        self.a = (self.tau - rho * (1.0 - pow(rho, self.tau)) / (1.0 - rho)) / (1.0 - rho)
        self.channel.send(Message(self.a, "local_a", self), self.server)


class FedNovaMargin(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedNovaMarginClient


# FedProx + FedMargin
class FedProxMarginClient(FedProxClient, FedMarginClient):

    def fit(self, override_local_epochs: int = 0):
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()
        W = deepcopy(self.model)
        self.model.to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                one_hot_y = torch.zeros(len(y),
                                        self.train_set.num_labels).scatter_(1,
                                                                            y.unsqueeze(1),
                                                                            1.).float()
                one_hot_y = one_hot_y.to(self.device)
                self.optimizer.zero_grad()
                feature_maps = self.model.encoder(X)
                y_hat = self.model.head(feature_maps)
                lam = self.hyper_params.margin_lam
                loss = (1. - lam) * (self.hyper_params.loss_fn(
                    y_hat, y) + (self.hyper_params.mu / 2) * self._proximal_loss(self.model, W)) + \
                    lam * LargeMarginLoss()(y_hat, one_hot_y, [feature_maps])
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.model.to("cpu")
        clear_cache()
        self.send_model()


class FedProxMargin(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedProxMarginClient


# MOON + FedMargin
class MOONMarginClient(MOONClient, FedMarginClient):
    def fit(self, override_local_epochs: int = 0):
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()
        cos = CosineSimilarity(dim=-1).to(self.device)
        self.model.to(self.device)
        self.prev_model.to(self.device)
        self.server_model.to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                one_hot_y = torch.zeros(len(y),
                                        self.train_set.num_labels).scatter_(1,
                                                                            y.unsqueeze(1),
                                                                            1.).float()
                one_hot_y = one_hot_y.to(self.device)
                self.optimizer.zero_grad()

                z_local = self.model.encoder(X)  # , -1)
                y_hat = self.model.head(z_local)
                loss_sup = self.hyper_params.loss_fn(y_hat, y)

                z_prev = self.prev_model.encoder(X)  # , -1)
                z_global = self.server_model.encoder(X)  # , -1)

                sim_lg = cos(z_local, z_global).reshape(-1, 1) / self.hyper_params.tau
                sim_lp = cos(z_local, z_prev).reshape(-1, 1) / self.hyper_params.tau
                loss_con = -torch.log(torch.exp(sim_lg) /
                                      (torch.exp(sim_lg) + torch.exp(sim_lp))).mean()

                lam = self.hyper_params.margin_lam
                loss = (1. - lam) * (loss_sup + self.hyper_params.mu * loss_con) + \
                    lam * LargeMarginLoss()(y_hat, one_hot_y, [z_local])
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.prev_model.to("cpu")
        self.server_model.to("cpu")
        self.model.to("cpu")
        clear_cache()
        self.send_model()


class MOONMargin(CentralizedFL):

    def get_client_class(self) -> Client:
        return MOONMarginClient
