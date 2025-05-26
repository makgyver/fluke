"""Implementation of the [pFedMe20]_ algorithm.

References:
    .. [pFedMe20] Canh T. Dinh, Nguyen H. Tran, and Tuan Dung Nguyen. Personalized Federated
       Learning with Moreau Envelopes. In NeurIPS (2020). URL: https://arxiv.org/abs/2006.08848
"""

import sys
from copy import deepcopy
from typing import Collection, Optional, Sequence

import torch
from torch.nn import Module
from torch.optim import Optimizer

from ..utils.model import safe_load_state_dict

sys.path.append(".")
sys.path.append("..")

from .. import FlukeENV  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils.model import aggregate_models  # NOQA

__all__ = ["PFedMeOptimizer", "PFedMeClient", "PFedMeServer", "PFedMe"]


class PFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(PFedMeOptimizer, self).__init__(params, defaults)

    def step(
        self, local_parameters: list[torch.nn.Parameter], closure: callable = None
    ) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure
        for group in self.param_groups:
            for param_p, param_l in zip(group["params"], local_parameters):
                param_p.data = param_p.data - group["lr"] * (
                    param_p.grad.data
                    + group["lamda"] * (param_p.data - param_l.data)
                    + group["mu"] * param_p.data
                )
        return loss


class PFedMeClient(Client):
    def __init__(
        self,
        index: int,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: torch.nn.Module,
        local_epochs: int,
        k: int,
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
        self.hyper_params.update(k=k)
        self.internal_model: Module | None = None
        self._attr_to_cache.append("internal_model")

    def receive_model(self) -> None:
        model = self.channel.receive(self.index, "server", msg_type="model").payload
        if self.model is None:
            self.model = model
            self.internal_model = deepcopy(model)
        else:
            safe_load_state_dict(self.model, model.state_dict())

    def _model_to_dataparallel(self):
        super()._model_to_dataparallel()
        self.internal_model = torch.nn.DataParallel(
            self.internal_model, device_ids=FlukeENV().get_device_ids()
        )

    def _dataparallel_to_model(self):
        super()._dataparallel_to_model()
        self.internal_model = self.internal_model.module

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )
        self.internal_model.to(self.device)
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        lamda = self.optimizer.defaults["lamda"]
        running_loss = 0.0
        loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                for _ in range(self.hyper_params.k):
                    self.optimizer.zero_grad()
                    y_hat = self.model(X)
                    loss = self.hyper_params.loss_fn(y_hat, y)
                    loss.backward()
                    self._clip_grads(self.model)
                    self.optimizer.step(self.model.parameters())

                lr = self.optimizer.param_groups[0]["lr"]
                params = zip(self.model.parameters(), self.internal_model.parameters())
                for param_p, param_l in params:
                    param_l.data = param_l.data - lamda * lr * (param_l.data - param_p.data)
                running_loss += loss.item()
            self.scheduler.step()
        running_loss /= epochs * len(self.train_set)
        self.internal_model.load_state_dict(self.model.state_dict())
        self.internal_model.cpu()
        self.model.cpu()
        return running_loss


class PFedMeServer(Server):
    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader,
        clients: Sequence[Client],
        weighted: bool = False,
        beta: float = 0.5,
        **kwargs,
    ):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.hyper_params.update(beta=beta)

    @torch.no_grad()
    def aggregate(self, eligible: Sequence[Client], client_models: Collection[Module]) -> None:
        weights = self._get_client_weights(eligible)
        agg_model_sd = aggregate_models(
            self.model, client_models, weights, self.hyper_params.lr, inplace=False
        ).state_dict()

        for key, param in self.model.named_parameters():
            param.data = (1 - self.hyper_params.beta) * param.data
            param.data += self.hyper_params.beta * agg_model_sd[key]


class PFedMe(CentralizedFL):

    def can_override_optimizer(self) -> bool:
        return False

    def get_optimizer_class(self) -> type[torch.optim.Optimizer]:
        return PFedMeOptimizer

    def get_client_class(self) -> type[Client]:
        return PFedMeClient

    def get_server_class(self) -> type[Server]:
        return PFedMeServer
