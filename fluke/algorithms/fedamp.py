"""Implementation of the FedAMP [FedAMP21]_ algorithm.

References:
    .. [FedAMP21] Yutao Huang, Lingyang Chu, Zirui Zhou, Lanjun Wang, Jiangchuan Liu, Jian Pei, Yong
       Zhang. Personalized Cross-Silo Federated Learning on Non-IID Data. In AAAI (2021).
       URL: https://arxiv.org/abs/2007.03797
"""
import sys
from copy import deepcopy
from typing import Any, Iterable

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..client import PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from . import PersonalizedFL  # NOQA


class FedAMPClient(PFLClient):
    def __init__(self,
                 index: int,
                 model: Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int,
                 lam: float,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, model=model, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.hyper_params.update(lam=lam)
        self.model = deepcopy(self.personalized_model)

    def _alpha(self):
        return self.optimizer.param_groups[0]["lr"]

    def _proximal_loss(self, local_model, u_model):
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), u_model.parameters()):
            proximal_term += torch.norm(w - w_t)**2
        return (self.hyper_params.lam / (2 * self._alpha())) * proximal_term

    def receive_model(self) -> None:
        try:
            msg = self.channel.receive(self, self.server, msg_type="model")
            safe_load_state_dict(self.personalized_model, msg.payload.state_dict())
        except ValueError:
            pass

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.model.to(self.device)
        self.personalized_model.to(self.device)
        self.model.train()

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(
                    y_hat, y) + self._proximal_loss(self.model, self.personalized_model)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.model.to("cpu")
        self.personalized_model.to("cpu")
        clear_cache()

        return running_loss


class FedAMPServer(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,  # not used
                 clients: Iterable[PFLClient],
                 sigma: float = 0.1,
                 alpha: float = 0.1):
        super().__init__(model=model, test_set=None, clients=clients, weighted=False)
        self.hyper_params.update(
            sigma=sigma,
            alpha=alpha
        )

    def __e(self, x: float):
        return torch.exp(-x / self.hyper_params.sigma) / self.hyper_params.sigma

    def _empty_model(self):
        empty_model = deepcopy(self.model)
        for param in empty_model.parameters():
            param.data.zero_()
        return empty_model

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[PFLClient]) -> None:
        clients_model = self.get_client_models(eligible, state_dict=False)
        clients_model = [client for client in clients_model]

        for i, client in enumerate(eligible):
            ci_model = clients_model[i]
            ui_model = self._empty_model()

            coef = torch.zeros(len(eligible))
            for j, cj_model in enumerate(clients_model):
                if i != j:
                    weights_i = torch.cat([p.data.view(-1)
                                           for p in ci_model.parameters()], dim=0)
                    weights_j = torch.cat([p.data.view(-1)
                                           for p in cj_model.parameters()], dim=0)
                    sub = (weights_i - weights_j).view(-1)
                    sub = torch.dot(sub, sub)
                    coef[j] = self.hyper_params.alpha * self.__e(sub)
            coef[i] = 1 - torch.sum(coef)

            for j, cj_model in enumerate(clients_model):
                for param_i, param_j in zip(ui_model.parameters(), cj_model.parameters()):
                    param_i.data += coef[j] * param_j

            self.channel.send(Message(ui_model, "model", self), client)

    def broadcast_model(self, eligible: Iterable[PFLClient]) -> None:
        # Models have already been sent to clients in aggregate
        pass


class FedAMP(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedAMPClient

    def get_server_class(self) -> Server:
        return FedAMPServer
