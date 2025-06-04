"""Implementation of the FedAMP [FedAMP21]_ algorithm.

References:
    .. [FedAMP21] Yutao Huang, Lingyang Chu, Zirui Zhou, Lanjun Wang, Jiangchuan Liu, Jian Pei, Yong
       Zhang. Personalized Cross-Silo Federated Learning on Non-IID Data. In AAAI (2021).
       URL: https://arxiv.org/abs/2007.03797
"""

import sys
from copy import deepcopy
from typing import Collection, Sequence

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..client import PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from . import PersonalizedFL  # NOQA


class FedAMPClient(PFLClient):
    def __init__(
        self,
        index: int,
        model: Module,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: torch.nn.Module,
        local_epochs: int,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        lam: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            index=index,
            model=model,
            train_set=train_set,
            test_set=test_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss_fn,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            **kwargs,
        )
        self.hyper_params.update(lam=lam)

    def _alpha(self):
        return self.optimizer.param_groups[0]["lr"]

    def _proximal_loss(self, local_model: Module, u_model: Module) -> float:
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), u_model.parameters()):
            proximal_term += torch.norm(w - w_t) ** 2
        return (self.hyper_params.lam / (2 * self._alpha())) * proximal_term

    def receive_model(self) -> None:
        try:
            msg = self.channel.receive(self.index, "server", msg_type="model")
            safe_load_state_dict(self.personalized_model, msg.payload.state_dict())
        except ValueError:
            pass

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )
        self.model.to(self.device)
        self.personalized_model.to(self.device)
        self.model.train()

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y) + self._proximal_loss(
                    self.model, self.personalized_model
                )
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= epochs * len(self.train_set)
        self.model.cpu()
        self.personalized_model.cpu()
        clear_cuda_cache()

        return running_loss

    def _load_from_cache(self) -> None:
        super()._load_from_cache()
        if self.model is None:
            self.model = deepcopy(self.personalized_model)


class FedAMPServer(Server):

    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader,  # not used
        clients: Sequence[PFLClient],
        sigma: float = 0.1,
        alpha: float = 0.1,
        **kwargs,
    ):
        super().__init__(model=model, test_set=None, clients=clients, weighted=False)
        self.hyper_params.update(sigma=sigma, alpha=alpha)

    def __e(self, x: float) -> float:
        return torch.exp(-x / self.hyper_params.sigma) / self.hyper_params.sigma

    def _empty_model(self) -> Module:
        empty_model = deepcopy(self.model)
        for param in empty_model.parameters():
            param.data.zero_()
        return empty_model

    @torch.no_grad()
    def aggregate(self, eligible: Sequence[PFLClient], client_models: Collection[Module]) -> None:

        client_models = list(client_models)
        for i, (client, ci_model) in enumerate(zip(eligible, client_models)):
            ui_model = self._empty_model()

            coef = torch.zeros(len(eligible))
            for j, cj_model in enumerate(client_models):
                if i != j:
                    weights_i = torch.cat([p.data.view(-1) for p in ci_model.parameters()], dim=0)
                    weights_j = torch.cat([p.data.view(-1) for p in cj_model.parameters()], dim=0)
                    sub = (weights_i - weights_j).view(-1)
                    sub = torch.dot(sub, sub)
                    coef[j] = self.hyper_params.alpha * self.__e(sub)
            coef[i] = 1 - torch.sum(coef)

            for j, cj_model in enumerate(client_models):
                for param_i, param_j in zip(ui_model.parameters(), cj_model.parameters()):
                    param_i.data += coef[j] * param_j

            self.channel.send(Message(ui_model, "model", "server", inmemory=True), client.index)

    def broadcast_model(self, eligible: Sequence[PFLClient]) -> None:
        # Models have already been sent to clients in aggregate
        pass


class FedAMP(PersonalizedFL):

    def get_client_class(self) -> type[PFLClient]:
        return FedAMPClient

    def get_server_class(self) -> type[Server]:
        return FedAMPServer
