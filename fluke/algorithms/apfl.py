"""Implementation of the APFL [APFL20]_ algorithm.

References:
    .. [APFL20] Yuyang Deng, Mohammad Mahdi Kamani, and Mehrdad Mahdavi. Adaptive Personalized
       Federated Learning. In arXiv (2020). URL: https://arxiv.org/abs/2003.13461
"""

import sys
from typing import Collection, Sequence

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..algorithms import PersonalizedFL  # NOQA
from ..client import Client, PFLClient  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from ..utils.model import merge_models  # NOQA

__all__ = ["APFLClient", "APFLServer", "APFL"]


class APFLClient(PFLClient):

    def __init__(
        self,
        index: int,
        model: torch.nn.Module,
        train_set: FastDataLoader,
        test_set: FastDataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,
        local_epochs: int = 3,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        lam: float = 0.25,
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

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )

        self.model.train()
        self.personalized_model.train()

        self.model.to(self.device)
        self.personalized_model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        if self.pers_optimizer is None:
            self.pers_optimizer, self.pers_scheduler = self._optimizer_cfg(self.personalized_model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)

                # Global
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()

                # Local
                self.pers_optimizer.zero_grad()
                y_hat = merge_models(self.model, self.personalized_model, self.hyper_params.lam)(X)
                local_loss = self.hyper_params.loss_fn(y_hat, y)
                local_loss.backward()
                self._clip_grads(self.personalized_model)
                self.pers_optimizer.step()

            self.scheduler.step()
            self.pers_scheduler.step()

        running_loss /= epochs * len(self.train_set)
        self.model.cpu()
        self.personalized_model.cpu()
        clear_cuda_cache()

        self.personalized_model = merge_models(
            self.model, self.personalized_model, self.hyper_params.lam
        )

        return running_loss


class APFLServer(Server):

    def __init__(
        self,
        model: Module,
        test_set: FastDataLoader,
        clients: Sequence[Client],
        weighted: bool = False,
        tau: int = 3,
        **kwargs,
    ):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.hyper_params.update(tau=tau)

    @torch.no_grad()
    def aggregate(self, eligible: Sequence[Client], client_models: Collection[Module]) -> None:
        """Aggregate the models of the eligible clients every `hyper_params.tau` rounds.

        Args:
            eligible (Sequence[Client]): The clients that are eligible to participate in the
                aggregation.
            client_models (Collection[Module]): The models of the clients to aggregate.
        """
        if self.rounds % self.hyper_params.tau != 0:
            # Ignore the sent models and clear the channel's cache
            self.channel.clear(self)
        else:
            super().aggregate(eligible, client_models)


class APFL(PersonalizedFL):

    def get_client_class(self) -> type[PFLClient]:
        return APFLClient

    def get_server_class(self) -> type[Server]:
        return APFLServer
