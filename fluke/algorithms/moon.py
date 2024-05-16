from torch.nn import CosineSimilarity
import torch
from typing import Callable
from copy import deepcopy
import sys
sys.path.append(".")
sys.path.append("..")

from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from ..data import FastDataLoader  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA


class MOONClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 mu: float,
                 tau: float):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update(
            mu=mu,
            tau=tau
        )
        self.prev_model = None
        self.server_model = None

    def receive_model(self) -> None:
        model = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            # one of these deepcopy is not needed
            self.model = deepcopy(model)
            self.prev_model = deepcopy(model)
        else:
            self.prev_model.load_state_dict(deepcopy(self.model.state_dict()))
            safe_load_state_dict(self.model, deepcopy(model.state_dict()))
        self.server_model = model

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
                self.optimizer.zero_grad()

                y_hat = self.model(X)
                z_local = self.model.forward_encoder(X)  # , -1)
                loss_sup = self.hyper_params.loss_fn(y_hat, y)

                z_prev = self.prev_model.forward_encoder(X)  # , -1)
                z_global = self.server_model.forward_encoder(X)  # , -1)

                sim_lg = cos(z_local, z_global).reshape(-1, 1) / self.hyper_params.tau
                sim_lp = cos(z_local, z_prev).reshape(-1, 1) / self.hyper_params.tau
                loss_con = -torch.log(torch.exp(sim_lg) /
                                      (torch.exp(sim_lg) + torch.exp(sim_lp))).mean()

                loss = loss_sup + self.hyper_params.mu * loss_con
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.prev_model.to("cpu")
        self.server_model.to("cpu")
        self.model.to("cpu")
        clear_cache()
        self.send_model()


class MOON(CentralizedFL):

    def get_client_class(self) -> Client:
        return MOONClient
