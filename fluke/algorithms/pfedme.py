from torch.optim import Optimizer
from torch.nn import Module
import torch
from typing import Callable, Sequence
from collections import OrderedDict
from copy import deepcopy
import sys
sys.path.append(".")
sys.path.append("..")

from ..utils import OptimizerConfigurator  # NOQA
from ..utils.model import STATE_DICT_KEYS_TO_IGNORE, safe_load_state_dict  # NOQA
from ..data import FastDataLoader  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..server import Server  # NOQA
from ..client import Client, PFLClient  # NOQA
from ..comm import Message  # NOQA


class PFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(PFedMeOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, local_parameters: list[torch.nn.Parameter]):
        group = None
        for group in self.param_groups:
            for param_p, param_l in zip(group["params"], local_parameters):
                param_p.data = param_p.data - group["lr"] * (
                    param_p.grad.data
                    + group["lamda"] * (param_p.data - param_l.data)
                    + group["mu"] * param_p.data
                )


class PFedMeClient(PFLClient):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 k: int):

        super().__init__(index, None, test_set, train_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update(k=k)

    def receive_model(self) -> None:
        model = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.personalized_model = model
            self.model = deepcopy(self.personalized_model)
        else:
            safe_load_state_dict(self.personalized_model, model.state_dict())

    def fit(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()
        self.personalized_model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.personalized_model)

        lamda = self.optimizer.defaults["lamda"]
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                for _ in range(self.hyper_params.k):
                    self.optimizer.zero_grad()
                    y_hat = self.personalized_model(X)
                    loss = self.hyper_params.loss_fn(y_hat, y)
                    loss.backward()
                    self.optimizer.step(self.model.parameters())

                lr = self.optimizer.param_groups[0]["lr"]
                params = zip(self.personalized_model.parameters(), self.model.parameters())
                for param_p, param_l in params:
                    param_l.data = param_l.data - lamda * lr * (param_l.data - param_p.data)

            self.scheduler.step()
        self.personalized_model.load_state_dict(self.model.state_dict())
        self.send_model()

    def send_model(self):
        self.channel.send(Message(self.model, "model", self), self.server)


class PFedMeServer(Server):
    def __init__(self,
                 model: Module,
                 test_data: FastDataLoader,
                 clients: Sequence[Client],
                 eval_every: int = 1,
                 weighted: bool = False,
                 beta: float = 0.5):
        super().__init__(model, test_data, clients, eval_every, weighted)
        self.hyper_params.update(beta=beta)

    @torch.no_grad()
    def aggregate(self, eligible: Sequence[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self.get_client_models(eligible)
        weights = self._get_client_weights(eligible)
        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                # avg_model_sd[key] = clients_sd[0][key].clone()
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue
            for i, client_sd in enumerate(clients_sd):
                if key not in avg_model_sd:
                    avg_model_sd[key] = weights[i] * client_sd[key]
                else:
                    avg_model_sd[key] += weights[i] * client_sd[key]

        for key, param in self.model.named_parameters():
            param.data = (1 - self.hyper_params.beta) * param.data
            param.data += self.hyper_params.beta * avg_model_sd[key]


class PFedMe(CentralizedFL):
    def get_optimizer_class(self) -> torch.optim.Optimizer:
        return PFedMeOptimizer

    def get_client_class(self) -> Client:
        return PFedMeClient

    def get_server_class(self) -> Server:
        return PFedMeServer
