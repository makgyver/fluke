import torch
from typing import Any, Callable, Sequence
import sys
sys.path.append(".")
sys.path.append("..")

from ..algorithms import PersonalizedFL  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from ..server import Server  # NOQA
from ..client import PFLClient  # NOQA
from ..data import FastDataLoader  # NOQA
from ..comm import Message  # NOQA
from ..nets import EncoderHeadNet, EncoderGlobalHeadLocalNet  # NOQA


# https://arxiv.org/abs/2102.07078
class FedRepClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable[..., Any],
                 local_epochs: int = 3,
                 tau: int = 3):
        super().__init__(index, EncoderGlobalHeadLocalNet(model),
                         train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.pers_optimizer = None
        self.pers_scheduler = None
        self.hyper_params.update(tau=tau)

    def fit(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()
        self.model.train()
        self.model.to(self.device)

        # update head layers
        for parameter in self.model.get_local().parameters():
            parameter.requires_grad = True
        for parameter in self.model.get_global().parameters():
            parameter.requires_grad = False

        if self.pers_optimizer is None:
            self.pers_optimizer, self.pers_scheduler = self.optimizer_cfg(self.model.get_local())

        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.pers_optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.pers_optimizer.step()
            self.pers_scheduler.step()

        # update encoder layers
        for parameter in self.model.get_local().parameters():
            parameter.requires_grad = False
        for parameter in self.model.get_global().parameters():
            parameter.requires_grad = True

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model.get_global())

        for _ in range(self.hyper_params.tau):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        self.model.to("cpu")
        clear_cache()
        self.send_model()

    def send_model(self):
        self.channel.send(Message(self.model.get_global(), "model", self), self.server)

    def receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model
        msg = self.channel.receive(self, self.server, msg_type="model")
        safe_load_state_dict(self.model.get_global(), msg.payload.state_dict())


class FedRepServer(Server):

    def __init__(self,
                 model: torch.nn.Module,
                 test_data: FastDataLoader,  # test_data is not used
                 clients: Sequence[PFLClient],
                 eval_every: int = 1,
                 weighted: bool = False):
        super().__init__(model, None, clients, eval_every, weighted)


class FedRep(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedRepClient

    def get_server_class(self) -> Server:
        return FedRepServer
