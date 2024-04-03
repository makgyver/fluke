import sys
sys.path.append(".")
sys.path.append("..")
from copy import deepcopy
from typing import Callable, Iterable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from rich.progress import Progress

from .. import GlobalSettings
from ..comm import Message
from ..evaluation import ClassificationEval
from ..client import Client, PFLClient
from ..server import Server
from ..data import FastTensorDataLoader
from ..utils import OptimizerConfigurator
from ..algorithms import PersonalizedFL
from ..net import GlobalLocalNet
    
def relative_projection(encoder: nn.Module, 
                        x: torch.Tensor, 
                        anchors: torch.Tensor, 
                        normalize_out: bool=True) -> torch.Tensor:
    enc_x = encoder(x)
    enc_a = encoder(anchors)
    x = F.normalize(enc_x, p=2, dim=-1)
    anchors = F.normalize(enc_a, p=2, dim=-1)
    rel_proj =  torch.einsum("bm, am -> ba", x, anchors)
    return rel_proj if not normalize_out else F.normalize(rel_proj, p=2, dim=-1)


class FLHalfClient(PFLClient):
    def __init__(self,
                 index: int,
                 model: GlobalLocalNet,
                 train_set: FastTensorDataLoader,
                 test_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 tau: int,
                 relative: bool=True):
        super().__init__(index, 
                         model, 
                         train_set, 
                         test_set,
                         optimizer_cfg, 
                         loss_fn, 
                         local_epochs)
        # self.personalized_model.init()
        self.hyper_params.update({
            "tau": tau,
            "relative": relative
        })
        self.anchors = None

    def _private_train(self):
        if self.anchors is None:
            self.anchors = self.channel.receive(self, self.server, msg_type="anchors").payload

        self.personalized_model.train()
        self.private_optimizer, self.private_scheduler = self.optimizer_cfg(self.personalized_model)
        for _ in range(self.hyper_params.tau):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.private_optimizer.zero_grad()
                y_hat = self.personalized_model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.private_optimizer.step()
            self.private_scheduler.step()
            
    def _receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        if self.model is None:
            self.model = deepcopy(msg.payload)
        self.model.load_state_dict(msg.payload.state_dict())

    def local_train(self, override_local_epochs: int=0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self._receive_model()
        self.model.train()

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                rel_x = (self.personalized_model.forward_local(X) if not self.hyper_params.relative 
                         else relative_projection(self.personalized_model.get_local(), 
                                                  X.view(X.size(0), -1), 
                                                  self.anchors))
                y_hat = self.model(rel_x)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        self._send_model()

    def validate(self):
        if self.test_set is not None:
            n_classes = self.model.output_size
            if self.hyper_params.relative:
                test_loader = self.test_set.transform(
                    lambda x: relative_projection(self.personalized_model.forward_local, 
                                                  x.view(x.size(0), -1), 
                                                  self.anchors)
                )
            else:
                test_loader = self.test_set.transform(lambda x: self.personalized_model.forward_local(x))
            return ClassificationEval(self.hyper_params.loss_fn, n_classes).evaluate(self.model, test_loader)



class FLHalfServer(Server):
    def __init__(self,
                 model: Module,
                 test_data: FastTensorDataLoader,
                 clients: Iterable[Client],
                 n_anchors: int=100,
                 seed_anchors: int=98765,
                 weighted: bool=True):
        super().__init__(model, None, clients, weighted)
        self.n_anchors = n_anchors
        self.seed_anchors = seed_anchors

    def fit(self, n_rounds: int=10, eligible_perc: float=0.1) -> None:
        GlobalSettings().set_seed(self.seed_anchors)
        anchors =  torch.randn((self.n_anchors, 784)) #FIXME
        for client in self.clients:
            self.channel.send(Message(anchors, "anchors", self), client)

        # Preparation step
        # the following code run private_train across all clients with progress bar
        with Progress() as progress:
            task = progress.add_task("[cyan]Client's Private Training", total=len(self.clients))
            for client in self.clients:
                self.channel.send(Message((client._private_train, {}), "__action__", self), client)
                progress.update(task, advance=1)
        
        # Training step
        super().fit(n_rounds, eligible_perc)


class FLHalf(PersonalizedFL):
    
    # def get_optimizer_class(self) -> torch.optim.Optimizer:
    #     return torch.optim.Adam
    
    def get_client_class(self) -> Client:
        return FLHalfClient

    def get_server_class(self) -> Server:
        return FLHalfServer