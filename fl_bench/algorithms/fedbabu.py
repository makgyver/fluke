import sys
sys.path.append(".")
sys.path.append("..")

from torch.nn import Module
from typing import Any, Callable, Sequence
from copy import deepcopy
from rich.progress import Progress

from .. import Message
from ..server import Server
from ..algorithms import PersonalizedFL
from ..client import PFLClient
from ..data import FastTensorDataLoader
from ..utils import OptimizerConfigurator, clear_cache
from ..net  import EncoderHeadNet


class FedBABUClient(PFLClient):

    def __init__(self, 
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastTensorDataLoader, 
                 test_set: FastTensorDataLoader, 
                 optimizer_cfg: OptimizerConfigurator, 
                 loss_fn: Callable[..., Any],
                 local_epochs: int,
                 mode: str,
                 fine_tune_epochs: int):
        assert mode in ["head", "body", "full"]
        super().__init__(index, model, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update({
            "mode": mode,
            "fine_tune_epochs": fine_tune_epochs
        })
        self.model = self.personalized_model

    def _send_model(self):
        self.channel.send(Message(deepcopy(self.personalized_model.get_encoder()), "model", self), self.server)

    def _receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.personalized_model.get_encoder().load_state_dict(msg.payload.state_dict())

        # Deactivate gradient
        for param in self.personalized_model.get_head().parameters():
            param.requires_grad = False
    
    def _fine_tune(self):
        if self.hyper_params.mode == "full":
            for param in self.personalized_model.parameters():
                param.requires_grad = True
        elif self.hyper_params.mode == "body":
            for param in self.personalized_model.get_encoder().parameters():
                param.requires_grad = True
            for param in self.personalized_model.get_head().parameters():
                param.requires_grad = False
        else: # head
            for param in self.personalized_model.get_encoder().parameters():
                param.requires_grad = False
            for param in self.personalized_model.get_head().parameters():
                param.requires_grad = True
        
        self.personalized_model.train()
        self.personalized_model.to(self.device)
        self.optimizer, self.scheduler = self.optimizer_cfg(self.personalized_model)
        
        for _ in range(self.hyper_params.fine_tune_epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.personalized_model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        
        self.personalized_model.to("cpu")
        clear_cache()


class FedBABUServer(Server):

    def __init__(self, 
                 model: Module, 
                 test_data: FastTensorDataLoader, 
                 clients: Sequence[PFLClient], 
                 weighted: bool = False):
        super().__init__(model, None, clients, weighted)

    def finalize(self) -> None:

        with Progress() as progress:
            task = progress.add_task("[cyan]Client's fine tuning", total=len(self.clients))
            for client in self.clients:
                self.channel.send(Message((client._fine_tune, {}), "__action__", self), client)
                progress.update(task, advance=1)

        client_evals = [client.validate() for client in self.clients]
        self.notify_finalize(client_evals if client_evals[0] else None)


class FedBABU(PersonalizedFL):
    
    def get_client_class(self) -> PFLClient:
        return FedBABUClient

    def get_server_class(self) -> Server:
        return FedBABUServer