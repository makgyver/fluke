"""Implementation of the Federated Averaging with Body Aggregation and Body Update [FedBABU22]_
algorithm.

References:
    .. [FedBABU22] Jaehoon Oh, Sangmook Kim, Se-Young Yun. FedBABU: Towards Enhanced Representation
       for Federated Image Classification. In ICLR (2022). URL: https://arxiv.org/abs/2106.06042
"""
import sys
from typing import Any, Iterable

from rich.progress import Progress
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from .. import GlobalSettings  # NOQA
from ..algorithms import PersonalizedFL  # NOQA
from ..client import PFLClient  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..nets import EncoderHeadNet  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA


class FedBABUClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Module,
                 local_epochs: int,
                 mode: str,
                 fine_tune_epochs: int,
                 **kwargs: dict[str, Any]):
        assert mode in ["head", "body", "full"]
        super().__init__(index=index, model=model, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.hyper_params.update(
            mode=mode,
            fine_tune_epochs=fine_tune_epochs
        )
        self.model = self.personalized_model

    def send_model(self):
        self.channel.send(Message(self.personalized_model.encoder, "model", self), self.server)

    def receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        safe_load_state_dict(self.personalized_model.encoder, msg.payload.state_dict())

        # Deactivate gradient
        for param in self.personalized_model.head.parameters():
            param.requires_grad = False

    def fine_tune(self) -> None:
        """Fine-tune the personalized model."""

        if self.hyper_params.mode == "full":
            for param in self.personalized_model.parameters():
                param.requires_grad = True
        elif self.hyper_params.mode == "body":
            for param in self.personalized_model.encoder.parameters():
                param.requires_grad = True
            for param in self.personalized_model.head.parameters():
                param.requires_grad = False
        else:  # head
            for param in self.personalized_model.encoder.parameters():
                param.requires_grad = False
            for param in self.personalized_model.head.parameters():
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

    def finalize(self) -> None:
        metrics = self.evaluate(GlobalSettings().get_evaluator(), self.test_set)
        if metrics:
            self._notify_evaluation(-1, "post-fit", metrics)


class FedBABUServer(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,  # not used
                 clients: Iterable[PFLClient],
                 weighted: bool = False):
        super().__init__(model=model, test_set=None, clients=clients, weighted=weighted)

    def finalize(self) -> None:

        with Progress(transient=True) as progress:
            task = progress.add_task("[cyan]Client's fine tuning", total=len(self.clients))
            for client in self.clients:
                client.fine_tune()
                client.finalize()
                progress.update(task, advance=1)

        self._notify_finalize()


class FedBABU(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedBABUClient

    def get_server_class(self) -> Server:
        return FedBABUServer
