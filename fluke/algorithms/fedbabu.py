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

from .. import FlukeENV  # NOQA
from ..algorithms import PersonalizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..nets import EncoderHeadNet  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cuda_cache  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA


class FedBABUClient(Client):

    def __init__(self,
                 index: int,
                 model: EncoderHeadNet,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Module,
                 local_epochs: int,
                 mode: str,
                 fine_tuning_epochs: int,
                 clipping: float = 0,
                 **kwargs: dict[str, Any]):
        assert mode in ["head", "body", "full"]
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, clipping=clipping, **kwargs)
        self.hyper_params.update(mode=mode)
        self._load_from_cache()
        self.model = model
        self._save_to_cache()

    def send_model(self):
        self.channel.send(Message(self.model.encoder,
                          "model", self, inmemory=True), self.server)

    def receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        safe_load_state_dict(self.model.encoder, msg.payload.state_dict())

        # Deactivate gradient
        for param in self.model.head.parameters():
            param.requires_grad = False

    def fine_tune(self) -> None:
        """Fine-tune the personalized model."""

        self._load_from_cache()
        if self.hyper_params.mode == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        elif self.hyper_params.mode == "body":
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            for param in self.model.head.parameters():
                param.requires_grad = False
        else:  # head
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True

        self.model.train()
        self.model.to(self.device)
        self.pers_optimizer, self.pers_scheduler = self._optimizer_cfg(self.model)

        for _ in range(self.hyper_params.fine_tuning_epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.pers_optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.pers_optimizer.step()
            self.pers_scheduler.step()

        self.model.cpu()
        clear_cuda_cache()
        self._save_to_cache()

    def finalize(self) -> None:
        self._load_from_cache()
        metrics = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
        if metrics:
            self._notify_evaluation(-1, "post-fit", metrics)
        self._save_to_cache()


class FedBABUServer(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,  # not used
                 clients: Iterable[Client],
                 weighted: bool = False):
        super().__init__(model=model, test_set=None, clients=clients, weighted=weighted)

    def finalize(self) -> None:

        with Progress(transient=True) as progress:
            task = progress.add_task("[cyan]Client's fine tuning", total=len(self._participants))
            clients_ft = [client for client in self.clients if client.index in self._participants]
            for client in clients_ft:
                client.fine_tune()
                client.finalize()
                progress.update(task, advance=1)

        self._notify_finalize()


class FedBABU(PersonalizedFL):

    def get_client_class(self) -> Client:
        return FedBABUClient

    def get_server_class(self) -> Server:
        return FedBABUServer
