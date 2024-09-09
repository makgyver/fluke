"""
The module ``fluke.client`` provides the base classes for the clients in ``fluke``.
"""
from __future__ import annotations

import sys
from typing import Any, Literal

import torch
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

sys.path.append(".")

from . import DDict, GlobalSettings, ObserverSubject  # NOQA
from .comm import Channel, Message  # NOQA
from .data import FastDataLoader  # NOQA
from .evaluation import Evaluator  # NOQA
from .server import Server  # NOQA
from .utils import OptimizerConfigurator, clear_cache  # NOQA
from .utils.model import safe_load_state_dict  # NOQA

__all__ = [
    "Client",
    "PFLClient"
]


class Client(ObserverSubject):
    """Base ``Client`` class. This class is the base class for all clients in the ``fluke``.
    The behavior of the client is based on the Federated Averaging algorithm. The default behavior
    of a client includes:

    - Receiving the global model from the server;
    - Training the model locally for a number of epochs using the local training set;
    - Sending the updated local model back to the server;
    - (Optional) Evaluating the model on the local test set.

    Attributes:
        hyper_params (DDict): The hyper-parameters of the client. The default hyper-parameters are:

            - ``loss_fn``: The loss function.
            - ``local_epochs``: The number of local epochs.

            When a new client class inherits from this class, it must add all its hyper-parameters
            to this dictionary.

        train_set (FastDataLoader): The local training set.
        test_set (FastDataLoader): The local test set.
        optimizer_cfg (OptimizerConfigurator): The optimizer configurator. This is used to create
          the optimizer and the learning rate scheduler.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
        device (torch.device): The device where the client trains the model. By default, it is the
          device defined in :class:`fluke.GlobalSettings`.

    Attention:
        **The client should not directly call methods of the server**. The communication between the
        client and the server must be done through the :attr:`channel`.

    Caution:
        When inheriting from this class, make sure to put all the specific hyper-parameters in the
        :attr:`hyper_params` attribute. In this way ``fluke`` can properly handle the
        hyper-parameters of the client in the federated learning process.

        For example:

        .. code-block:: python
            :linenos:

            class MyClient(Client):
                # We omit the type hints for brevity
                def __init__(self, index, train_set, test_set, optimizer_cfg, loss_fn, my_param):
                    super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn)
                    self.hyper_params.update(my_param=my_param) # This is important
    """

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Module,
                 local_epochs: int = 3,
                 **kwargs: dict[str, Any]):
        super().__init__()
        self.hyper_params: DDict = DDict(
            loss_fn=loss_fn,
            local_epochs=local_epochs
        )

        self._index: int = index
        self.train_set: FastDataLoader = train_set
        self.test_set: FastDataLoader = test_set
        self.model: Module = None
        self.optimizer_cfg: OptimizerConfigurator = optimizer_cfg
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None
        self.device: device = GlobalSettings().get_device()
        self._server: Server = None
        self._channel: Channel = None
        self._last_round: int = 0

    @property
    def index(self) -> int:
        """The client identifier. This might be useful to identify the client in the federated
        learning process for logging or debugging purposes.

        Returns:
            int: The client identifier.
        """
        return self._index

    @property
    def n_examples(self) -> int:
        """The number of examples in the local training set.

        Returns:
            int: The number of examples in the local training set.
        """
        return self.train_set.size

    @property
    def channel(self) -> Channel:
        """The communication channel.

        Attention:
            Use this channel to exchange data/information with the server.

        Returns:
            Channel: The communication channel.
        """
        return self._channel

    @property
    def server(self) -> Server:
        """The server to which the client is connected. This reference must only be
        used to send messages through the channel.

        Returns:
            Server: The server.
        """
        return self._server

    def set_server(self, server: Server) -> None:
        """Set the reference to the server. Along with the server, the communication channel is also
        set and the client must use this channel to communicate with the server.

        Args:
            server (Server): The server that orchestrates the federated learning process.
        """
        self._server = server
        self._channel = server.channel

    def receive_model(self) -> None:
        """Receive the global model from the server. This method is responsible for receiving the
        global model from the server and updating the local model accordingly. The model is received
        as a payload of a :class:`fluke.comm.Message` with  ``msg_type="model"`` from the inbox
        of the client itself. The method uses the channel to receive the message.
        """
        msg = self.channel.receive(self, self.server, msg_type="model")
        if self.model is None:
            self.model = msg.payload
        else:
            safe_load_state_dict(self.model, msg.payload.state_dict())

    def send_model(self) -> None:
        """Send the current model to the server. The model is sent as a ``Message`` with
        ``msg_type`` "model" to the server. The method uses the channel to send the message.
        """
        self.channel.send(Message(self.model, "model", self), self.server)

    def local_update(self, current_round: int) -> None:
        """Client's local update procedure.
        Before starting the local training, the client receives the global model from the server.
        Then, the training occurs for a number of ``hyper_params.local_epochs`` epochs using the
        local training set and as loss function the one defined in ``hyper_params.loss_fn``.
        The training happens on the device defined in the client. After the training, the client
        sends the model to the server.

        Args:
            current_round (int): The current round of the federated learning process.
        """
        self._last_round = current_round
        self.receive_model()

        if GlobalSettings().get_eval_cfg().pre_fit:
            metrics = self.evaluate(GlobalSettings().get_evaluator(), self.test_set)
            if metrics:
                self._notify_evaluation(current_round, "pre-fit", metrics)

        self._notify_start_fit(current_round)
        loss = self.fit()
        self._notify_end_fit(current_round, loss)

        if GlobalSettings().get_eval_cfg().post_fit:
            metrics = self.evaluate(GlobalSettings().get_evaluator(), self.test_set)
            if metrics:
                self._notify_evaluation(current_round, "post-fit", metrics)

        self.send_model()

    def fit(self, override_local_epochs: int = 0) -> float:
        """Client's local training procedure.

        Args:
            override_local_epochs (int, optional): Overrides the number of local epochs,
                by default 0 (use the default number of local epochs).

        Returns:
            float: The average loss of the model during the training.
        """
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.model.to("cpu")
        clear_cache()
        return running_loss

    def evaluate(self,
                 evaluator: Evaluator,
                 test_set: FastDataLoader) -> dict[str, float]:
        """Evaluate the local model on the client's :attr:`test_set`. If the test set is not set or
        the client has not received the global model from the server, the method returns an empty
        dictionary.

        Args:
            evaluator (Evaluator): The evaluator to use for the evaluation.
            test_set (FastDataLoader): The test set to use for the evaluation.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
            the results.
        """
        if test_set is not None and self.model is not None:
            return evaluator.evaluate(self._last_round, self.model, test_set)
        return {}

    def finalize(self) -> None:
        """Finalize the client. This method is called at the end of the federated learning process.
        The default behavior is to receive the global model from the server that is then potentially
        used to evaluate the performance of the model on the local test set.

        Attention:
            When inheriting from this class, make sure to override this method if this behavior is
            not desired.
        """
        self.receive_model()

        if GlobalSettings().get_eval_cfg().pre_fit:
            metrics = self.evaluate(GlobalSettings().get_evaluator(), self.test_set)
            if metrics:
                self._notify_evaluation(-1, "pre-fit", metrics)

    def state_dict(self) -> dict:
        """Get the client state as a dictionary.

        Returns:
            dict: The client state.
        """
        return {
            "model": self.model.state_dict() if self.model is not None else None,
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "index": self.index,
            "last_round": self._last_round
        }

    def save(self, path: str) -> None:
        """Save the client state to a file.

        Args:
            path (str): The path to the file where the client state will be saved.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the client state from a file.

        Args:
            path (str): The path to the file where the client state is saved.
        """
        state = torch.load(path, weights_only=True)
        if "model" in state and state["model"] is not None:
            self.model.load_state_dict(state["model"])
            if state["optimizer"] is not None:
                self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
                self.optimizer.load_state_dict(state["optimizer"])
                if state["scheduler"] is not None:
                    self.scheduler.load_state_dict(state["scheduler"])
        else:
            self.model = None

        self._index = state["index"]
        self._last_round = state["last_round"]

    def __str__(self) -> str:
        hpstr = ", ".join([f"{h}={str(v)}" for h, v in self.hyper_params.items()])
        hpstr = ", " + hpstr if hpstr else ""
        return f"{self.__class__.__name__}[{self._index}](optim={self.optimizer_cfg}, " + \
               f"batch_size={self.train_set._batch_size}{hpstr})"

    def __repr__(self) -> str:
        return str(self)

    def _notify_evaluation(self,
                           round: int,
                           phase: Literal["pre-fit", "post-fit"],
                           metrics: dict[str, float]) -> None:
        for obs in self._observers:
            obs.client_evaluation(round, self.index, phase, metrics)

    def _notify_start_fit(self, round: int) -> None:
        for obs in self._observers:
            obs.start_fit(round, self.index, self.model)

    def _notify_end_fit(self, round: int, loss: float) -> None:
        for obs in self._observers:
            obs.end_fit(round, self.index, self.model, loss)


class PFLClient(Client):
    """Personalized Federated Learning client.
    This class is a personalized version of the ``Client`` class. It is used to implement
    personalized federated learning algorithms. The main difference is that the client has a
    personalized model (i.e., the attribute ``personalized_model``).

    Note:
        The client evaluation is performed using ``personalized_model`` instead of the global model
        (i.e., ``model``).

    Attributes:
        personalized_model (torch.nn.Module): The personalized model.
    """

    def __init__(self,
                 index: int,
                 model: Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Module,
                 local_epochs: int = 3,
                 **kwargs: dict[str, Any]):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.personalized_model: Module = model

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        """Evaluate the personalized model on the ``test_set``.
        If the test set is not set or the client has no personalized model, the method returns an
        empty dictionary.

        Args:
            evaluator (Evaluator): The evaluator to use for the evaluation.
            test_set (FastDataLoader): The test set to use for the evaluation.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
            the results.
        """
        if test_set is not None and self.personalized_model is not None:
            return evaluator.evaluate(self._last_round, self.personalized_model, test_set)
        return {}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["personalized_model"] = \
            self.personalized_model.state_dict() if self.personalized_model is not None else None
        return state

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only=True)
        if "model" in state and state["model"] is not None:
            self.model.load_state_dict(state["model"])
            if state["optimizer"] is not None:
                self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
                self.optimizer.load_state_dict(state["optimizer"])
                if state["scheduler"] is not None:
                    self.scheduler.load_state_dict(state["scheduler"])
        else:
            self.model = None

        if "personalized_model" in state and state["personalized_model"] is not None:
            self.personalized_model.load_state_dict(state["personalized_model"])
        else:
            self.personalized_model = None

        self._index = state["index"]
        self._last_round = state["last_round"]
