from __future__ import annotations
from torch import device
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import Callable, Dict
from copy import deepcopy
from abc import ABC
import sys
sys.path.append(".")

from .evaluation import ClassificationEval  # NOQA
from .data import FastTensorDataLoader  # NOQA
from .utils import OptimizerConfigurator, clear_cache  # NOQA
from .utils.model import safe_load_state_dict  # NOQA
from .comm import Channel, Message  # NOQA
from .server import Server  # NOQA
from . import GlobalSettings, DDict  # NOQA


class Client(ABC):
    """Base ``Client`` class. This class is the base class for all clients in the ``FLUKE``.
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

            When a new client class inherits from this class, it can add all its hyper-parameters
            to this dictionary.
        index (int): The client identifier.
        train_set (FastTensorDataLoader): The local training set.
        test_set (FastTensorDataLoader): The local test set.
        optimizer_cfg (OptimizerConfigurator): The optimizer configurator. This is used to create
            the optimizer and the learning rate scheduler.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device where the client trains the model.
        server (Server): The server.
        channel (Channel): The channel to communicate with the server.
    """

    def __init__(self,
                 index: int,
                 train_set: FastTensorDataLoader,
                 test_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int = 3):
        #  **additional_hyper_params):

        self.hyper_params: DDict = DDict(
            loss_fn=loss_fn,
            local_epochs=local_epochs
        )
        # self.hyper_params.update(**additional_hyper_params)

        self._index: int = index
        self.train_set: FastTensorDataLoader = train_set
        self.test_set: FastTensorDataLoader = test_set
        self.model: Module = None
        self.optimizer_cfg: OptimizerConfigurator = optimizer_cfg
        self.optimizer: Optimizer = None
        self.scheduler: _LRScheduler = None
        self.device: device = GlobalSettings().get_device()
        self._server: Server = None
        self._channel: Channel = None

    @property
    def n_examples(self) -> int:
        """The number of examples in the local training set.

        Returns:
            int: The number of examples in the local training set.
        """
        return self.train_set.size

    @property
    def index(self) -> int:
        """The client identifier.

        Returns:
            int: The client identifier.
        """
        return self._index

    @property
    def channel(self) -> Channel:
        """The communication channel.

        Returns:
            Channel: The communication channel.
        """
        return self._channel

    @property
    def server(self) -> Server:
        """The server to which the client is connected.

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

    def _receive_model(self) -> None:
        """Receive the global model from the server. This method is responsible for receiving the
        global model from the server and updating the local model accordingly. The model is received
        as a ``Message`` with  ``msg_type`` "model" from the inbox of the client iself.
        The method uses the channel to receive the message.
        """
        msg = self.channel.receive(self, self.server, msg_type="model")
        if self.model is None:
            self.model = deepcopy(msg.payload)
        else:
            safe_load_state_dict(self.model, msg.payload.state_dict())
            # self.model.load_state_dict(msg.payload.state_dict())

    def _send_model(self):
        """Send the current model to the server. The model is sent as a ``Message`` with
        ``msg_type`` "model" to the server. The method uses the channel to send the message.
        """
        self.channel.send(Message(deepcopy(self.model), "model", self), self.server)

    def fit(self, override_local_epochs: int = 0) -> None:
        """Client's local training.
        Before starting the training, the client receives the global model from the server.
        The training occurs for a number of ``hyper_params.local_epochs`` epochs using the local
        training set and as loss function the one defined in ``hyper_params.loss_fn``. The training
        happens on the device defined in the client.
        After the training, the client sends the model to the server.

        Args:
            override_local_epochs (int, optional): Overrides the number of local epochs,
                by default 0 (use the default number of local epochs).
        """
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)
        self._receive_model()
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        for _ in range(epochs):
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
        self._send_model()

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the local model on the client's ``test_set``. If the test set is not set,
        the method returns an empty dictionary.

        Warning:
            To date, only classification tasks are supported.

        Returns:
            Dict[str, float]: The evaluation results. The keys are the metrics and the values are
            the results.
        """
        if self.test_set is not None:
            if self.model is None:
                # ask for the model and receive it
                self.channel.send(Message(self.server.model, "model", self.server), self)
                self._receive_model()

            return ClassificationEval(self.hyper_params.loss_fn,
                                      self.model.output_size,
                                      self.device).evaluate(self.model,
                                                            self.test_set)
        return {}

    def finalize(self) -> None:
        """Finalize the client. This method is called at the end of the federated learning process.
        The default behavior is to receive the global model from the server that is then potentially
        used to evaluate the performance of the client's model on the local test set.
        """
        self._receive_model()

    def __str__(self) -> str:
        hpstr = ",".join([f"{h}={str(v)}" for h, v in self.hyper_params.items()])
        hpstr = "," + hpstr if hpstr else ""
        return f"{self.__class__.__name__}[{self._index}](optim={self.optimizer_cfg}," + \
               f"batch_size={self.train_set._batch_size}{hpstr})"

    def __repr__(self) -> str:
        return str(self)


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
                 train_set: FastTensorDataLoader,
                 test_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int = 3):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.personalized_model: Module = model

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the personalized model on the ``test_set``.
        If the test set is not set, the method returns an empty dictionary.

        Warning:
            To date, only classification tasks are supported.

        Returns:
            Dict[str, float]: The evaluation results. The keys are the metrics and the values are
            the results.
        """
        if self.test_set is not None:
            evaluator = ClassificationEval(self.hyper_params.loss_fn,
                                           self.personalized_model.output_size,
                                           self.device)
            return evaluator.evaluate(self.personalized_model, self.test_set)

        return {}
