"""
The module ``fluke.client`` provides the base classes for the clients in ``fluke``.
"""
from __future__ import annotations
from torch import device
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from typing import Callable
import sys
sys.path.append(".")

from .evaluation import ClassificationEval  # NOQA
from .data import FastDataLoader  # NOQA
from .utils import OptimizerConfigurator, clear_cache  # NOQA
from .utils.model import safe_load_state_dict  # NOQA
from .comm import Channel, Message  # NOQA
from .server import Server  # NOQA
from . import GlobalSettings, DDict  # NOQA


class Client():
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

            When a new client class inherits from this class, it can add all its hyper-parameters
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
                 loss_fn: Callable,
                 local_epochs: int = 3):

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
            Use this channel to communicate with the server.

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
            # self.model.load_state_dict(msg.payload.state_dict())

    def send_model(self) -> None:
        """Send the current model to the server. The model is sent as a ``Message`` with
        ``msg_type`` "model" to the server. The method uses the channel to send the message.
        """
        self.channel.send(Message(self.model, "model", self), self.server)

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
        self.receive_model()
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
        self.send_model()

    def evaluate(self) -> dict[str, float]:
        """Evaluate the local model on the client's :attr:`test_set`. If the test set is not set or
        the client has not received the global model from the server, the method returns an empty
        dictionary.

        Warning:
            To date, only classification tasks are supported.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
            the results.
        """
        if self.test_set is not None and self.model is not None:
            return ClassificationEval(self.hyper_params.loss_fn,
                                      #   self.model.output_size,
                                      self.train_set.num_labels,
                                      self.device).evaluate(self.model,
                                                            self.test_set)
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
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int = 3):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.personalized_model: Module = model

    def evaluate(self) -> dict[str, float]:
        """Evaluate the personalized model on the ``test_set``.
        If the test set is not set or the client has no personalized model, the method returns an
        empty dictionary.

        Warning:
            To date, only classification tasks are supported.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
            the results.
        """
        if self.test_set is not None and self.personalized_model is not None:
            evaluator = ClassificationEval(self.hyper_params.loss_fn,
                                           #    self.personalized_model.output_size,
                                           self.train_set.num_labels,
                                           self.device)
            return evaluator.evaluate(self.personalized_model, self.test_set)

        return {}
