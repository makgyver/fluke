from __future__ import annotations
import sys
sys.path.append(".")

from abc import ABC
from copy import deepcopy
from typing import Callable, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.nn import Module
from torch import device

from . import GlobalSettings
from .server import Server
from .comm import Channel, Message
from .utils import DDict, OptimizerConfigurator, clear_cache
from .data import FastTensorDataLoader
from .evaluation import ClassificationEval



class Client(ABC):
    """Base Client class.

    This class is the base class for all clients in the `FL-bench`. The behavior of the client is 
    based on the Federated Averaging algorithm. The default behavior of a client includes:
    - Receiving the global model from the server;
    - Training the model locally for a number of epochs using the local training set;
    - Sending the updated local model back to the server;
    - (Optional) Evaluating the model on the local test set.

    Attributes:
        hyper_params (DDict): The hyper-parameters of the client. The default hyper-parameters are:
            - loss_fn: The loss function.
            - local_epochs: The number of local epochs.
            When a new client class inherits from this class, it can add all its hyper-parameters
            to this dictionary.
        index (int): The client identifier.
        train_set (FastTensorDataLoader): The local training set.
        test_set (FastTensorDataLoader): The local test set.
        optimizer_cfg (OptimizerConfigurator): The optimizer configurator. This is used to create
            the optimizer and the learning rate scheduler.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.LRScheduler): The learning rate scheduler.
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
                 local_epochs: int=3,
                 **additional_hyper_params):
        
        self.hyper_params: DDict = DDict({
            "loss_fn": loss_fn,
            "local_epochs": local_epochs
        })
        self.hyper_params.update(additional_hyper_params)

        self.index: int = index
        self.train_set: FastTensorDataLoader = train_set
        self.test_set: FastTensorDataLoader = test_set
        self.n_examples: int = train_set.size
        self.model: Module = None
        self.optimizer_cfg: OptimizerConfigurator = optimizer_cfg
        self.optimizer: Optimizer = None
        self.scheduler: Scheduler = None
        self.device: device = GlobalSettings().get_device()
        self.server: Server = None
        self.channel: Channel = None
    
    def set_server(self, server: Server) -> None:
        """Set the server.

        Along with the server, the channel is also set and the client must use this channel to
        communicate with the server.

        Args:
            server (Server): The server.
        """
        self.server = server
        self.channel = server.channel

    def _receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        if self.model is None:
            self.model = deepcopy(msg.payload)
        else:
            self.model.load_state_dict(msg.payload.state_dict())
    
    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model), "model", self), self.server)

    def local_train(self, override_local_epochs: int=0) -> None:
        """Client's local training.

        The training occurs for a number of `hyper_params.local_epochs` epochs using the local 
        training set and as loss function the one defined in `hyper_params.loss_fn`.
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
    
    def validate(self) -> Dict[str, float]:
        """Test the local model on the `test_set`.

        If the test set is not set, the method returns an empty dictionary.

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
                                      self.model.output_size).evaluate(self.model, 
                                                                       self.test_set)
        return {}

    def __str__(self) -> str:
        hpstr = ",".join([f"{h}={str(v)}" for h,v in self.hyper_params.items()])
        hpstr = "," + hpstr if hpstr else ""
        return f"{self.__class__.__name__}[{self.index}](optim={self.optimizer_cfg}, "+\
               f"batch_size={self.train_set.batch_size}{hpstr})"

    def __repr__(self) -> str:
        return super().__repr__()


class PFLClient(Client):
    """Personalized Federated Learning client.

    This class is a personalized version of the `Client` class. It is used to implement
    personalized federated learning algorithms. The main difference is that the client has a
    personalized model (`personalized_model`).

    The client evaluation is performed using the personalized model instead of the global model.

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
                 local_epochs: int=3):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.personalized_model: Module = model
    
    def validate(self) -> Dict[str, float]:
        """Test the personalized model on the :`test_set`.

        If the test set is not set, the method returns an empty dictionary.

        Warning:
            To date, only classification tasks are supported.

        Returns:
            Dict[str, float]: The evaluation results. The keys are the metrics and the values are 
                the results.
        """
        if self.test_set is not None:
            return ClassificationEval(self.hyper_params.loss_fn,
                                      self.personalized_model.output_size).evaluate(self.personalized_model, 
                                                                                    self.test_set)
        return {}