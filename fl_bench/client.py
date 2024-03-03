from __future__ import annotations
import sys

import torch; sys.path.append(".")

from abc import ABC
from copy import deepcopy
from typing import Callable

from fl_bench.server import Server
from fl_bench import GlobalSettings, Message
from fl_bench.utils import DDict, OptimizerConfigurator, clear_cache
from fl_bench.data import FastTensorDataLoader
from fl_bench.evaluation import ClassificationEval


class Client(ABC):
    """Standard client of a federated learning system.

    Parameters
    ----------
    train_set : FastTensorDataLoader
        The local training set.
    optimizer_cfg : OptimizerConfigurator
        The optimizer configurator.
    loss_fn : Callable
        The loss function.
    validation_set : FastTensorDataLoader, optional
        The local validation/test set, by default None.
    local_epochs : int, optional
        The number of local epochs, by default 3.
    """
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 validation_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int=3):
        self.hyper_params = DDict({
            "loss_fn": loss_fn,
            "local_epochs": local_epochs
        })
        self.train_set = train_set
        self.validation_set = validation_set
        self.n_examples = train_set.size
        self.model = None
        self.optimizer_cfg = optimizer_cfg
        self.optimizer = None
        self.scheduler = None
        self.device = GlobalSettings().get_device()
        self.server = None
    
    def set_server(self, server: Server):
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

    def local_train(self, override_local_epochs: int=0) -> dict:
        """Train the local model.

        Parameters
        ----------
        override_local_epochs : int, optional
            Override the number of local epochs, by default 0. If 0, use the default value.
        
        Returns
        -------
        dict
            The evaluation results if the validation set is not None, otherwise None.
        """
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
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
    
    def validate(self):
        """Validate/test the local model.

        Returns
        -------
        dict
            The evaluation results.
        """
        if self.validation_set is not None:
            return ClassificationEval(self.hyper_params.loss_fn,
                                      self.model.output_size).evaluate(self.model, 
                                                                       self.validation_set)
    
    def checkpoint(self):
        """Checkpoint the optimizer and the scheduler.
        
        Returns
        -------
        dict
            The checkpoint. 
        """

        return {
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None
        }

    def restore(self, checkpoint):
        """Restore the optimizer and the scheduler from a checkpoint.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint.
        """
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        

    def __str__(self) -> str:
        hpstr = ",".join([f"{h}={str(v)}" for h,v in self.hyper_params.items()])
        hpstr = "," + hpstr if hpstr else ""
        return f"{self.__class__.__name__}(optim={self.optimizer_cfg}, "+\
               f"batch_size={self.train_set.batch_size}{hpstr})"


class PFLClient(Client):

    def __init__(self,
                 model: torch.nn.Module,
                 train_set: FastTensorDataLoader,
                 validation_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int=3):
        super().__init__(train_set, validation_set, optimizer_cfg, loss_fn, local_epochs)
        self.private_model = model
    
    def validate(self):
        if self.validation_set is not None:
            return ClassificationEval(self.hyper_params.loss_fn,
                                      self.private_model.output_size).evaluate(self.private_model, 
                                                                               self.validation_set)