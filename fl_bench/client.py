from abc import ABC
from copy import deepcopy
from typing import Callable

import torch

import sys; sys.path.append(".")
from fl_bench import GlobalSettings
from fl_bench.utils import OptimizerConfigurator
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
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):

        self.train_set = train_set
        self.validation_set = validation_set
        self.n_examples = train_set.size
        self.model = None
        self.optimizer_cfg = optimizer_cfg
        self.loss_fn = loss_fn
        self.local_epochs = local_epochs
        self.stateful = False
        self.optimizer = None
        self.scheduler = None
        self.device = GlobalSettings().get_device()

    def send(self) -> torch.nn.Module:
        """Send the model to the server.

        Returns
        -------
        torch.nn.Module
            A deep copy of the local model.
        """
        return deepcopy(self.model)
    
    def receive(self, model) -> None:
        """Receive the model from the server.

        Parameters
        ----------
        model : torch.nn.Module
            The (gloal) model to be received.
        """
        if self.model is None:
            self.control = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
            self.model = deepcopy(model)
        else:
            self.model.load_state_dict(model.state_dict())
    
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
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()     
        return self.validate()
    
    def validate(self):
        """Validate/test the local model.

        Returns
        -------
        dict
            The evaluation results.
        """
        if self.validation_set is not None:
            n_classes = self.model.output_size
            return ClassificationEval(self.validation_set, self.loss_fn, n_classes).evaluate(self.model)
        

