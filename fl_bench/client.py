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

    def __init__(self,
                 train_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable, # CHECK ME
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3,
                 seed: int=42):

        self.seed = seed
        self.train_set = train_set
        self.validation_set = validation_set
        self.n_examples = train_set.size
        self.model = None
        self.optimizer_cfg = optimizer_cfg
        self.loss_fn = loss_fn
        self.local_epochs = local_epochs
        self.stateful = False
        self.optimizer = None
        self.device = GlobalSettings().get_device()

    def send(self):
        return deepcopy(self.model)
    
    def receive(self, model):
        if self.model is None:
            self.control = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
            self.model = deepcopy(model)
        else:
            self.model.load_state_dict(model.state_dict())
    
    def local_train(self, override_local_epochs: int=0):
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        # total_step = len(self.dataset)
        self.model.train()
        if self.optimizer is None:
            self.optimizer = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()          
            
                #if log_interval and (i+1) % log_interval == 0:
                #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                #        .format(epoch + 1, self.local_epochs, i + 1, total_step, loss.item()))
        
        return self.validate()
    
    def validate(self):
        if self.validation_set is not None:
            n_classes = len(torch.unique(self.validation_set.tensors[1]))
            return ClassificationEval(self.validation_set, self.loss_fn, n_classes).evaluate(self.model)
        

