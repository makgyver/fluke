from abc import ABC
from copy import deepcopy
from typing import Callable

import torch
from torch.utils.data import DataLoader

from utils import OptimizerConfigurator


class Client(ABC):

    def __init__(self,
                 dataset: DataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable, # CHECK ME
                 local_epochs: int=3,
                 device: torch.device=torch.device('cpu'),
                 seed: int=42):

        self.seed = seed
        self.dataset = dataset
        self.n_examples = len(dataset.dataset)
        self.model = None
        self.optimizer_cfg = optimizer_cfg
        self.loss_fn = loss_fn
        self.local_epochs = local_epochs
        self.device = device
        self.stateful = False
        self.optimizer = None

    def send(self):
        return deepcopy(self.model)
    
    def receive(self, model):
        if self.model is None:
            self.control = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
            self.model = deepcopy(model)
        else:
            self.model.load_state_dict(model.state_dict())
    
    def local_train(self, override_local_epochs: int=0, log_interval: int=0):
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        total_step = len(self.dataset)
        self.model.train()
        if self.optimizer is None:
            self.optimizer = self.optimizer_cfg(self.model)
        for epoch in range(epochs):
            loss = None
            for i, (X, y) in enumerate(self.dataset):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()          
            
                #if log_interval and (i+1) % log_interval == 0:
                #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                #        .format(epoch + 1, self.local_epochs, i + 1, total_step, loss.item()))
        return None # CHECK ME
        

