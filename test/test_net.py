from torch import nn
import torch
import torch.nn.functional as F

class MNIST_2NN_TEST(nn.Module):
    def __init__(self, 
                 hidden_size: tuple[int, int]=(200, 200),
                 softmax: bool=False):
        super(MNIST_2NN_TEST, self).__init__()
        self.input_size = 28*28
        self.output_size = 10
        self.use_softmax = softmax

        self.fc1 = nn.Linear(28*28, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_softmax:
            return F.softmax(self.fc3(x), dim=1)
        else:
            return torch.sigmoid(self.fc3(x))



import torch
from copy import deepcopy
from typing import Callable

from fl_bench.algorithms import CentralizedFL
from fl_bench.client import Client
from fl_bench.data import FastTensorDataLoader
from fl_bench.utils import OptimizerConfigurator, clear_cache

class MyFedProxClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastTensorDataLoader,
                 test_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 mu: float):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update({
            "mu": mu
        })

    
    def _proximal_loss(self, local_model, global_model):
        proximal_term = 0.0
        for w, w_t in zip(local_model.parameters(), global_model.parameters()):
            proximal_term += torch.norm(w - w_t)**2
        return proximal_term

    def fit(self, override_local_epochs: int=0):
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self._receive_model()
        W = deepcopy(self.model)
        self.model.to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y) + (self.hyper_params.mu / 2) * self._proximal_loss(self.model, W)
                loss.backward()
                self.optimizer.step()          
            self.scheduler.step()

        self.model.to("cpu")
        clear_cache()
        self._send_model()


class MyFedProx(CentralizedFL):

    def get_client_class(self) -> Client:
        return MyFedProxClient