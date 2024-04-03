import sys
sys.path.append(".")
sys.path.append("..")

from copy import deepcopy
from typing import Callable, Iterable

import torch

from ..comm import Message
from ..algorithms import CentralizedFL
from ..client import Client
from ..server import Server
from ..data import FastTensorDataLoader
from ..utils import OptimizerConfigurator


class FedNovaClient(Client):
    
    def __init__(self,
                 index: int,
                 train_set: FastTensorDataLoader,
                 test_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int):
        super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.tau = 0
    
    def _get_momentum(self):
        if self.optimizer is None:
            if "momentum" in self.optimizer_cfg.optimizer_kwargs:
                return self.optimizer_cfg.optimizer_kwargs["momentum"]
            else:
                return 0
        else:
            return self.optimizer.param_groups[0]["momentum"]

    def local_train(self, override_local_epochs: int=0) -> None:
        super().local_train(override_local_epochs)
        self.tau += self.hyper_params.local_epochs * self.train_set.n_batches
        rho = self._get_momentum()
        self.a = (self.tau - rho * (1.0 - pow(rho, self.tau)) / (1.0 - rho)) / (1.0 - rho)
        self.channel.send(Message(self.a, "local_a", self), self.server)
    

class FedNovaServer(Server):
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        clients_sd = self._get_client_models(eligible)
        weights = self._get_client_weights(eligible)
        a_i = [
            self.channel.receive(self, client, "local_a").payload
            for client in eligible
        ]

        coeff = sum([a_i[i] * weights[i] for i in range(len(eligible))])
        avg_model_sd = deepcopy(self.model.state_dict())
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    # avg_model_sd[key] = clients_sd[0][key].clone()
                    continue
                    
                for i, client_sd in enumerate(clients_sd):
                    avg_model_sd[key] += coeff * weights[i] * torch.true_divide(client_sd[key] - avg_model_sd[key], a_i[i])
            
            self.model.load_state_dict(avg_model_sd)


class FedNova(CentralizedFL):
    
    def get_client_class(self) -> Client:
        return FedNovaClient
    
    def get_server_class(self) -> Server:
        return FedNovaServer

