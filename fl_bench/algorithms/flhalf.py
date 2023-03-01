from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Iterable

import torch
from torch.nn import Module, MSELoss

import sys; sys.path.append(".")
from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench.utils import OptimizerConfigurator
from fl_bench.algorithms import CentralizedFL


class FLHalfClient(Client):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 private_layers: Iterable,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):
        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.private_layers = private_layers
    
    def _generate_fake_examples(self):
        shape = list(self.train_set.tensors[0].shape)
        fake_data = torch.rand(shape)
        _, fake_targets = self.model.forward_(fake_data, len(self.private_layers))
        return fake_data, fake_targets
    
    def receive(self, model):
        if self.model is None:
            self.model = deepcopy(model)
        else:
            with torch.no_grad():
                for key in model.state_dict().keys():
                    skip = bool(sum([key.startswith(n) for n in self.private_layers]))
                    if not skip:
                        self.model.state_dict()[key].data.copy_(model.state_dict()[key])
    
    def send(self):
        return super().send(), self._generate_fake_examples()


class FLHalfServer(Server):
    def __init__(self,
                 model: Module,
                 clients: Iterable[Client],
                 private_layers: Iterable,
                 n_epochs: int,
                 batch_size: int,
                 optimizer_cfg: OptimizerConfigurator,
                 global_step: float=.05,
                 eligibility_percentage: float=0.5,):
        super().__init__(model, clients, eligibility_percentage)
        self.control = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.global_step = global_step
        self.private_layers = private_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer, self.scheduler = optimizer_cfg(self.model)
    
    def _private_train(self, clients_fake_x, clients_fake_y):
        train_loader = FastTensorDataLoader(clients_fake_x, 
                                            clients_fake_y, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        loss_fn = MSELoss()
        for _ in range(self.n_epochs):
            loss = None
            for _, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                _, y_hat = self.model.forward_(X)
                loss = loss_fn(y_hat, y)
                loss.backward(retain_graph=True)
                self.optimizer.step()
            self.scheduler.step()

    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = []
        clients_fake_x = []
        clients_fake_y = []
        for i in range(len(eligible)):
            client_model, (client_fake_x, client_fake_y) = eligible[i].send()
            clients_sd.append(client_model.state_dict())
            clients_fake_x.append(client_fake_x)
            clients_fake_y.append(client_fake_y)
        
        self._private_train(torch.cat(clients_fake_x, 0), torch.cat(clients_fake_y, 0))

        global_model_dict = self.model.state_dict()
        with torch.no_grad():
            for key in global_model_dict.keys():
                if key.split(".")[0] in self.private_layers:
                    avg_model_sd[key] = deepcopy(global_model_dict[key])
                    continue
                elif "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0][key])
                    continue

                den = 0
                for i, client_sd in enumerate(clients_sd):
                    weight = eligible[i].n_examples
                    den += weight
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weight * client_sd[key]
                    else:
                        avg_model_sd[key] += weight * client_sd[key]
                avg_model_sd[key] /= den #len(eligible)
        self.model.load_state_dict(avg_model_sd)


class FLHalf(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int, 
                 optimizer_cfg: OptimizerConfigurator, 
                 model: Module, 
                 loss_fn: Callable, 
                 eligibility_percentage: float,
                 private_layers: Iterable,
                 server_n_epochs: int,
                 server_batch_size: int,
                 server_optimizer_cfg: OptimizerConfigurator=OptimizerConfigurator(torch.optim.SGD, lr=0.01)):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         eligibility_percentage)
        self.private_layers = private_layers
        self.server_n_epochs = server_n_epochs
        self.server_batch_size = server_batch_size
        self.server_optimizer_cfg = server_optimizer_cfg

    def init_parties(self, data_splitter: DataSplitter, callback: Callable=None):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.clients = [FLHalfClient (train_set=data_splitter.client_train_loader[i], 
                                      private_layers=self.private_layers,
                                      optimizer_cfg=self.optimizer_cfg, 
                                      loss_fn=self.loss_fn, 
                                      validation_set=data_splitter.client_test_loader[i],
                                      local_epochs=self.n_epochs) for i in range(self.n_clients)]

        self.server = FLHalfServer(self.model,
                                   self.clients, 
                                   private_layers=self.private_layers, 
                                   n_epochs=self.server_n_epochs,
                                   batch_size=self.server_batch_size,
                                   optimizer_cfg=self.server_optimizer_cfg,
                                   eligibility_percentage=self.eligibility_percentage)
        self.server.register_callback(callback)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
               f"P={self.eligibility_percentage},{self.optimizer_cfg}, pri={self.private_layers}," + \
               f"SE={self.server_n_epochs},SB={self.server_batch_size})"