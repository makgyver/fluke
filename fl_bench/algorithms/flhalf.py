from collections import OrderedDict
from typing import Callable, Iterable

import torch
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader, TensorDataset

from client import Client
from server import Server
from data import Datasets
from utils import OptimizerConfigurator, print_params
from . import CentralizedFL



class FLHalfClient(Client):
    def __init__(self,
                 dataset: DataLoader,
                 private_layers: Iterable,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable, # CHECK ME
                 local_epochs: int=3,
                 device: torch.device=torch.device('cpu'),
                 seed: int=42):
        super().__init__(dataset, optimizer_cfg, loss_fn, local_epochs, device, seed)
        self.private_layers = private_layers
    
    def _generate_fake_examples(self):
        shape = self.dataset.dataset.data.shape
        fake_data = torch.rand(shape)
        fake_targets = self.model.forward_(fake_data)
        return fake_data, fake_targets
    
    def send(self):
        return super().send(), self._generate_fake_examples()


class FLHalfServer(Server):
    def __init__(self,
                 model: Module,
                 clients: Iterable[Client],
                 private_layers: Iterable,
                 global_step: float=.01,
                 elegibility_percentage: float=0.5, 
                 seed: int=42):
        super().__init__(model, clients, elegibility_percentage, seed)
        self.control = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.global_step = global_step
        self.private_layers = private_layers
    
    def _private_train(self, clients_fake_x, clients_fake_y, epochs=1, batch_size=32):
        train = TensorDataset(clients_fake_x, clients_fake_y)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        #model = FLHalfModule(self.model, self.private_layers)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.global_step)
        loss_fn = MSELoss()
        #print("BEFORE PRIVATE TRAIN")
        #print_params(self.model)
        for epoch in range(epochs):
            loss = None
            for _, (X, y) in enumerate(train_loader):
                #X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_hat = self.model.forward_(X)
                loss = loss_fn(y_hat, y)
                loss.backward(retain_graph=True)
                optimizer.step()
        #print("AFTER PRIVATE TRAIN")
        #print_params(self.model)

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
        for key in global_model_dict.keys():
            if key.split(".")[0] in self.private_layers:
                avg_model_sd[key] = global_model_dict[key]
                continue
            for i, client_sd in enumerate(clients_sd):
                if key not in avg_model_sd:
                    avg_model_sd[key] = client_sd[key]
                else:
                    avg_model_sd[key] += client_sd[key]
            avg_model_sd[key] /= len(eligible)
        self.model.load_state_dict(avg_model_sd)
        #print("AFTER AGGREGATE")
        #print_params(self.model)


class FLHalf(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int, 
                 batch_size: int, 
                 train_set: Datasets,
                 optimizer_cfg: OptimizerConfigurator, 
                 model: Module, 
                 private_layers: Iterable,
                 loss_fn: Callable, 
                 elegibility_percentage: float=0.5,
                 device: torch.device=torch.device('cpu'),
                 seed: int=42):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         batch_size,
                         train_set,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         elegibility_percentage,
                         device, 
                         seed)
        self.private_layers = private_layers

    def init_parties(self, callback: Callable=None):
        assert self.client_loader is not None, 'You must prepare data before initializing parties'
        self.clients = [FLHalfClient (dataset=self.client_loader[i], 
                                      private_layers=self.private_layers,
                                      optimizer_cfg=self.optimizer_cfg, 
                                      loss_fn=self.loss_fn, 
                                      local_epochs=self.n_epochs,
                                      device=self.device,
                                      seed=self.seed) for i in range(self.n_clients)]

        self.server = FLHalfServer(self.model.to(self.device),
                                   self.clients, 
                                   private_layers=self.private_layers, 
                                   elegibility_percentage=self.elegibility_percentage, 
                                   seed=self.seed)
        self.server.register_callback(callback)
