import sys; sys.path.append(".")

import torch
from copy import deepcopy
from typing import Iterable, Any
from collections import OrderedDict

from fl_bench import Message
from fl_bench.server import Server
from fl_bench.client import Client
from fl_bench.algorithms import CentralizedFL
from fl_bench.data import FastTensorDataLoader
from fl_bench.utils import DDict, OptimizerConfigurator, get_loss


class FedBNClient(Client):

    def receive(self, message: Message) -> None:
        if message.msg_type == "model":
            model = message.payload
            if self.model is None:
                self.model = deepcopy(model)
            else:
                with torch.no_grad():
                    for key in model.state_dict().keys():
                        if not key.startswith("bn"):
                            self.model.state_dict()[key].data.copy_(model.state_dict()[key])


class FedBNServer(Server):

    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self._get_client_models(eligible)
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0][key])
                    continue
                #elif key.startswith("bn"):
                #    avg_model_sd[key] = deepcopy(self.model.state_dict()[key])
                #    continue
                den = 0
                for i, client_sd in enumerate(clients_sd):
                    weight = 1 if not self.weighted else eligible[i].n_examples
                    den += weight
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weight * client_sd[key]
                    else:
                        avg_model_sd[key] += weight * client_sd[key]
                avg_model_sd[key] /= den
            self.model.load_state_dict(avg_model_sd)


class FedBN(CentralizedFL):
    
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict):
        optimizer_cfg=OptimizerConfigurator(self.get_optimizer_class(), 
                                            lr=config.optimizer.lr, 
                                            scheduler_kwargs=config.optimizer.scheduler_kwargs)
        self.loss = get_loss(config.loss)
        self.clients = [FedBNClient(train_set=clients_tr_data[i],  
                                    optimizer_cfg=optimizer_cfg, 
                                    loss_fn=self.loss, 
                                    validation_set=clients_te_data[i],
                                    local_epochs=config.n_epochs) for i in range(self.n_clients)]
    
    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = FedBNServer(model, data, self.clients, **config)