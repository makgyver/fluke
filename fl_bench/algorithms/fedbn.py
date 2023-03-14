from collections import OrderedDict
from copy import deepcopy
from typing import Iterable, Any, Optional, Union
import torch

import sys; sys.path.append(".")
from fl_bench.server import Server
from fl_bench.client import Client
from fl_bench.data import DataSplitter;
from fl_bench.algorithms import CentralizedFL
    

class FedBNClient(Client):
    def receive(self, model):
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
        clients_sd = [eligible[i].send().state_dict() for i in range(len(eligible))]
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
    
    def init_parties(self, 
                     data_splitter: DataSplitter, 
                     callbacks: Optional[Union[Any, Iterable[Any]]]=None):
        assert data_splitter.n_clients == self.n_clients, \
            "Number of clients in data splitter and the FL environment must be the same"
        self.data_assignment = data_splitter.assignments
        self.clients = [FedBNClient(train_set=data_splitter.client_train_loader[i], 
                                    optimizer_cfg=self.optimizer_cfg, 
                                    loss_fn=self.loss_fn, 
                                    validation_set=data_splitter.client_test_loader[i],
                                    local_epochs=self.n_epochs) for i in range(self.n_clients)]

        self.server = FedBNServer(self.model, self.clients, self.eligibility_percentage, weighted=True)
        self.server.attach(callbacks)