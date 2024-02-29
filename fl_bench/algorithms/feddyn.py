from collections import OrderedDict
import sys; sys.path.append(".")

from copy import deepcopy
from typing import Callable, Iterable, Any
import numpy as np

import torch
from torch.nn import Module
from algorithms import CentralizedFL
from fl_bench import Message
from server import Server

from fl_bench.utils import DDict, OptimizerConfigurator, get_loss
from fl_bench.client import Client
from fl_bench.data import FastTensorDataLoader


def get_all_params_of(model, copy=True) -> torch.Tensor:
    # restituisce un unico tensore che contiene tutti i parametri del modello
    result = None
    for param in model.parameters():
        if result == None:
            result = param.clone().detach().reshape(-1) if copy else param.reshape(-1)
        else:
            result = torch.cat((result, param.clone().detach().reshape(-1)), 0) if copy else torch.cat((result, param.reshape(-1)), 0)
    return result

def load_all_params(device, model, params):
    dict_param = deepcopy(dict(model.named_parameters()))
    idx = 0
    for name, param in model.named_parameters():
        weights = param.data 
        length = len(weights.reshape(-1)) # numero di elementi nel parametro
        dict_param[name].data.copy_(params[idx:idx+length].clone().detach().reshape(weights.shape).to(device))
        idx += length
    
    model.load_state_dict(dict_param)    



class FedDynServer(Server):
    def __init__(self,
                 model: Module,
                 test_data: FastTensorDataLoader,
                 clients: Iterable[Client],
                 alpha: float=0.01):
        super().__init__(model, test_data, clients, False)
        self.alpha = alpha
        self.cld_mdl = deepcopy(self.model)

    def _broadcast_model(self, eligible: Iterable[Client]) -> None:
        self.channel.broadcast(Message((self.model, self.cld_mdl), "model", self), eligible)

    def aggregate(self, eligible: Iterable[Client]) -> None:
        
        avg_model_sd = OrderedDict()
        clients_sd = self._get_client_models(eligible, state_dict=False)
        
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0].state_dict()[key])
                    continue
                den = 0
                for i, client_sd in enumerate(clients_sd):
                    client_sd = client_sd.state_dict()
                    weight = 1 if not self.weighted else eligible[i].n_examples
                    den += weight
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weight * client_sd[key]
                    else:
                        avg_model_sd[key] += weight * client_sd[key]
                avg_model_sd[key] /= den

            avg_grad = None
            grad_count = 0
            for cl in self.clients:
                if cl.prev_grads != None:
                    grad_count += 1
                    if avg_grad == None:
                        avg_grad = cl.prev_grads.clone().detach()
                    else:
                        avg_grad += cl.prev_grads.clone().detach()

            if grad_count > 0:
                avg_grad /= grad_count

            # load_all_params(self.device, self.model, avg_model_sd)
            self.model.load_state_dict(avg_model_sd)
            load_all_params(self.device, self.cld_mdl, get_all_params_of(self.model) + avg_grad)
        

class FedDynClient(Client):
    
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 alpha: float,
                 optimizer_cfg: OptimizerConfigurator,
                 weight_decay: float,
                 loss_fn: Callable,
                 weight_list: float,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3):
        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.alpha = alpha             
        self.weight_list = weight_list
        self.weight_decay = weight_decay             
                    
        self.prev_grads = None 
    

    def _receive_model(self) -> None:
        model, cld_mdl = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.model = deepcopy(model)
            self.prev_grads = torch.zeros_like(get_all_params_of(self.model))
        else:
            self.model.load_state_dict(cld_mdl.state_dict())
              
        
    def local_train(self, override_local_epochs: int=0):
        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        self._receive_model()

        alpha_coef_adpt = self.alpha / self.weight_list # adaptive alpha coef da implementazione ufficiale 
        
        server_params = get_all_params_of(self.model)

        for params in self.model.parameters():
            params.requires_grad = True

        if not self.optimizer:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model, weight_decay=alpha_coef_adpt + self.weight_decay)

        self.model.train()

        for e in range(epochs):
            loss = None
            for i, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)

                # Dynamic regularization
                curr_params = get_all_params_of(self.model, False)
                penalty = alpha_coef_adpt * torch.sum(curr_params * (-server_params + self.prev_grads))
                loss = loss + penalty

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()

            self.scheduler.step()

        # update the previous gradients 
        curr_params = get_all_params_of(self.model)
        self.prev_grads += curr_params - server_params
        
        self._send_model()
    
    def __str__(self) -> str:
        to_str = super().__str__()
        return f"{to_str[:-1]},alpha={self.alpha}, weight_decay={self.weight_decay})"
        

class FedDyn(CentralizedFL):
    """FedDyn Federated Learning Environment."""
    
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict):
        scheduler_kwargs = config.optimizer.scheduler_kwargs
        optimizer_args = config.optimizer
        del optimizer_args['scheduler_kwargs']
        optimizer_cfg = OptimizerConfigurator(self.get_optimizer_class(), 
                                              **optimizer_args,
                                              scheduler_kwargs=scheduler_kwargs)
        self.loss = get_loss(config.loss)
        weight_list = np.asarray([clients_tr_data[i].tensors[0].shape[0] for i in range(self.n_clients)])
        weight_list = weight_list / np.sum(weight_list) * self.n_clients 
    
        self.clients = [FedDynClient(train_set=clients_tr_data[i], 
                                        optimizer_cfg=optimizer_cfg, 
                                        loss_fn=self.loss,
                                        weight_decay=config.weight_decay,
                                        weight_list=weight_list[i],
                                        validation_set=clients_te_data[i],
                                        alpha=config.alpha,
                                        local_epochs=config.n_epochs) for i in range(self.n_clients)]
        
    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = FedDynServer(model,
                                   data,
                                   self.clients,
                                   **config)
    
    
    