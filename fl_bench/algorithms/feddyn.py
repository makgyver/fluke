from copy import deepcopy
from typing import Callable, Iterable, Union, Any, Optional
import numpy as np

import torch
from torch.nn import Module
from algorithms import CentralizedFL
from server import Server

import sys; sys.path.append(".")
from fl_bench.utils import OptimizerConfigurator
from fl_bench.client import Client
from fl_bench.data import DataSplitter, FastTensorDataLoader


def get_all_params_of(model) -> torch.Tensor:
    # restituisce un unico tensore che contiene tutti i parametri del modello
    result = None
    for param in model.parameters():
        if result == None:
            result = param.clone().detach().reshape(-1)
        else:
            result = torch.cat((result, param.clone().detach().reshape(-1)), 0)
    return result

def wrap_all_params_of(model) -> torch.Tensor:
    result = None
    for param in model.parameters():
        if result == None:
            result = param.reshape(-1)
        else:
            result = torch.cat((result, param.reshape(-1)), 0)
    return result

def load_all_params(device, model, params):
    # aggiornamento parametri model 
    dict_param = deepcopy(dict(model.named_parameters()))
    idx = 0
    for name, param in model.named_parameters():
        weights = param.data 
        length = len(weights.reshape(-1)) # numero di elementi nel parametro
        #dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        # copia i valori dei parametri dal vettore params al parametro corrispondente nel modello
        dict_param[name].data.copy_(params[idx:idx+length].clone().detach().reshape(weights.shape).to(device))
        idx += length
    
    model.load_state_dict(dict_param)    



class FedDynServer(Server):
    def __init__(self,
                 model: Module,
                 clients: Iterable[Client],
                 eligibility_percentage: float=0.5,
                 alpha: float=0.01):
        super().__init__(model, clients, eligibility_percentage)
        self.alpha = alpha
        self.cld_mdl = deepcopy(self.model)
        

    def aggregate(self, eligible: Iterable[Client]) -> None:
        
        clients_sd = {key: eligible[key].send() for key in range(len(eligible))}
        num_participants = len(eligible)
        with torch.no_grad():

            # da implementazione ufficiale:
            # https://github.com/alpemreacar/FedDyn/tree/48a19fac440ef079ce563da8e0c2896f8256fef9
            #avg_mdl_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            #cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)

            avg_mdl = None
            for idx in clients_sd.keys():
                client_params = get_all_params_of(clients_sd[idx])
                if avg_mdl == None:
                    avg_mdl = client_params
                else:
                    avg_mdl += client_params

            avg_mdl /= num_participants

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

            load_all_params(self.device, self.model, avg_mdl)
            load_all_params(self.device, self.cld_mdl, avg_mdl + avg_grad)


    def broadcast(self, eligible: Iterable[Client]=None) -> None:
        eligible = eligible if eligible is not None else self.clients
        for client in eligible:
            client.receive(deepcopy(self.model), self.cld_mdl)   
        

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
        
                     
    def receive(self, model, cld_mdl):
        if self.model is None:
            self.model = deepcopy(model)
            # Initialize the first gradient to zero
            self.prev_grads = torch.zeros_like(get_all_params_of(self.model))
        else:
            #self.model.load_state_dict(deepcopy(cld_mdl.state_dict()))
            self.model = deepcopy(cld_mdl)
              
        
    def local_train(self, override_local_epochs: int=0):

        n_trn = self.train_set.size

        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        alpha_coef_adpt = self.alpha / self.weight_list # adaptive alpha coef da implementazione ufficiale 
        
        server_params = get_all_params_of(self.model)

        for params in self.model.parameters():
            params.requires_grad = True

        #self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        #self.optimizer, self.scheduler = self.optimizer_cfg(self.model, alpha_coef_adpt + self.weight_decay)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay= alpha_coef_adpt + self.weight_decay)
        self.model.train()

        for e in range(epochs):
            # Training 
            epoch_loss = 0 
            loss = None
            for i, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)

                # --- Dynamic regularization --- #
                curr_params = wrap_all_params_of(self.model)
                penalty = alpha_coef_adpt * torch.sum(curr_params * (-server_params + self.prev_grads))
                loss = loss + penalty

                self.optimizer.zero_grad() 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                epoch_loss += loss.item() * list(y.size())[0]

            if (e+1) % 5 == 0:
                epoch_loss /= n_trn
                if self.weight_decay != None:
                    # Add L2 loss to complete f_i
                    curr_params = get_all_params_of(self.model)
                    epoch_loss += (alpha_coef_adpt+self.weight_decay)/2 * torch.sum(curr_params * curr_params)
                print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
                self.model.train()

        for params in self.model.parameters():
            params.requires_grad = False
        self.model.eval()

        # update the previous gradients 
        curr_params = get_all_params_of(self.model)
        self.prev_grads += curr_params - server_params
        
        return self.validate
        

class FedDyn(CentralizedFL):
    """FedDyn Federated Learning Environment.
    Parameters
    ----------
    n_clients : int
        Number of clients in the FL environment.
    n_rounds : int
        Number of communication rounds.
    n_epochs : int
        Number of epochs per communication round.
    optimizer_cfg : OptimizerConfigurator
        Optimizer configurator for the clients.
    model : torch.nn.Module
        Model to be trained.
    loss_fn : Callable
        Loss function.
    eligibility_percentage : float, optional
        Percentage of clients to be selected for each communication round, by default 0.5.
    alpha : float, optional
        By default 0.1. 
    """
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 n_epochs: int,
                 optimizer_cfg: OptimizerConfigurator,
                 model: Module,
                 alpha: float,
                 weight_decay: float,
                 loss_fn: Callable,
                 eligibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model, 
                         optimizer_cfg, 
                         loss_fn,
                         eligibility_percentage)
        self.alpha = alpha
        self.weight_decay = weight_decay
    
    def init_parties(self, 
                     data_splitter: DataSplitter, 
                     callbacks: Optional[Union[Any, Iterable[Any]]]=None):
        
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
                         
        weight_list = np.asarray([data_splitter.client_train_loader[i].tensors[0].shape[0] for i in range(self.n_clients)])
        weight_list = weight_list / np.sum(weight_list) * self.n_clients 
    
        self.clients = [FedDynClient(train_set=data_splitter.client_train_loader[i], 
                                        optimizer_cfg=self.optimizer_cfg, 
                                        weight_decay = self.weight_decay,
                                        loss_fn=self.loss_fn, 
                                        weight_list = weight_list[i],
                                        validation_set=data_splitter.client_test_loader[i],
                                        alpha = self.alpha,
                                        local_epochs=self.n_epochs) for i in range(self.n_clients)]
                         
                        
    
                         
        self.server = FedDynServer(self.model, 
                                     self.clients, 
                                     self.eligibility_percentage,
                                     self.alpha)
        self.server.attach(callbacks)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
               f"A={self.alpha},P={self.eligibility_percentage},{self.optimizer_cfg})"
    
    
    