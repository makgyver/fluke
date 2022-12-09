from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import Iterable, Callable
import numpy as np 
from torch.nn import Module
from rich.progress import track, Progress

from client import Client


class Server(ABC):

    def __init__(self,
                 model: Module,
                 clients: Iterable[Client], 
                 elegibility_percentage: float=0.5, 
                 seed: int=42):
        self.model = model
        self.clients = clients
        self.n_clients = len(clients)
        self.elegibility_percentage = elegibility_percentage
        self.seed = seed
        self.callbacks = []
        np.random.seed(self.seed)
    
    def fit(self, n_rounds: int=10, log_interval: int=0):
        with Progress() as progress:
            client_x_round = int(self.n_clients*self.elegibility_percentage)
            task_rounds = progress.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            task_local = progress.add_task("[green]Local Training", total=client_x_round)

            for round in range(n_rounds):
                eligible = self.get_eligible_clients()
                self.broadcast(eligible)
                for c, client in enumerate(eligible):
                    client.local_train(log_interval=log_interval)
                    progress.update(task_id=task_local, completed=c + 1)
                    progress.update(task_id=task_rounds, advance=1)
                self.aggregate(eligible)
                self.notify_all(self.model, round + 1)


    def get_eligible_clients(self):
        if self.elegibility_percentage == 1:
            return self.clients
        n = int(self.n_clients * self.elegibility_percentage)
        return np.random.choice(self.clients, n)

    def broadcast(self, eligible: Iterable[Client]=None) -> Iterable[Client]:
        eligible = eligible if eligible is not None else self.clients
        for client in eligible:
            client.receive(deepcopy(self.model))

    def init(self):
        pass

    # def aggregate(self):
    #     w = [self.clients[i].send().state_dict() for i in range(len(self.clients))]
    #     weights_avg = deepcopy(w[0])
    #     for k in weights_avg.keys():
    #         for i in range(1, self.n_clients):
    #             weights_avg[k] += w[i][k]

    #     weights_avg[k] = torch.div(weights_avg[k], len(w))
    #     self.model.load_state_dict(weights_avg)
    
    def aggregate(self, eligible: Iterable[Client], weighted: bool=False) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = [eligible[i].send().state_dict() for i in range(len(eligible))]
        for key in self.model.state_dict().keys():
            den = 0
            for i, client_sd in enumerate(clients_sd):
                weight = 1 if not weighted else eligible[i].n_examples
                den += weight
                if key not in avg_model_sd:
                    avg_model_sd[key] = weight * client_sd[key]
                else:
                    avg_model_sd[key] += weight * client_sd[key]
            avg_model_sd[key] /= den
        self.model.load_state_dict(avg_model_sd)
    
    def register_callback(self, callback: Callable):
        if callback is not None and callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def notify_all(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)
    
