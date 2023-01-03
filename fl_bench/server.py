from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import Iterable, Callable
import numpy as np 
import torch
from torch.nn import Module
from rich.progress import Progress

from fl_bench.client import Client
from fl_bench import GlobalSettings

class Server(ABC):
    """Standard Server for Federated Learning.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    clients : Iterable[Client]
        The clients participating in the federation.
    elegibility_percentage : float, optional
        The percentage of clients that will be selected for each round, by default 0.5.
    weighted : bool, optional
        Whether to weight the clients by their number of samples, by default False.
    """
    def __init__(self,
                 model: Module,
                 clients: Iterable[Client], 
                 elegibility_percentage: float=0.5, 
                 weighted: bool=False):
        self.device = GlobalSettings().get_device()
        self.model = model.to(self.device)
        self.clients = clients
        self.n_clients = len(clients)
        self.elegibility_percentage = elegibility_percentage
        self.weighted = weighted
        self.callbacks = []
    
    def fit(self, n_rounds: int=10) -> None:
        """Run the federated learning algorithm.

        Parameters
        ----------
        n_rounds : int, optional
            The number of rounds to run, by default 10.
        """
        with Progress() as progress:
            client_x_round = int(self.n_clients*self.elegibility_percentage)
            task_rounds = progress.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            task_local = progress.add_task("[green]Local Training", total=client_x_round)

            for round in range(n_rounds):
                eligible = self.get_eligible_clients()
                self.broadcast(eligible)
                client_evals = []
                for c, client in enumerate(eligible):
                    client_eval = client.local_train()
                    if client_eval:
                        client_evals.append(client_eval)
                    progress.update(task_id=task_local, completed=c+1)
                    progress.update(task_id=task_rounds, advance=1)
                self.aggregate(eligible)
                self.notify_all(self.model, round + 1, client_evals)

    def get_eligible_clients(self) -> Iterable[Client]:
        """Get the clients that will participate in the current round.

        The number of clients is determined by the `elegibility_percentage` attribute.

        Returns
        -------
        Iterable[Client]
            The clients that will participate in the current round.
        """
        if self.elegibility_percentage == 1:
            return self.clients
        n = int(self.n_clients * self.elegibility_percentage)
        return np.random.choice(self.clients, n)

    def broadcast(self, eligible: Iterable[Client]=None) -> None:
        """Broadcast the model to the clients.

        The broadcast is done by calling the `receive` method of each participating client.

        Parameters
        ----------
        eligible : Iterable[Client], optional
            The clients that will receive the model, by default None. 
            If None, all clients will receive the model.
        """
        eligible = eligible if eligible is not None else self.clients
        for client in eligible:
            client.receive(deepcopy(self.model))

    def init(self, path: str=None, **kwargs) -> None:
        """Initialize the server model from a checkpoint.

        Parameters
        ----------
        path : str, optional
            The path to the checkpoint, by default None. 
            If None, the model will be initialized randomly.
        """
        if path is not None:
            self.load(path)
    
    def aggregate(self, eligible: Iterable[Client]) -> None:
        """Aggregate the models of the clients.

        The aggregation is done by averaging the models of the clients. If the attribute `weighted` 
        is True, the clients are weighted by their number of samples.

        Parameters
        ----------
        eligible : Iterable[Client]
            The clients whose models will be aggregated.
        """
        avg_model_sd = OrderedDict()
        clients_sd = [eligible[i].send().state_dict() for i in range(len(eligible))]
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    avg_model_sd[key] = deepcopy(clients_sd[0][key])
                    continue
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
    
    def register_callback(self, callback: Callable) -> None:
        """Register a callback to be called after each round.

        Parameters
        ----------
        callback : Callable
            The callback to be registered.
        """
        if callback is not None and callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def notify_all(self, *args, **kwargs) -> None:
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            callback(*args, **kwargs)
    
    def save(self, path: str) -> None:
        """Save the model to a checkpoint.

        Parameters
        ----------
        path : str
            The path to the checkpoint.
        """
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load the model from a checkpoint.

        Parameters
        ----------
        path : str
            The path to the checkpoint.
        """
        self.model.load_state_dict(torch.load(path))