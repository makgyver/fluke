from __future__ import annotations

import numpy as np
from copy import deepcopy
import multiprocessing as mp
from typing import Iterable, Any
from collections import OrderedDict

import torch
from torch.nn import Module

from fl_bench.channel import Channel
from fl_bench.data import FastTensorDataLoader
from fl_bench import GlobalSettings, Message, ObserverSubject

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fl_bench.client import Client

class Server(ObserverSubject):
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
                 test_data: FastTensorDataLoader,
                 clients: Iterable[Client], 
                 weighted: bool=False):
        super().__init__()
        self.device = GlobalSettings().get_device()
        self.model = model
        self.clients = clients
        self.channel = Channel()
        self.n_clients = len(clients)
        self.weighted = weighted
        self.callbacks = []
        self.rounds = 0
        self.checkpoint_path = None
        self.test_data = test_data

        for client in self.clients:
            client.set_server(self)
    
    def _local_train(self, client: Client) -> None:
        """Train the client model locally.

        Parameters
        ----------
        client : Client
            The client to train.
        """
        self.channel.send(Message((client.local_train, {}), "__action__", self), client)
    
    def _broadcast_model(self, eligible: Iterable[Client]) -> None:
        self.channel.broadcast(Message(self.model, "model", self), eligible)

    def fit(self, n_rounds: int=10, eligible_perc: float=0.1) -> None:
        """Run the federated learning algorithm.
        Parameters
        ----------
        n_rounds : int, optional
            The number of rounds to run, by default 10.
        """
        if GlobalSettings().get_workers() > 1:
            return self._fit_multiprocess(n_rounds)

        with GlobalSettings().get_live_renderer():
            progress_fl = GlobalSettings().get_progress_bar("FL")
            progress_client = GlobalSettings().get_progress_bar("clients")
            client_x_round = int(self.n_clients*eligible_perc)
            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            task_local = progress_client.add_task("[green]Local Training", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            for round in range(self.rounds, total_rounds):
                self.notify_start_round(round + 1, self.model)
                eligible = self.get_eligible_clients(eligible_perc)
                self.notify_selected_clients(round + 1, eligible)
                self._broadcast_model(eligible)
                client_evals = []
                for c, client in enumerate(eligible):
                    self.channel.send(Message((client.local_train, {}), "__action__", self), client)
                    client_eval = client.validate()
                    if client_eval:
                        client_evals.append(client_eval)
                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)
                self.aggregate(eligible)
                self.notify_end_round(round + 1, self.model, client_evals)
                self.rounds += 1 
                if self.checkpoint_path is not None:
                    self.save(self.checkpoint_path)
            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)
    
    def _fit_multiprocess(self, n_rounds: int=10, eligible_perc: float=0.1) -> None:
        """Run the federated learning algorithm using multiprocessing.

        Parameters
        ----------
        n_rounds : int, optional
            The number of rounds to run, by default 10.
        """
        progress_fl = GlobalSettings().get_progress_bar("FL")
        progress_client = GlobalSettings().get_progress_bar("clients")
        def callback_progress(result):
            progress_fl.update(task_id=task_rounds, advance=1)
            progress_client.update(task_id=task_local, advance=1)

        client_x_round = int(self.n_clients*eligible_perc)
        task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
        task_local = progress_client.add_task("[green]Local Training", total=client_x_round)

        total_rounds = self.rounds + n_rounds
        
        with GlobalSettings().get_live_renderer():
            for round in range(self.rounds, total_rounds):
                self.notify_start_round(round + 1, self.model)
                client_evals = []
                eligible = self.get_eligible_clients(eligible_perc)
                self.notify_selected_clients(round + 1, eligible)
                self._broadcast_model(eligible)
                progress_client.update(task_id=task_local, completed=0)
                with mp.Pool(processes=GlobalSettings().get_workers()) as pool:
                    for client in eligible:
                        pool.apply_async(self._local_train,
                                         args=(client,), 
                                         callback=callback_progress)
                        client_eval = client.validate()
                        if client_eval:
                            client_evals.append(client_eval)
                    pool.close()
                    pool.join()
                client_evals = [c.get() for c in client_evals]
                self.aggregate(eligible)
                self.notify_end_round(round + 1, self.model, client_evals if client_evals[0] is not None else None)
                self.rounds += 1
                if self.checkpoint_path is not None:
                    self.save(self.checkpoint_path)
        progress_fl.remove_task(task_rounds)
        progress_client.remove_task(task_local)

    def get_eligible_clients(self, eligible_perc: float) -> Iterable[Client]:
        """Get the clients that will participate in the current round.

        The number of clients is determined by the `elegibility_percentage` attribute.

        Returns
        -------
        Iterable[Client]
            The clients that will participate in the current round.
        """
        if eligible_perc == 1:
            return self.clients
        n = int(self.n_clients * eligible_perc)
        return np.random.choice(self.clients, n)


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

    def _get_client_models(self, eligible: Iterable[Client]):
        return [self.channel.receive(self, client, "model").payload.state_dict() for client in eligible]

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
        clients_sd = self._get_client_models(eligible)
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
    
    def notify_start_round(self, round: int, global_model: Any) -> None:
        for observer in self._observers:
            observer.start_round(round, global_model)
    
    def notify_end_round(self, round: int, global_model: Any, client_evals: Iterable[Any]) -> None:
        for observer in self._observers:
            observer.end_round(round, global_model, client_evals)
    
    def notify_selected_clients(self, round: int, clients: Iterable[Any]) -> None:
        for observer in self._observers:
            observer.selected_clients(round, clients)
    
    def notify_error(self, error: str) -> None:
        for observer in self._observers:
            observer.error(error)
    
    def notify_send(self, msg: Message) -> None:
        for observer in self._observers:
            observer.send(msg)
    
    def notify_receive(self, msg: Message) -> None:
        for observer in self._observers:
            observer.receive(msg)
    
    def save(self, path: str) -> None:
        """Save the model to a checkpoint.

        Parameters
        ----------
        path : str
            The path to the checkpoint.
        """

        torch.save({
            'round': self.rounds,
            'state_dict': self.model.state_dict(),
            'client_optimizers': {
                i: client.checkpoint() for i, client in enumerate(self.clients)
            }
        }, path)
    
    def load(self, path: str) -> None:
        """Load the model from a checkpoint.

        Parameters
        ----------
        path : str
            The path to the checkpoint.
        """
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.rounds = checkpoint['round']
            for i, client in enumerate(self.clients):
                client.restore(checkpoint['client_optimizers'][i])
        except Exception as e:
            self.notify_error(f"Unable to load the checkpoint:\n\t{e}.\nCheckpoint will be ignored.")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(weighted={self.weighted})"

    def __repr__(self) -> str:
        return self.__str__()