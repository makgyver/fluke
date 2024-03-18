import sys
sys.path.append(".")
sys.path.append("..")
from copy import deepcopy
import random
from typing import Callable, Iterable

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from rich.progress import Progress

from .. import GlobalSettings, Message
from ..evaluation import ClassificationEval
from ..client import Client
from ..server import Server
from ..data import DataSplitter, FastTensorDataLoader
from ..utils import DDict, OptimizerConfigurator
from ..algorithms import CentralizedFL
# from .net import EDModule

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    
def relative_projection(encoder: nn.Module, x: torch.Tensor, anchors: torch.Tensor, normalize_out: bool=True) -> torch.Tensor:
    # return x
    enc_x = encoder(x)
    enc_a = encoder(anchors)
    x = F.normalize(enc_x, p=2, dim=-1)
    anchors = F.normalize(enc_a, p=2, dim=-1)
    rel_proj =  torch.einsum("bm, am -> ba", x, anchors)
    return rel_proj if not normalize_out else F.normalize(rel_proj, p=2, dim=-1)

def generate_anchors(num_anchors: int, dim: int, seed: int=98765) -> torch.Tensor:
    _set_seed(seed)
    return torch.randn((num_anchors, dim))


class FLHalfClient(Client):
    def __init__(self,
                 index: int,
                 model: nn.Module,
                 train_set: FastTensorDataLoader,
                 validation_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 local_epochs: int,
                 tau: int):
        super().__init__(index, model, train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)
        self.personalized_model.init()
        self.hyper_params.update({
            "tau": tau
        })
        self.anchors = None

    def _private_train(self):
        if self.anchors is None:
            self.anchors = self.channel.receive(self, self.server, msg_type="anchors").payload

        self.personalized_model.train()
        self.private_optimizer, self.private_scheduler = self.optimizer_cfg(self.personalized_model)
        for _ in range(self.hyper_params.tau):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.private_optimizer.zero_grad()
                # y_hat = self.private_model(X)
                rel_x = relative_projection(self.personalized_model.E, X.view(X.size(0), -1), self.anchors)
                y_hat = self.personalized_model.D(rel_x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.private_optimizer.step()
            self.private_scheduler.step()
        self.model = deepcopy(self.personalized_model)
    
    def _receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.D.load_state_dict(msg.payload.state_dict())

    def local_train(self, override_local_epochs: int=0) -> dict:
        # if self.anchors is None:
        #     self.anchors = self.channel.receive(self, self.server, msg_type="anchors").payload

        epochs = override_local_epochs if override_local_epochs else self.local_epochs
        self._receive_model()
        self.model.train()
        # print(self.anchors.shape)
        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                # X = relative_projection(self.private_model.E, X.view(X.size(0), -1), self.anchors)
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                # y_hat = self.model(X)
                # print(self.model.E, X.shape, self.anchors.shape)
                rel_x = relative_projection(self.model.E, X.view(X.size(0), -1), self.anchors)
                y_hat = self.model.D(rel_x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        self.channel.send(Message(deepcopy(self.model.D), "model", self), self.server)

    def validate(self):
        if self.test_set is not None:
            n_classes = self.model.output_size
            test_loader = self.test_set.transform(lambda x: relative_projection(self.personalized_model.E, x.view(x.size(0), -1), self.anchors))
            return ClassificationEval(self.loss_fn, n_classes).evaluate(self.model.D, test_loader)



class FLHalfServer(Server):
    def __init__(self,
                 model: Module,
                 test_data: FastTensorDataLoader,
                 clients: Iterable[Client],
                 n_anchors: int=100,
                 seed_anchors: int=98765,
                 weighted: bool=True):
        super().__init__(model, test_data, clients, weighted)
        self.n_anchors = n_anchors
        self.seed_anchors = seed_anchors

    def fit(self, n_rounds: int=10, eligible_perc: float=0.1) -> None:
        """Run the federated learning algorithm.
        Parameters
        ----------
        n_rounds : int, optional
            The number of rounds to run, by default 10.
        """
        # if GlobalSettings().get_workers() > 1:
        #     return self._fit_multiprocess(n_rounds)

        anchors = generate_anchors(self.n_anchors, 784, seed=self.seed_anchors) #FIX ME
        for client in self.clients:
            self.channel.send(Message(anchors, "anchors", self), client)

        # Preparation step
        # the following code run private_train across all clients with progress bar
        with Progress() as progress:
            task = progress.add_task("[cyan]Client's Private Training", total=len(self.clients))
            for client in self.clients:
                self.channel.send(Message((client._private_train, {}), "__action__", self), client)
                progress.update(task, advance=1)
    
        # anchors = generate_anchors(self.n_anchors, 784, seed=self.seed_anchors) #FIX ME
        # for client in self.clients:
        #     self.channel.send(Message(anchors, "anchors", self), client)

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
                self.notify_end_round(round + 1, self.model, None, client_evals)
                self.rounds += 1 
            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)
        
        self.finalize()


#FIXME
class FLHalf(CentralizedFL):
    def __init__(self, 
                 n_clients: int,
                 data_splitter: DataSplitter, 
                 hyperparameters: DDict):
        
        hyperparameters["net_args"] = {
            "input_size": hyperparameters.server.n_anchors,
            "output_size": data_splitter.num_classes()
        }
        super().__init__(n_clients, data_splitter, hyperparameters)
    
    def get_optimizer_class(self) -> torch.optim.Optimizer:
        return torch.optim.Adam
    
    def get_client_class(self) -> Client:
        return FLHalfClient

    def get_server_class(self) -> Server:
        return FLHalfServer