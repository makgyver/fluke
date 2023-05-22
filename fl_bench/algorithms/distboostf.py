from copy import deepcopy
from typing import Tuple, Union, Any, List, Optional
from pyparsing import Iterable
from sklearn.base import ClassifierMixin 

import torch
from math import log
import numpy as np
from numpy.random import choice

import sys; sys.path.append(".")
from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench.data import DataSplitter
from fl_bench import GlobalSettings, Message
from fl_bench.algorithms import CentralizedFL


class StrongClassifier:
    def __init__(self, num_classes: int):
        self.alpha = []
        self.clfs = []
        self.K = num_classes
    
    def update(self, clf: ClassifierMixin, alpha: float):
        self.alpha.append(alpha)
        self.clfs.append(clf)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((np.shape(X)[0], self.K))
        for i, clf in enumerate(self.clfs):
            pred = clf.predict(X)
            for j, c in enumerate(pred):
                y_pred[j, int(c)] += self.alpha[i]
        return np.argmax(y_pred, axis=1)
    

    
class DistboostHyp(ClassifierMixin):
    def __init__(self, ht: List[ClassifierMixin], K: int):
        self.ht = ht
        self.K = K

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((X.shape[0], self.K))
        for h in self.ht:
            pred = h.predict(X)
            for j, c in enumerate(pred):
                y_pred[j, int(c)] += 1
        return np.argmax(y_pred, axis=1)
    

class DistboostClient(Client):

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 base_classifier: ClassifierMixin):
        self.X = X
        self.y = y
        self.base_classifier = base_classifier
        self.d = np.ones(self.X.shape[0])
        self.cache = {}
    
    def local_train(self) -> ClassifierMixin:
        clf = deepcopy(self.base_classifier)
        ids = choice(self.X.shape[0], size=self.X.shape[0], replace=True, p=self.d/self.d.sum())
        X_, y_ = self.X[ids], self.y[ids]
        clf.fit(X_, y_)
        self.channel.send(Message(clf, "weak_classifier", sender=self), self.server)
        # self.cache["weak_classifier"] = clf
    
    # def send(self, msg_type: str) -> Message:
    #     if msg_type == "norm":
    #         return Message(sum(self.d), "norm")
    #     elif msg_type == "error":
    #         errors = self.compute_error()
    #         return Message(errors, "error")
    #     elif msg_type == "weak_classifier":
    #         return Message(self.cache["weak_classifier"], "weak_classifier")

    # def receive(self, msg: Message):
    #     self.cache[msg.msg_type] = msg.payload
            
    def compute_error(self) -> List[float]:
        self.round_hyp = self.channel.receive(self, msg_type="round_hyp").payload
        self.predictions = self.round_hyp.predict(self.X)
        errors = sum(self.d[self.y != self.predictions])
        self.channel.send(Message(errors, "error", sender=self), self.server)

    def update_dist(self) -> None:
        alpha = self.channel.receive(self, msg_type="alpha").payload
        self.d *= np.exp(alpha * (self.y != self.predictions))
    
    def send_norm(self) -> None:
        self.channel.send(Message(sum(self.d), "norm", sender=self), self.server)
    
    def validate(self):
        # TODO: implement validation
        raise NotImplementedError
    
    def checkpoint(self):
        raise NotImplementedError("DistboostF does not support checkpointing")

    def restore(self, checkpoint):
        raise NotImplementedError("DistboostF does not support checkpointing")


class DistboostFServer(Server):
    def __init__(self,
                 clients: Iterable[DistboostClient], 
                 eligibility_percentage: float=0.5, 
                 n_classes: int = 2):
        super().__init__(StrongClassifier(n_classes), clients, eligibility_percentage, False)
        self.K = n_classes
    
    def init(self):
        pass
    
    def fit(self, n_rounds: int) -> None:

        with GlobalSettings().get_live_renderer():

            progress_fl = GlobalSettings().get_progress_bar("FL")
            progress_client = GlobalSettings().get_progress_bar("clients")
            client_x_round = int(self.n_clients*self.eligibility_percentage)
            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            task_local = progress_client.add_task("[green]Local Training", total=client_x_round)
            
            total_rounds = self.rounds + n_rounds
            
            for round in range(self.rounds, total_rounds):
                self.notify_start_round(round + 1, self.model)
                eligible = self.get_eligible_clients()
                self.notify_selected_clients(round + 1, eligible)

                weak_classifiers = []
                for c, client in enumerate(eligible):
                    self.channel.send(Message((client.local_train, {}), "__action__"), client)
                    weak_classifiers.append(self.channel.receive(self, msg_type="weak_classifier").payload)
                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)

                best_clf, alpha = self.aggregate(eligible, weak_classifiers)
                self.model.update(best_clf, alpha)
                
                self.channel.broadcast(Message(alpha, "alpha"), eligible)
                self.channel.broadcast(Message(("update_dist", {}), "__action__"), eligible)

                self.notify_end_round(round + 1, self.model, 0)
                self.rounds += 1 

                # if self.checkpoint_path is not None:
                #     self.save(self.checkpoint_path)

            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

    
    def aggregate(self, 
                  eligible: Iterable[DistboostClient], 
                  weak_learners: Iterable[ClassifierMixin]) -> Tuple[ClassifierMixin, float]:

        aggr_hp = DistboostHyp(weak_learners, self.K)
        self.channel.broadcast(Message(aggr_hp, "round_hyp", self), eligible)
        self.channel.broadcast(Message(("compute_error", {}), "__action__"), eligible)
        self.channel.broadcast(Message(("send_norm", {}), "__action__"), eligible)
        errors = np.array([self.channel.receive(self, client, "error").payload for client in eligible])
        norm = sum([self.channel.receive(self, client, "norm").payload for client in eligible])
        wl_errs = errors.sum() / norm
        epsilon = wl_errs.min()
        alpha = log((1 - epsilon) / (epsilon + 1e-10)) + log(self.K - 1)
        return aggr_hp, alpha


class DistboostF(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 base_classifier: ClassifierMixin,
                 eligibility_percentage: float=0.5):
        
        super().__init__(n_clients,
                         n_rounds,
                         0,
                         None, 
                         None, 
                         None,
                         eligibility_percentage)
        self.base_classifier = base_classifier
    
    def init_clients(self, data_splitter: DataSplitter, **kwargs):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.data_assignment = data_splitter.assignments
        self.clients = []
        for i in range(self.n_clients):
            loader = data_splitter.client_train_loader[i]
            tensor_X, tensor_y = loader.tensors
            X, y = tensor_X.numpy(), tensor_y.numpy()
            self.clients.append(DistboostClient(X, y, deepcopy(self.base_classifier)))

    def init_server(self, n_classes: int):
        self.server = DistboostFServer(self.clients, self.eligibility_percentage, n_classes)
        

    def init_parties(self, 
                     data_splitter: DataSplitter, 
                     callbacks: Optional[Union[Any, Iterable[Any]]]=None):
        
        self.init_clients(data_splitter)
        self.init_server(len(torch.unique(data_splitter.y)))
        self.server.attach(callbacks)
        self.server.channel.attach(callbacks)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},clf={self.base_classifier}," + \
               f"P={self.eligibility_percentage})"

    def activate_checkpoint(self, path: str):
        raise NotImplementedError("DistboostF does not support checkpointing")
    
    def load_checkpoint(self, path: str):
        raise NotImplementedError("DistboostF does not support checkpointing")
