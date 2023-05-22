from copy import deepcopy
from typing import Union, Any, Optional
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

class Samme():

    def __init__(self, 
                 n_clfs: int,
                 base_classifier: ClassifierMixin):
        self.n_clf = n_clfs
        self.base_classifier = base_classifier
        self.clfs = None
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            num_labels = None):

        self.K = len(set(y)) if not num_labels else num_labels # assuming that all classes are in y
        n_samples = X.shape[0]
        D = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        self.alpha = []
        for t in range(self.n_clf):
            clf = deepcopy(self.base_classifier)
            ids = choice(n_samples, size=n_samples, replace=True, p=D)
            X_, y_ = X[ids], y[ids]
            clf.fit(X_, y_)

            predictions = clf.predict(X)
            min_error = np.sum(D[y != predictions]) / np.sum(D)
            # kind of additive smoothing
            self.alpha.append(
                log((1.0 - min_error) / (min_error + 1e-10)) + log(self.K-1))
            D *= np.exp(self.alpha[t] * (y != predictions))
            D /= np.sum(D)
            self.clfs.append(clf)

    def predict(self,
                X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((np.shape(X)[0], self.K))
        for i, clf in enumerate(self.clfs):
            pred = clf.predict(X)
            for j, c in enumerate(pred):
                y_pred[j, int(c)] += self.alpha[i]
        return np.argmax(y_pred, axis=1)

# FIX ME
class StrongClassifier():
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
    

class PreweakF2Client(Client):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 n_classes: int,
                 n_estimators: int,
                 base_classifier: ClassifierMixin):
        self.X = X
        self.y = y
        self.K = n_classes
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
        self.d = np.ones(self.X.shape[0])
        self.server = None
    
    def local_train(self) -> None:
        samme = Samme(self.n_estimators, self.base_classifier)
        samme.fit(self.X, self.y, self.K)
        self.weak_classifiers = samme.clfs
        self.predictions = np.array([clf.predict(self.X) for clf in self.weak_classifiers])

    def compute_best_clf(self) -> None:
        d_norm = self.d / self.d.sum()
        errors = [sum(d_norm[self.y != self.predictions[i]]) for i, _ in enumerate(self.weak_classifiers)]
        self.channel.send(Message(self.weak_classifiers[np.argmin(errors)], "best_clf", sender=self), self.server)
    
    def compute_error(self) -> None:
        self.best_clf = self.channel.receive(self, self.server, msg_type="best_clf").payload
        predictions = self.best_clf.predict(self.X)
        error = sum(self.d[self.y != predictions])
        self.channel.send(Message(error, "error", sender=self), self.server)

    def update_dist(self) -> None:
        alpha = self.channel.receive(self, self.server, msg_type="alpha").payload
        predictions = self.best_clf.predict(self.X)
        self.d *= np.exp(alpha * (self.y != predictions))
    
    def send_norm(self) -> None:
        self.channel.send(Message(sum(self.d), "norm", sender=self), self.server)

    def validate(self):
        # TODO: implement validation
        raise NotImplementedError
    
    def checkpoint(self):
        raise NotImplementedError("PreweakF does not support checkpointing")

    def restore(self, checkpoint):
        raise NotImplementedError("PreweakF does not support checkpointing")


class PreweakF2Server(Server):
    def __init__(self,
                 clients: Iterable[PreweakF2Client], 
                 eligibility_percentage: float=0.5, 
                 n_classes: int = 2):
        super().__init__(StrongClassifier(n_classes), clients, eligibility_percentage, False)
        self.K = n_classes
    
    def init(self):
        pass
    
    def fit(self, n_rounds: int) -> None:

        with GlobalSettings().get_live_renderer():

            progress_fl = GlobalSettings().get_progress_bar("FL")
            progress_samme = progress_fl.add_task("Local Samme fit", total=self.n_clients)

            for client in self.clients:
                self.channel.send(Message((client.local_train, {}), "__action__", self), client)
                progress_fl.update(task_id=progress_samme, advance=1)
            
            progress_fl.remove_task(progress_samme)

            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds)
            total_rounds = self.rounds + n_rounds

            for round in range(self.rounds, total_rounds):
                self.notify_start_round(round + 1, self.model)
                eligible = self.get_eligible_clients()
                self.notify_selected_clients(round + 1, eligible)
                
                best_weak_clf, alpha = self.aggregate(eligible)
                self.model.update(best_weak_clf, alpha)
                
                self.channel.broadcast(Message(alpha, "alpha", self), eligible)
                self.channel.broadcast(Message(("update_dist", {}), "__action__", self), eligible)

                progress_fl.update(task_id=task_rounds, advance=1)
                self.notify_end_round(round + 1, self.model, 0)
                self.rounds += 1 

            progress_fl.remove_task(task_rounds)

    
    def aggregate(self, eligible: Iterable[PreweakF2Client]) -> None:
        selected_client = np.random.choice(eligible)
        self.channel.send(Message(("compute_best_clf", {}), "__action__", self), selected_client)
        best_weak_clf = self.channel.receive(self, selected_client, "best_clf").payload

        self.channel.broadcast(Message(best_weak_clf, "best_clf", self), eligible)
        self.channel.broadcast(Message(("compute_error", {}), "__action__", self), eligible)
        self.channel.broadcast(Message(("send_norm", {}), "__action__", self), eligible)

        error = np.array([self.channel.receive(self, sender=client, msg_type="error").payload for client in eligible])
        norm = sum([self.channel.receive(self, client, "norm").payload for client in eligible])
        epsilon = error.sum() / norm
        alpha = log((1 - epsilon) / (epsilon + 1e-10)) + log(self.K - 1)
        return best_weak_clf, alpha


class PreweakF2(CentralizedFL):
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
    
    def init_clients(self, data_splitter: DataSplitter, n_classes: int, **kwargs):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.data_assignment = data_splitter.assignments
        self.clients = []
        for i in range(self.n_clients):
            loader = data_splitter.client_train_loader[i]
            tensor_X, tensor_y = loader.tensors
            X, y = tensor_X.numpy(), tensor_y.numpy()
            self.clients.append(PreweakF2Client(X, y, n_classes, self.n_rounds, deepcopy(self.base_classifier)))

    def init_server(self, n_classes: int):
        self.server = PreweakF2Server(self.clients, self.eligibility_percentage, n_classes)
        

    def init_parties(self, 
                     data_splitter: DataSplitter, 
                     callbacks: Optional[Union[Any, Iterable[Any]]]=None):
        
        n_classes = len(torch.unique(data_splitter.y))
        self.init_clients(data_splitter, n_classes)
        self.init_server(n_classes)
        self.server.attach(callbacks)
        self.server.channel.attach(callbacks)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},clf={self.base_classifier}," + \
               f"P={self.eligibility_percentage})"

    def activate_checkpoint(self, path: str):
        raise NotImplementedError("PreweakF does not support checkpointing")
    
    def load_checkpoint(self, path: str):
        raise NotImplementedError("PreweakF does not support checkpointing")
