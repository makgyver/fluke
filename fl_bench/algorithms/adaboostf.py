from copy import deepcopy
from typing import Union, Any, List, Optional
from pyparsing import Iterable
from sklearn.base import BaseEstimator

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


class StrongClassifier():
    def __init__(self, num_classes: int):
        self.alpha = []
        self.clfs = []
        self.K = num_classes
    
    def update(self, clf: BaseEstimator, alpha: float):
        self.alpha.append(alpha)
        self.clfs.append(clf)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros((np.shape(X)[0], self.K))
        for i, clf in enumerate(self.clfs):
            pred = clf.predict(X)
            for j, c in enumerate(pred):
                y_pred[j, int(c)] += self.alpha[i]
        return np.argmax(y_pred, axis=1)
    

class AdaboostClient(Client):

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 base_classifier: BaseEstimator):
        self.X = X
        self.y = y
        self.base_classifier = base_classifier
        self.d = np.ones(self.X.shape[0])
        self.cache = {}
    
    def local_train(self) -> BaseEstimator:
        clf = deepcopy(self.base_classifier)
        ids = choice(self.X.shape[0], size=self.X.shape[0], replace=True, p=self.d/self.d.sum())
        X_, y_ = self.X[ids], self.y[ids]
        clf.fit(X_, y_)
        self.cache["weak_classifier"] = clf
    
    def send(self, msg_type: str):
        if msg_type == "norm":
            return Message(sum(self.d), "norm")
        elif msg_type == "errors":
            errors = self.compute_errors()
            return Message(errors, "errors")
        elif msg_type == "weak_classifier":
            return Message(self.cache["weak_classifier"], "weak_classifier")

    def receive(self, msg: Message):
        self.cache[msg.msg_type] = msg.payload
            
    def compute_errors(self) -> List[float]:
        errors = []
        for clf in self.cache["weak_learners"]:
            predictions = clf.predict(self.X)
            errors.append(sum(self.d[self.y != predictions]))
        return errors

    def update_dist(self) -> None:
        best_clf, alpha = self.cache["best_clf"], self.cache["alpha"]
        predictions = best_clf.predict(self.X)
        self.d *= np.exp(alpha * (self.y != predictions))
    
    def validate(self):
        # TODO: implement validation
        raise NotImplementedError
    
    def checkpoint(self):
        raise NotImplementedError("AdaboostF does not support checkpointing")

    def restore(self, checkpoint):
        raise NotImplementedError("AdaboostF does not support checkpointing")


class AdaboostFServer(Server):
    def __init__(self,
                 clients: Iterable[AdaboostClient], 
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
                    client.local_train()
                    weak_classifiers.append(self.receive(client, "weak_classifier").payload)
                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)

                best_clf, alpha = self.aggregate(eligible, weak_classifiers)
                self.model.update(best_clf, alpha)
                
                self.broadcast(Message(best_clf, "best_clf"), eligible)
                self.broadcast(Message(alpha, "alpha"), eligible)
                for client in eligible:
                    client.update_dist()

                self.notify_end_round(round + 1, self.model, 0)
                self.rounds += 1 

                # if self.checkpoint_path is not None:
                #     self.save(self.checkpoint_path)

            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

    
    def aggregate(self, 
                  eligible: Iterable[AdaboostClient], 
                  weak_learners: Iterable[BaseEstimator]) -> None:

        self.broadcast(Message(weak_learners, "weak_learners"), eligible)
        errors = np.array([self.receive(client, "errors").payload for client in eligible])
        norm = sum([self.receive(client, "norm").payload for client in eligible])
        wl_errs = errors.sum(axis=0) / norm
        best_clf = weak_learners[wl_errs.argmin()]
        epsilon = wl_errs.min()
        alpha = log((1 - epsilon) / (epsilon + 1e-10)) + log(self.K - 1)
        return best_clf, alpha


class AdaboostF(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int, 
                 base_classifier: BaseEstimator,
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
            self.clients.append(AdaboostClient(X, y, deepcopy(self.base_classifier)))

    def init_server(self, n_classes: int):
        self.server = AdaboostFServer(self.clients, self.eligibility_percentage, n_classes)
        

    def init_parties(self, 
                     data_splitter: DataSplitter, 
                     callbacks: Optional[Union[Any, Iterable[Any]]]=None):
        
        self.init_clients(data_splitter)
        self.init_server(len(torch.unique(data_splitter.y)))
        self.server.attach(callbacks)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},clf={self.base_classifier}," + \
               f"P={self.eligibility_percentage})"

    def activate_checkpoint(self, path: str):
        raise NotImplementedError("AdaboostF does not support checkpointing")
    
    def load_checkpoint(self, path: str):
        raise NotImplementedError("AdaboostF does not support checkpointing")
