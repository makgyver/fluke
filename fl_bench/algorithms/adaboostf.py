from copy import deepcopy
from typing import Union, Any, List, Optional
from pyparsing import Iterable
from sklearn.base import ClassifierMixin

from math import log
import numpy as np
from numpy.random import choice

import sys; sys.path.append(".")

import torch

from fl_bench.data import DataSplitter
from fl_bench import GlobalSettings, ObserverSubject
from fl_bench.algorithms import CentralizedFL


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
    

class AdaBoostFClient():

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 base_classifier: ClassifierMixin):
        self.X = X
        self.y = y
        self.base_classifier = base_classifier
        self.d = np.ones(self.X.shape[0])
    
    def local_train(self) -> ClassifierMixin:
        clf = deepcopy(self.base_classifier)
        ids = choice(self.X.shape[0], size=self.X.shape[0], replace=True, p=self.d/self.d.sum())
        X_, y_ = self.X[ids], self.y[ids]
        clf.fit(X_, y_)
        return clf
    
    def compute_errors(self, clfs: List[ClassifierMixin]) -> List[float]:
        errors = []
        for clf in clfs:
            predictions = clf.predict(self.X)
            errors.append(sum(self.d[self.y != predictions]))
        return errors

    def update_dist(self, best_clf: ClassifierMixin, alpha: float) -> None:
        predictions = best_clf.predict(self.X)
        self.d *= np.exp(alpha * (self.y != predictions))

    def get_norm(self) -> float:
        return sum(self.d)


class AdaboostFServer(ObserverSubject):
    def __init__(self,
                 clients: Iterable[AdaBoostFClient], 
                 eligibility_percentage: float=0.5, 
                 n_classes: int = 2):
        super().__init__()
        self.model = StrongClassifier(n_classes)
        self.n_clients = len(clients)
        self.clients = clients
        self.eligibility_percentage = eligibility_percentage
        self.K = n_classes
        self.rounds = 0
    
    def init(self):
        pass

    def get_eligible_clients(self) -> Iterable[AdaBoostFClient]:
        if self.eligibility_percentage == 1:
            return self.clients
        n = int(self.n_clients * self.eligibility_percentage)
        return np.random.choice(self.clients, n)
    
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

                weak_learners = []
                for c, client in enumerate(eligible):
                    weak_learners.append(client.local_train())
                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)

                best_clf, alpha = self.aggregate(eligible, weak_learners)
                self.model.update(best_clf, alpha)
                
                for client in eligible:
                    client.update_dist(best_clf, alpha)

                self.notify_end_round(round + 1, self.model, 0)
                self.rounds += 1 

                # if self.checkpoint_path is not None:
                #     self.save(self.checkpoint_path)

            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)



    def aggregate(self, 
                  eligible: Iterable[AdaBoostFClient], 
                  weak_learners: Iterable[ClassifierMixin]) -> None:
        errors = np.array([client.compute_errors(weak_learners) for client in eligible])
        norm = sum([client.get_norm() for client in eligible])
        wl_errs = errors.sum(axis=0) / norm
        best_clf = weak_learners[wl_errs.argmin()]
        epsilon = wl_errs.min()
        alpha = log((1 - epsilon) / (epsilon + 1e-10)) + log(self.K - 1)
        return best_clf, alpha
            
    def notify_start_round(self, round: int, global_model: StrongClassifier) -> None:
        for observer in self._observers:
            observer.start_round(round, global_model)
    
    def notify_end_round(self, round: int, global_model: StrongClassifier, client_evals: Iterable[Any]) -> None:
        for observer in self._observers:
            observer.end_round(round, global_model, client_evals)
    
    def notify_selected_clients(self, round: int, clients: Iterable[Any]) -> None:
        for observer in self._observers:
            observer.selected_clients(round, clients)
    
    def notify_error(self, error: str) -> None:
        for observer in self._observers:
            observer.error(error)


class AdaboostF(CentralizedFL):
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
            # FIXME random state
            self.clients.append(AdaBoostFClient(X, y, self.base_classifier(max_leaf_nodes=10, random_state=1)))

    def init_server(self, n_classes: int = 2):
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
        pass
    
    def load_checkpoint(self, path: str):
        pass
