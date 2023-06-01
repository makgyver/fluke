import sys; sys.path.append(".")

import numpy as np
from math import log
from typing import Any
from copy import deepcopy
from pyparsing import Iterable
from numpy.random import choice
from sklearn.base import ClassifierMixin

from fl_bench.client import Client
from fl_bench.server import Server
from fl_bench import GlobalSettings, Message
from fl_bench.algorithms import CentralizedFL
from fl_bench.utils import DDict, import_module_from_str
from fl_bench.evaluation import ClassificationSklearnEval
from fl_bench.data import DataSplitter, FastTensorDataLoader

class Samme():

    def __init__(self, 
                 n_clfs: int,
                 base_classifier: ClassifierMixin):
        self.n_clf = n_clfs
        self.base_classifier = base_classifier
        self.clfs = None
        self.server = None
    
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
    

class PreweakFClient(Client):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 n_classes: int,
                 n_estimators: int,
                 base_classifier: ClassifierMixin,
                 validation_set = None):
        self.X = X
        self.y = y
        self.K = n_classes
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
        self.d = np.ones(self.X.shape[0])
        self.strong_clf = StrongClassifier(self.K)
        self.server = None
        self.validation_set = validation_set
    
    def local_train(self) -> None:
        samme = Samme(self.n_estimators, self.base_classifier)
        samme.fit(self.X, self.y, self.K)
        self.channel.send(Message(samme.clfs, "weak_classifiers", self), self.server)
    
    def compute_predictions(self) -> None:
        self.weak_classifiers = self.channel.receive(self, self.server, "all_weak_classifiers").payload
        self.predictions = np.array([clf.predict(self.X) for clf in self.weak_classifiers])

    def compute_errors(self) -> None:
        errors = []
        for preds in self.predictions:
            errors.append(sum(self.d[self.y != preds]))
        self.channel.send(Message(errors, "errors", sender=self), self.server)

    def update_dist(self) -> None:
        best_clf_id = self.channel.receive(self, self.server, msg_type="best_clf_id").payload
        alpha = self.channel.receive(self, self.server, msg_type="alpha").payload
        self.strong_clf.update(self.weak_classifiers[best_clf_id], alpha)
        predictions = self.predictions[best_clf_id]
        self.d *= np.exp(alpha * (self.y != predictions))
    
    def send_norm(self) -> None:
        self.channel.send(Message(sum(self.d), "norm", sender=self), self.server)
    
    def validate(self):
        if self.validation_set is not None:
            return ClassificationSklearnEval(self.validation_set).evaluate(self.strong_clf)
    
    def checkpoint(self):
        raise NotImplementedError("PreweakF does not support checkpointing")

    def restore(self, checkpoint):
        raise NotImplementedError("PreweakF does not support checkpointing")


class PreweakFServer(Server):
    def __init__(self,
                 model: Any,
                 clients: Iterable[PreweakFClient], 
                 test_data: FastTensorDataLoader,
                 n_classes: int = 2):
        super().__init__(model, test_data, clients, False)
        self.K = n_classes
    
    def init(self):
        pass
    
    def fit(self, n_rounds: int, eligible_perc: float) -> None:

        with GlobalSettings().get_live_renderer():
            progress_fl = GlobalSettings().get_progress_bar("FL")
            client_x_round = int(self.n_clients*eligible_perc)

            progress_samme = progress_fl.add_task("Local Samme fit", total=self.n_clients)
            weak_classifiers = []
            for client in self.clients:
                self.channel.send(Message((client.local_train, {}), "__action__", self), client)
                weak_classifiers.extend(self.channel.receive(self, client, "weak_classifiers").payload)
                progress_fl.update(task_id=progress_samme, advance=1)
            progress_fl.remove_task(progress_samme)

            self.channel.broadcast(Message(weak_classifiers, "all_weak_classifiers", self), self.clients)

            progress_preds = progress_fl.add_task("Local predictions", total=self.n_clients)
            for client in self.clients:
                self.channel.send(Message((client.compute_predictions, {}), "__action__", self), client)
                progress_fl.update(task_id=progress_preds, advance=1)
            progress_fl.remove_task(progress_preds)

            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            total_rounds = self.rounds + n_rounds

            for round in range(self.rounds, total_rounds):
                self.notify_start_round(round + 1, self.model)
                eligible = self.get_eligible_clients(eligible_perc)
                self.notify_selected_clients(round + 1, eligible)

                best_clf_id, alpha = self.aggregate(eligible)
                self.model.update(weak_classifiers[best_clf_id], alpha)
                
                self.channel.broadcast(Message(best_clf_id, "best_clf_id", self), eligible)
                self.channel.broadcast(Message(alpha, "alpha", self), eligible)
                self.channel.broadcast(Message(("update_dist", {}), "__action__", self), eligible)

                progress_fl.update(task_id=task_rounds, advance=1)
                client_evals = [client.validate() for client in eligible]
                self.notify_end_round(round + 1, self.model, client_evals)
                self.rounds += 1 

            progress_fl.remove_task(task_rounds)

    
    def aggregate(self, eligible: Iterable[PreweakFClient]) -> None:
        self.channel.broadcast(Message(("compute_errors", {}), "__action__", self), eligible)
        self.channel.broadcast(Message(("send_norm", {}), "__action__", self), eligible)
        errors = np.array([self.channel.receive(self, sender=client, msg_type="errors").payload for client in eligible])
        norm = sum([self.channel.receive(self, sender=client, msg_type="norm").payload for client in eligible])
        wl_errs = errors.sum(axis=0) / norm
        best_clf_id = wl_errs.argmin()
        epsilon = wl_errs.min()
        alpha = log((1 - epsilon) / (epsilon + 1e-10)) + log(self.K - 1)
        return best_clf_id, alpha


class PreweakF(CentralizedFL):
    def __init__(self, 
                 n_clients: int,
                 data_splitter: DataSplitter, 
                 hyperparameters: DDict):
        
        self.n_clients = n_clients
        (clients_tr_data, clients_te_data), server_data = data_splitter.assign(n_clients, 
                                                                               hyperparameters.client.batch_size)
        hyperparameters.client.n_classes = data_splitter.num_classes()
        hyperparameters.server.n_classes = data_splitter.num_classes()
        self.init_clients(clients_tr_data, clients_te_data, hyperparameters.client)
        self.init_server(StrongClassifier(hyperparameters.server.n_classes), 
                         server_data, 
                         hyperparameters.server)
        
    def init_clients(self, 
                     clients_tr_data: list[FastTensorDataLoader], 
                     clients_te_data: list[FastTensorDataLoader], 
                     config: DDict):
        self.clients = []
        config.clf_args.random_state = GlobalSettings().get_seed()
        base_model = import_module_from_str(config.base_classifier)(**config.clf_args)
        for i in range(self.n_clients):
            loader = clients_tr_data[i]
            tensor_X, tensor_y = loader.tensors
            X, y = tensor_X.numpy(), tensor_y.numpy()
            self.clients.append(PreweakFClient(X, 
                                               y, 
                                               config.n_classes, 
                                               config.n_clfs, 
                                               deepcopy(base_model), 
                                               clients_te_data[i]))
    
    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = PreweakFServer(model, self.clients, data, **config)
        
    def activate_checkpoint(self, path: str):
        raise NotImplementedError("PreweakF does not support checkpointing")
    
    def load_checkpoint(self, path: str):
        raise NotImplementedError("PreweakF does not support checkpointing")
