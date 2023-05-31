import sys; sys.path.append(".")

import numpy as np
from math import log
from copy import deepcopy
from typing import Tuple, Any
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
    

class AdaboostF2Client(Client):

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 n_classes: int,
                 base_classifier: ClassifierMixin,
                 validation_set = None):
        self.X = X
        self.y = y
        self.K = n_classes
        self.base_classifier = base_classifier
        self.d = np.ones(self.X.shape[0])
        self.server = None
        self.strong_clf = StrongClassifier(self.K)
        self.validation_set = validation_set
    
    def local_train(self) -> None:
        self.clf = deepcopy(self.base_classifier)
        ids = choice(self.X.shape[0], size=self.X.shape[0], replace=True, p=self.d/self.d.sum())
        X_, y_ = self.X[ids], self.y[ids]
        self.clf.fit(X_, y_)
    
    def send_weak_clf(self) -> None:
        self.channel.send(Message(self.clf, "weak_classifier", sender=self), self.server)
            
    def compute_error(self) -> None:
        self.best_clf = self.channel.receive(self, self.server, msg_type="best_clf").payload
        predictions = self.best_clf.predict(self.X)
        error = sum(self.d[self.y != predictions])
        self.channel.send(Message(error, "error", sender=self), self.server)

    def update_dist(self) -> None:
        alpha = self.channel.receive(self, self.server, msg_type="alpha").payload
        self.strong_clf.update(self.best_clf, alpha)
        predictions = self.best_clf.predict(self.X)
        self.d *= np.exp(alpha * (self.y != predictions))
    
    def send_norm(self) -> None:
        self.channel.send(Message(sum(self.d), "norm", sender=self), self.server)
    
    def validate(self):
        if self.validation_set is not None:
            return ClassificationSklearnEval(self.validation_set).evaluate(self.strong_clf)
    
    def checkpoint(self):
        raise NotImplementedError("AdaboostF2 does not support checkpointing")

    def restore(self, checkpoint):
        raise NotImplementedError("AdaboostF2 does not support checkpointing")


class AdaboostF2Server(Server):
    
    def __init__(self,
                 model: Any,
                 clients: Iterable[AdaboostF2Client], 
                 test_data: FastTensorDataLoader,
                 n_classes: int = 2):
        super().__init__(model, test_data, clients, False)
        self.K = n_classes

    def init(self):
        pass
    
    def fit(self, n_rounds: int, eligible_perc: float) -> None:

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

                for c, client in enumerate(eligible):
                    self.channel.send(Message((client.local_train, {}), "__action__", self), client)
                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)

                best_clf, alpha = self.aggregate(eligible)
                self.model.update(best_clf, alpha)
                
                self.channel.broadcast(Message(alpha, "alpha", self), eligible)
                self.channel.broadcast(Message(("update_dist", {}), "__action__", self), eligible)

                client_evals = [client.validate() for client in eligible]
                self.notify_end_round(round + 1, self.model, client_evals)
                self.rounds += 1 

            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

    
    def aggregate(self, 
                  eligible: Iterable[AdaboostF2Client]) -> Tuple[ClassifierMixin, float]:

        selected_client = np.random.choice(eligible)
        self.channel.send(Message(("send_weak_clf", {}), "__action__", self), selected_client)
        best_weak_clf = self.channel.receive(self, selected_client, msg_type="weak_classifier").payload

        self.channel.broadcast(Message(best_weak_clf, "best_clf", self), eligible)
        self.channel.broadcast(Message(("compute_error", {}), "__action__", self), eligible)
        self.channel.broadcast(Message(("send_norm", {}), "__action__", self), eligible)

        error = np.array([self.channel.receive(self, sender=client, msg_type="error").payload for client in eligible])
        norm = sum([self.channel.receive(self, client, "norm").payload for client in eligible])
        epsilon = error.sum() / norm
        alpha = log((1 - epsilon) / (epsilon + 1e-10)) + log(self.K - 1)
        return best_weak_clf, alpha


class AdaboostF2(CentralizedFL):

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
            self.clients.append(AdaboostF2Client(X, y, config.n_classes, deepcopy(base_model), clients_te_data[i]))

    def init_server(self, model: Any, data: FastTensorDataLoader, config: DDict):
        self.server = AdaboostF2Server(model, self.clients, data, **config)

    def activate_checkpoint(self, path: str):
        raise NotImplementedError("AdaboostF2 does not support checkpointing")
    
    def load_checkpoint(self, path: str):
        raise NotImplementedError("AdaboostF2 does not support checkpointing")
