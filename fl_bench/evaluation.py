import sys; sys.path.append(".")
from abc import ABC, abstractmethod

import torch
from sklearn.base import ClassifierMixin
from typing import Callable, Literal, Union
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from fl_bench.data import FastTensorDataLoader

class Evaluator(ABC):
    """Base class for all evaluators.

    Parameters
    ----------
    data_loader : FastTensorDataLoader
        The data loader to use for evaluation.
    loss_fn : Callable
        The loss function to consider.
    """
    def __init__(self, 
                # data_loader: Union[FastTensorDataLoader, list[FastTensorDataLoader]], 
                loss_fn: Callable):
        # self.data_loader = data_loader
        self.loss_fn = loss_fn
    
    @abstractmethod
    def evaluate(self, model, eval_data_loader):
        """Evaluate the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        """
        pass

    def __call__(self, model, eval_data_loader):
        return self.evaluate(model, eval_data_loader)


class ClassificationEval(Evaluator):
    """Evaluate a classification model.

    Parameters
    ----------
    data_loader : FastTensorDataLoader
        The data loader to use for evaluation.
    loss_fn : Callable
        The loss function to consider.
    n_classes : int, optional
        The number of classes.
    average : Literal["micro","macro"], optional
        The average to use for the metrics, by default "macro".
    """
    def __init__(self, 
                #  data_loader: Union[FastTensorDataLoader, list[FastTensorDataLoader]], 
                 loss_fn: Callable, 
                 n_classes: int, 
                 average: Literal["micro","macro"]="macro",
                 device: torch.device=torch.device("cpu")):
        super().__init__(loss_fn)
        self.average = average
        self.n_classes = n_classes
        self.device = device

    def evaluate(self, model: torch.nn.Module, eval_data_loader: Union[FastTensorDataLoader, list[FastTensorDataLoader]]) -> dict:
        model.eval()
        # TODO: check if this is correct
        task = "multiclass" #if self.n_classes >= 2 else "binary"
        accs, precs, recs, f1s = [], [], [], []
        loss, cnt = 0, 0
        if not isinstance(eval_data_loader, list):
            eval_data_loader = [eval_data_loader]

        for data_loader in eval_data_loader:
            accuracy = Accuracy(task=task, num_classes=self.n_classes, top_k=1, average=self.average)
            precision = Precision(task=task, num_classes=self.n_classes, top_k=1, average=self.average)
            recall = Recall(task=task, num_classes=self.n_classes, top_k=1, average=self.average)
            f1 = F1Score(task=task, num_classes=self.n_classes, top_k=1, average=self.average)

            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                with torch.no_grad():
                    y_hat = model(X)
                    if self.loss_fn is not None:
                        loss += self.loss_fn(y_hat, y).item()

                accuracy.update(y_hat, y)
                precision.update(y_hat, y)
                recall.update(y_hat, y)
                f1.update(y_hat, y)

            cnt += len(data_loader)
            accs.append(accuracy.compute().item())
            precs.append(precision.compute().item())
            recs.append(recall.compute().item())
            f1s.append(f1.compute().item())

        return {
            "accuracy":  round(sum(accs) / len(accs), 5),
            "precision": round(sum(precs) / len(precs), 5),
            "recall":    round(sum(recs) / len(recs), 5),
            "f1":        round(sum(f1s) / len(f1s), 5),
            "loss":      round(loss / cnt, 5) if self.loss_fn is not None else None
        }
    

class ClassificationSklearnEval(Evaluator):

    def __init__(self, 
                #  data_loader: Union[FastTensorDataLoader, list[FastTensorDataLoader]], 
                 average: Literal["micro","macro"]="macro"):
        super().__init__(None)
        self.average = average

    def evaluate(self, model: ClassifierMixin, eval_data_loader: Union[FastTensorDataLoader, list[FastTensorDataLoader]]) -> dict:
        accs, precs, recs, f1s = [], [], [], []
        if not isinstance(eval_data_loader, list):
            eval_data_loader = [eval_data_loader]

        for data_loader in eval_data_loader:

            X, y = data_loader.tensors
            y_hat = model.predict(X)

            accs.append(accuracy_score(y, y_hat))
            precs.append(precision_score(y, y_hat, average=self.average)) #TODO zero_division
            recs.append(recall_score(y, y_hat, average=self.average))
            f1s.append(f1_score(y, y_hat, average=self.average))

        return {
            "accuracy":  round(sum(accs) / len(accs), 5),
            "precision": round(sum(precs) / len(precs), 5),
            "recall":    round(sum(recs) / len(recs), 5),
            "f1":        round(sum(f1s) / len(f1s), 5),
        }
