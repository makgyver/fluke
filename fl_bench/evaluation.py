import sys; sys.path.append(".")
from abc import ABC, abstractmethod

import torch
from sklearn.base import ClassifierMixin
from typing import Callable, Literal, Union, Iterable
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
    def __init__(self, loss_fn: Callable):
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
    """Evaluate a classification pytorch model.

    The metrics computed are accuracy, precision, recall, f1 and the loss according to the provided
    loss function `loss_fn`.

    Parameters
    ----------
    loss_fn : Callable
        The loss function to consider.
    n_classes : int, optional
        The number of classes.
    average : Literal["micro","macro"], optional
        The average to use for the metrics, by default "micro".
    device : torch.device, optional
        The device to use for evaluation, by default torch.device("cpu").
    
    Returns
    -------
    dict[str, float]
        The dictionary containing the computed metrics.
    """
    def __init__(self, 
                 loss_fn: Callable, 
                 n_classes: int, 
                 average: Literal["micro","macro"]="micro",
                 device: torch.device=torch.device("cpu")):
        super().__init__(loss_fn)
        self.average = average
        self.n_classes = n_classes
        self.device = device

    def evaluate(self, 
                 model: torch.nn.Module, 
                 eval_data_loader: Union[FastTensorDataLoader, Iterable[FastTensorDataLoader]]) -> dict:
        """Evaluate the model.

        Parameters
        ----------
        eval_data_loader : FastTensorDataLoader or Iterable[FastTensorDataLoader]
            The data loader(s) to use for evaluation.
        
        Returns
        -------
        dict[str, float]
            The dictionary containing the computed metrics.
        """
        model.eval()
        task = "multiclass" #if self.n_classes >= 2 else "binary"
        accs, precs, recs, f1s = [], [], [], []
        loss, cnt = 0, 0
        if eval_data_loader is None:
            return {}
        
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

                accuracy.update(y_hat.cpu(), y.cpu())
                precision.update(y_hat.cpu(), y.cpu())
                recall.update(y_hat.cpu(), y.cpu())
                f1.update(y_hat.cpu(), y.cpu())

            cnt += len(data_loader)
            if cnt == 0:
                return {}
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
    """Evaluate a classification sklearn-compliant model.

    The metrics computed are accuracy, precision, recall, and f1.

    Parameters
    ----------
    average : Literal["micro","macro"], optional
        The average to use for the metrics, by default "macro".
    
    Returns
    -------
    dict[str, float]
        The dictionary containing the computed metrics.
    """

    def __init__(self, 
                 average: Literal["micro","macro"]="macro"):
        super().__init__(None)
        self.average = average

    def evaluate(self, 
                 model: ClassifierMixin, 
                 eval_data_loader: Union[FastTensorDataLoader, list[FastTensorDataLoader]]) -> dict:
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
