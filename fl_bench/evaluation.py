from abc import ABC, abstractmethod
import torch
from typing import Callable, Literal
from torchmetrics import Accuracy, Precision, Recall, F1Score

import sys; sys.path.append(".")
from fl_bench import GlobalSettings
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
    def __init__(self, data_loader: FastTensorDataLoader, loss_fn: Callable):
        self.data_loader = data_loader
        self.loss_fn = loss_fn
    
    @abstractmethod
    def evaluate(self, model):
        """Evaluate the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        """
        pass

    def __call__(self, model):
        return self.evaluate(model)


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
                 data_loader: FastTensorDataLoader, 
                 loss_fn: Callable, 
                 n_classes: int, 
                 average: Literal["micro","macro"]="macro"):
        super().__init__(data_loader, loss_fn)
        self.average = average
        self.n_classes = n_classes

    def evaluate(self, model: torch.nn.Module) -> dict:
        model.eval()
        task = "multiclass" if self.n_classes > 2 else "binary"
        accuracy = Accuracy()#task=task, num_classes=self.n_classes, top_k=1, average=self.average)
        precision = Precision(task=task, num_classes=self.n_classes, top_k=1, average=self.average)
        recall = Recall(task=task, num_classes=self.n_classes, top_k=1, average=self.average)
        f1 = F1Score(task=task, num_classes=self.n_classes, top_k=1, average=self.average)
        loss = 0
        device = GlobalSettings().get_device()
        for X, y in self.data_loader:
            X, y = X.to(device), y.to(device)
            
            with torch.no_grad():
                y_hat = model(X)
                if self.loss_fn is not None:
                    loss += self.loss_fn(y_hat, y).item()

            accuracy(y_hat, y)
            precision(y_hat, y)
            recall(y_hat, y)
            f1(y_hat, y)
        
        return {
            "accuracy": round(accuracy.compute().item(), 5),
            "precision": round(precision.compute().item(), 5),
            "recall": round(recall.compute().item(), 5),
            "f1": round(f1.compute().item(), 5),
            "loss": round(loss / len(self.data_loader), 5) if self.loss_fn is not None else None
        }
    
