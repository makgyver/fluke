import sys
sys.path.append(".")

from abc import ABC, abstractmethod
from typing import Callable, Literal, Union, Iterable

import torch
from torch.nn import Module
from torchmetrics import Accuracy, Precision, Recall, F1Score

from .data import FastTensorDataLoader

class Evaluator(ABC):
    """Base class for all evaluators.

    This class is the base class for all evaluators in the `FL-bench`.

    Attributes:
        loss_fn (Callable): The loss function.
    """
    def __init__(self, loss_fn: Callable):
        self.loss_fn: Callable = loss_fn
    
    @abstractmethod
    def evaluate(self, model: Module, eval_data_loader: FastTensorDataLoader) -> dict:
        """Evaluate the model.

        Args:
            model (Module): The model to evaluate.
            eval_data_loader (FastTensorDataLoader): The data loader to use for evaluation.
        """
        pass

    def __call__(self, model: Module, eval_data_loader: FastTensorDataLoader) -> dict:
        """Evaluate the model.

        Args:
            model (Module): The model to evaluate.
            eval_data_loader (FastTensorDataLoader): The data loader to use for evaluation.
        """
        return self.evaluate(model, eval_data_loader)


class ClassificationEval(Evaluator):
    """Evaluate a classification pytorch model.

    The metrics computed are accuracy, precision, recall, f1 and the loss according to the provided
    loss function `loss_fn`.

    Attributes:
        average (Literal["micro","macro"]): The average to use for the metrics.
        n_classes (int): The number of classes.
        device (torch.device): The device where the evaluation is performed.
    """
    def __init__(self, 
                 loss_fn: Callable, 
                 n_classes: int, 
                 average: Literal["micro","macro"]="micro",
                 device: torch.device=torch.device("cpu")):
        super().__init__(loss_fn)
        self.average: str = average
        self.n_classes: int = n_classes
        self.device: torch.device = device

    def evaluate(self, 
                 model: torch.nn.Module, 
                 eval_data_loader: Union[FastTensorDataLoader, Iterable[FastTensorDataLoader]]) -> dict:
        model.eval()
        model.to(self.device)
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
        
        model.to("cpu")

        return {
            "accuracy":  round(sum(accs) / len(accs), 5),
            "precision": round(sum(precs) / len(precs), 5),
            "recall":    round(sum(recs) / len(recs), 5),
            "f1":        round(sum(f1s) / len(f1s), 5),
            "loss":      round(loss / cnt, 5) if self.loss_fn is not None else None
        }
    