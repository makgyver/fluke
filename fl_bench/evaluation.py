from abc import ABC, abstractmethod
from typing import Callable
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch

import sys; sys.path.append(".")
from fl_bench import GlobalSettings
from fl_bench.data import FastTensorDataLoader

class Evaluator(ABC):
    def __init__(self, data_loader: FastTensorDataLoader, loss_fn: Callable):
        self.data_loader = data_loader
        self.loss_fn = loss_fn
    
    @abstractmethod
    def evaluate(self, model):
        pass

    def __call__(self, model):
        return self.evaluate(model)


class ClassificationEval(Evaluator):
    def evaluate(self, model: torch.nn.Module):
        model.eval()
        accuracy = Accuracy(average='micro')
        precision = Precision(average='micro')
        recall = Recall(average='micro')
        f1 = F1Score(average='micro')
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
    
