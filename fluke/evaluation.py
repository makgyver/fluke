from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.nn import Module
import torch
from typing import Callable, Optional, Union, Iterable
from abc import ABC, abstractmethod
import sys
sys.path.append(".")

from .data import FastTensorDataLoader  # NOQA
from . import GlobalSettings  # NOQA


class Evaluator(ABC):
    """This class is the base class for all evaluators in `FL-bench`.

    An evaluator object should be used to perform the evaluation of a model.

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

        This method is equivalent to `evaluate`.

        Args:
            model (Module): The model to evaluate.
            eval_data_loader (FastTensorDataLoader): The data loader to use for evaluation.
        """
        return self.evaluate(model, eval_data_loader)


class ClassificationEval(Evaluator):
    """Evaluate a classification pytorch model.

    The metrics computed are `accuracy`, `precision`, `recall`, `f1` and the loss according
    to the provided loss function `loss_fn`.

    Attributes:
        average (Literal["micro","macro"]): The average to use for the metrics.
        n_classes (int): The number of classes.
        device (Optional[torch.device]): The device where the evaluation is performed. If `None`,
            the device is the one set in the `GlobalSettings`.
    """

    def __init__(self,
                 loss_fn: Callable,
                 n_classes: int,
                 device: Optional[torch.device] = None):
        super().__init__(loss_fn)
        self.n_classes: int = n_classes
        self.device: torch.device = device if device is not None else GlobalSettings().get_device()

    def evaluate(self,
                 model: torch.nn.Module,
                 eval_data_loader: Union[FastTensorDataLoader,
                                         Iterable[FastTensorDataLoader]]) -> dict:
        """Evaluate the model.

        Args:
            model (torch.nn.Module): The model to evaluate. If `None`, the method returns an
                empty dictionary.
            eval_data_loader (Union[FastTensorDataLoader, Iterable[FastTensorDataLoader]]):
                The data loader(s) to use for evaluation. If `None`, the method returns an empty
                dictionary.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        if (model is None) or (eval_data_loader is None):
            return {}

        model.eval()
        model.to(self.device)
        task = "multiclass"  # if self.n_classes >= 2 else "binary"
        accs, losses = [], []
        micro_precs, micro_recs, micro_f1s = [], [], []
        macro_precs, macro_recs, macro_f1s = [], [], []
        loss, cnt = 0, 0

        if not isinstance(eval_data_loader, list):
            eval_data_loader = [eval_data_loader]

        for data_loader in eval_data_loader:
            accuracy = Accuracy(task=task, num_classes=self.n_classes, top_k=1, average="micro")
            micro_precision = Precision(
                task=task, num_classes=self.n_classes, top_k=1, average="micro")
            micro_recall = Recall(task=task, num_classes=self.n_classes, top_k=1, average="micro")
            micro_f1 = F1Score(task=task, num_classes=self.n_classes, top_k=1, average="micro")
            macro_precision = Precision(
                task=task, num_classes=self.n_classes, top_k=1, average="macro")
            macro_recall = Recall(task=task, num_classes=self.n_classes, top_k=1, average="macro")
            macro_f1 = F1Score(task=task, num_classes=self.n_classes, top_k=1, average="macro")
            loss = 0
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                with torch.no_grad():
                    y_hat = model(X)
                    if self.loss_fn is not None:
                        loss += self.loss_fn(y_hat, y).item()

                accuracy.update(y_hat.cpu(), y.cpu())
                micro_precision.update(y_hat.cpu(), y.cpu())
                micro_recall.update(y_hat.cpu(), y.cpu())
                micro_f1.update(y_hat.cpu(), y.cpu())
                macro_precision.update(y_hat.cpu(), y.cpu())
                macro_recall.update(y_hat.cpu(), y.cpu())
                macro_f1.update(y_hat.cpu(), y.cpu())

            cnt += len(data_loader)
            if cnt == 0:
                return {}
            accs.append(accuracy.compute().item())
            micro_precs.append(micro_precision.compute().item())
            micro_recs.append(micro_recall.compute().item())
            micro_f1s.append(micro_f1.compute().item())
            macro_precs.append(macro_precision.compute().item())
            macro_recs.append(macro_recall.compute().item())
            macro_f1s.append(macro_f1.compute().item())
            losses.append(loss / cnt)

        model.to("cpu")

        return {
            "accuracy":  round(sum(accs) / len(accs), 5),
            "micro_precision": round(sum(micro_precs) / len(micro_precs), 5),
            "micro_recall":    round(sum(micro_recs) / len(micro_recs), 5),
            "micro_f1":        round(sum(micro_f1s) / len(micro_f1s), 5),
            "macro_precision": round(sum(macro_precs) / len(macro_precs), 5),
            "macro_recall":    round(sum(macro_recs) / len(macro_recs), 5),
            "macro_f1":        round(sum(macro_f1s) / len(macro_f1s), 5),
            "loss":  round(sum(losses) / len(losses), 5) if self.loss_fn is not None else None
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(n_classes={self.n_classes},average={self.average}," + \
               f"device={self.device})[accuracy,precision,recall,f1," + \
               f"{self.loss_fn.__class__.__name__}]"

    def __repr__(self) -> str:
        return str(self)
