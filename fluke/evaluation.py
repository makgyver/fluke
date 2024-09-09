"""This module contains the definition of the evaluation classes used to perform the evaluation
of the model client-side and server-side."""
import sys
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch
from torch.nn import Module
from torchmetrics import Accuracy, F1Score, Precision, Recall

sys.path.append(".")
sys.path.append("..")

from .data import FastDataLoader  # NOQA

__all__ = [
    "Evaluator",
    "ClassificationEval"
]


class Evaluator(ABC):
    """This class is the base class for all evaluators in ``fluke``.
    An evaluator object should be used to perform the evaluation of a (federated) model.

    Args:
        eval_every (int): The evaluation frequency expressed as the number of rounds between
          two evaluations. Defaults to 1, i.e., evaluate the model at each round.

    Attributes:
        eval_every (int): The evaluation frequency.
    """

    def __init__(self, eval_every: int = 1):
        self.eval_every: int = eval_every

    @abstractmethod
    def evaluate(self,
                 round: int,
                 model: Module,
                 eval_data_loader: FastDataLoader,
                 loss_fn: Optional[torch.nn.Module],
                 **kwargs: dict[str, Any]) -> dict:
        """Evaluate the model.

        Args:
            round (int): The current
            model (Module): The model to evaluate.
            eval_data_loader (FastDataLoader): The data loader to use for evaluation.
            loss_fn (torch.nn.Module, optional): The loss function to use for evaluation.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    def __call__(self,
                 round: int,
                 model: Module,
                 eval_data_loader: FastDataLoader,
                 loss_fn: Optional[torch.nn.Module],
                 **kwargs: dict[str, Any]) -> dict:
        """Evaluate the model.

        Note:
            This method is equivalent to ``evaluate``.

        Args:
            round (int): The current round.
            model (Module): The model to evaluate.
            eval_data_loader (FastDataLoader): The data loader to use for evaluation.
            loss_fn (torch.nn.Module, optional): The loss function to use for evaluation.
            **kwargs: Additional keyword arguments.
        """
        return self.evaluate(round=round,
                             model=model,
                             eval_data_loader=eval_data_loader,
                             loss_fn=loss_fn,
                             **kwargs)


class ClassificationEval(Evaluator):
    """Evaluate a PyTorch model for classification.
    The metrics computed are ``accuracy``, ``precision``, ``recall``, ``f1`` and the loss according
    to the provided loss function ``loss_fn`` when calling the method ``evaluation``.
    Metrics are computed both in a micro and macro fashion.

    Args:
        eval_every (int): The evaluation frequency.
        n_classes (int): The number of classes.

    Attributes:
        eval_every (int): The evaluation frequency.
        n_classes (int): The number of classes.
    """

    def __init__(self, eval_every: int, n_classes: int):
        super().__init__(eval_every=eval_every)
        self.n_classes: int = n_classes

    def evaluate(self,
                 round: int,
                 model: torch.nn.Module,
                 eval_data_loader: Union[FastDataLoader,
                                         Iterable[FastDataLoader]],
                 loss_fn: Optional[torch.nn.Module] = None,
                 device: torch.device = torch.device("cpu")) -> dict:
        """Evaluate the model. The metrics computed are ``accuracy``, ``precision``, ``recall``,
        ``f1`` and the loss according to the provided loss function ``loss_fn``. Metrics are
        computed both in a micro and macro fashion.

        Args:
            round (int): The current round.
            model (torch.nn.Module): The model to evaluate. If ``None``, the method returns an
                empty dictionary.
            eval_data_loader (Union[FastDataLoader, Iterable[FastDataLoader]]):
                The data loader(s) to use for evaluation. If ``None``, the method returns an empty
                dictionary.
            loss_fn (torch.nn.Module, optional): The loss function to use for evaluation.
            device (torch.device, optional): The device to use for evaluation. Defaults to "cpu".

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        from .utils import clear_cache  # NOQA

        if round % self.eval_every != 0:
            return {}

        if (model is None) or (eval_data_loader is None):
            return {}

        model.eval()
        model.to(device)
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
                X, y = X.to(device), y.to(device)
                with torch.no_grad():
                    y_hat = model(X)
                    if loss_fn is not None:
                        loss += loss_fn(y_hat, y).item()

                accuracy.update(y_hat.cpu(), y.cpu())
                micro_precision.update(y_hat.cpu(), y.cpu())
                micro_recall.update(y_hat.cpu(), y.cpu())
                micro_f1.update(y_hat.cpu(), y.cpu())
                macro_precision.update(y_hat.cpu(), y.cpu())
                macro_recall.update(y_hat.cpu(), y.cpu())
                macro_f1.update(y_hat.cpu(), y.cpu())

            cnt += len(data_loader)
            accs.append(accuracy.compute().item())
            micro_precs.append(micro_precision.compute().item())
            micro_recs.append(micro_recall.compute().item())
            micro_f1s.append(micro_f1.compute().item())
            macro_precs.append(macro_precision.compute().item())
            macro_recs.append(macro_recall.compute().item())
            macro_f1s.append(macro_f1.compute().item())
            losses.append(loss / cnt)

        model.to("cpu")
        clear_cache()

        result = {
            "accuracy":  np.round(sum(accs) / len(accs), 5).item(),
            "micro_precision": np.round(sum(micro_precs) / len(micro_precs), 5).item(),
            "micro_recall":    np.round(sum(micro_recs) / len(micro_recs), 5).item(),
            "micro_f1":        np.round(sum(micro_f1s) / len(micro_f1s), 5).item(),
            "macro_precision": np.round(sum(macro_precs) / len(macro_precs), 5).item(),
            "macro_recall":    np.round(sum(macro_recs) / len(macro_recs), 5).item(),
            "macro_f1":        np.round(sum(macro_f1s) / len(macro_f1s), 5).item()
        }

        if loss_fn is not None:
            result["loss"] = np.round(sum(losses) / len(losses), 5).item()

        return result

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(eval_every={self.eval_every}" + \
               f", n_classes={self.n_classes})[accuracy, precision, recall, f1]"

    def __repr__(self) -> str:
        return str(self)
