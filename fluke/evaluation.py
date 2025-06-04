"""This module contains the definition of the evaluation classes used to perform the evaluation
of the model client-side and server-side."""

import sys
from abc import ABC, abstractmethod
from typing import Any, Collection, Optional, Union, Literal

from pandas import DataFrame
import numpy as np
import torch
from torch.nn import Module
from torchmetrics import Accuracy, F1Score, Metric, Precision, Recall

sys.path.append(".")
sys.path.append("..")

from .data import FastDataLoader  # NOQA

__all__ = ["Evaluator", "ClassificationEval", "PerformanceTracker"]


class Evaluator(ABC):
    """This class is the base class for all evaluators in :mod:`fluke`.
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
    def evaluate(
        self,
        round: int,
        model: Module,
        eval_data_loader: FastDataLoader,
        loss_fn: Optional[torch.nn.Module],
        additional_metrics: Optional[dict[str, Metric]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Evaluate the model.

        Args:
            round (int): The current
            model (Module): The model to evaluate.
            eval_data_loader (FastDataLoader): The data loader to use for evaluation.
            loss_fn (torch.nn.Module, optional): The loss function to use for evaluation.
            additional_metrics (dict[str, Metric], optional): Additional metrics to use for
                evaluation. If provided, they are added to the default metrics.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the computed metrics.
        """
        raise NotImplementedError

    def __call__(
        self,
        round: int,
        model: Module,
        eval_data_loader: FastDataLoader,
        loss_fn: Optional[torch.nn.Module],
        additional_metrics: Optional[dict[str, Metric]] = None,
        **kwargs,
    ) -> dict:
        """Evaluate the model.

        Note:
            This method is equivalent to ``evaluate``.

        Args:
            round (int): The current round.
            model (Module): The model to evaluate.
            eval_data_loader (FastDataLoader): The data loader to use for evaluation.
            loss_fn (torch.nn.Module, optional): The loss function to use for evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the computed metrics.
        """
        return self.evaluate(
            round=round,
            model=model,
            eval_data_loader=eval_data_loader,
            loss_fn=loss_fn,
            **kwargs,
        )


class ClassificationEval(Evaluator):
    """Evaluate a PyTorch model for classification.
    The metrics computed are ``accuracy``, ``precision``, ``recall``, ``f1`` and the loss according
    to the provided loss function ``loss_fn`` when calling the method ``evaluation``.
    Metrics are computed both in a micro and macro fashion.

    Args:
        eval_every (int): The evaluation frequency.
        n_classes (int): The number of classes.
        **metrics (dict[str, Metric]): The metrics to use for evaluation. If not provided, the
            default metrics are used: ``accuracy``, ``macro_precision``, ``macro_recall``,
            ``macro_f1``, ``micro_precision``, ``micro_recall`` and ``micro_f1``.

    Attributes:
        eval_every (int): The evaluation frequency.
        n_classes (int): The number of classes.
    """

    def __init__(self, eval_every: int, n_classes: int, **metrics: Metric):
        super().__init__(eval_every=eval_every)
        self.n_classes: int = n_classes

        self.metrics = {}

        # if kwargs is empty
        if not metrics:
            self.metrics = {
                "accuracy": Accuracy(task="multiclass", num_classes=self.n_classes, top_k=1),
                "macro_precision": Precision(
                    task="multiclass",
                    num_classes=self.n_classes,
                    top_k=1,
                    average="macro",
                ),
                "macro_recall": Recall(
                    task="multiclass",
                    num_classes=self.n_classes,
                    top_k=1,
                    average="macro",
                ),
                "macro_f1": F1Score(
                    task="multiclass",
                    num_classes=self.n_classes,
                    top_k=1,
                    average="macro",
                ),
                "micro_precision": Precision(
                    task="multiclass",
                    num_classes=self.n_classes,
                    top_k=1,
                    average="micro",
                ),
                "micro_recall": Recall(
                    task="multiclass",
                    num_classes=self.n_classes,
                    top_k=1,
                    average="micro",
                ),
                "micro_f1": F1Score(
                    task="multiclass",
                    num_classes=self.n_classes,
                    top_k=1,
                    average="micro",
                ),
            }
        else:
            self.metrics = metrics

    def add_metric(self, name: str, metric: Metric) -> None:
        """Add a metric to the evaluator.

        Args:
            name (str): The name of the metric.
            metric (Metric): The metric to add.
        """
        if name in self.metrics:
            raise ValueError(f"Metric {name} already exists.")
        self.metrics[name] = metric

    @torch.no_grad()
    def evaluate(
        self,
        round: int,
        model: torch.nn.Module,
        eval_data_loader: Union[FastDataLoader, Collection[FastDataLoader]],
        loss_fn: Optional[torch.nn.Module] = None,
        additional_metrics: Optional[dict[str, Metric]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """Evaluate the model. The metrics computed are ``accuracy``, ``precision``, ``recall``,
        ``f1`` and the loss according to the provided loss function ``loss_fn``. Metrics are
        computed both in a micro and macro fashion.

        Warning:
            The loss function ``loss_fn`` should be defined on the same device as the model.
            Moreover, it is assumed that the only arguments of the loss function are the predicted
            values and the true values.

        Args:
            round (int): The current round.
            model (torch.nn.Module): The model to evaluate. If ``None``, the method returns an
                empty dictionary.
            eval_data_loader (Union[FastDataLoader, Collection[FastDataLoader]]):
                The data loader(s) to use for evaluation. If ``None``, the method returns an empty
                dictionary.
            loss_fn (torch.nn.Module, optional): The loss function to use for evaluation.
            additional_metrics (dict[str, Metric], optional): Additional metrics to use for
                evaluation. If provided, they are added to the default metrics.
            device (torch.device, optional): The device to use for evaluation. Defaults to "cpu".

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        from .utils import clear_cuda_cache  # NOQA

        if (round != 1) and (round % self.eval_every != 0):
            return {}

        if (model is None) or (eval_data_loader is None):
            return {}

        model_device = torch.device("cpu")
        if next(model.parameters(), None) is not None:
            model_device = next(model.parameters()).device
        model.eval()
        model.to(device)
        losses = []
        matrics_values = {k: [] for k in self.metrics.keys()}
        loss, cnt = 0, 0

        if additional_metrics is None:
            additional_metrics = {}

        add_metric_values = {k: [] for k in additional_metrics.keys()}

        if not isinstance(eval_data_loader, list):
            eval_data_loader = [eval_data_loader]

        for data_loader in eval_data_loader:
            for metric in self.metrics.values():
                metric.reset()

            for metric in additional_metrics.values():
                metric.reset()

            loss = 0
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                with torch.no_grad():
                    y_hat = model(X)
                    if loss_fn is not None:
                        loss += loss_fn(y_hat, y).item()

                for metric in self.metrics.values():
                    metric.update(y_hat.cpu(), y.cpu())

                if additional_metrics:
                    for metric in additional_metrics.values():
                        metric.update(y_hat.cpu(), y.cpu())

            cnt += len(data_loader)

            for k, v in self.metrics.items():
                matrics_values[k].append(v.compute().item())

            if additional_metrics:
                for k, v in additional_metrics.items():
                    add_metric_values[k].append(v.compute().item())

            losses.append(loss / cnt)

        model.to(model_device)
        clear_cuda_cache()

        result = {m: np.round(sum(v) / len(v), 5).item() for m, v in matrics_values.items()}
        result.update(
            {m: np.round(sum(v) / len(v), 5).item() for m, v in add_metric_values.items()}
        )

        if loss_fn is not None:
            result["loss"] = np.round(sum(losses) / len(losses), 5).item()

        return result

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(eval_every={self.eval_every}"
            + f", n_classes={self.n_classes})[{', '.join(self.metrics.keys())}]"
        )

    def __repr__(self) -> str:
        return str(self)


def _compute_mean(evals: dict[str, float]) -> dict[str, float]:
    df_data = DataFrame(evals.values())
    client_mean = df_data.mean(numeric_only=True).to_dict()
    return {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}


class PerformanceTracker:

    def __init__(self):

        self._performance = {
            "global": {},  # round -> evals
            "locals": {},  # round -> {client_id -> evals}
            "pre-fit": {},  # round -> {client_id -> evals}
            "post-fit": {},  # round -> {client_id -> evals}
            "comm": {0: 0},  # round -> comm_cost
            "mem": {},  # round -> mem_usage
        }

    def add(
        self,
        perf_type: Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem"],
        metrics: dict[str, float] | float,
        round: int = 0,
        client_id: Optional[int] = None,
    ) -> None:
        """Add performance metrics for a specific type and client.

        Args:
            perf_type (Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem"]):
                The type of performance metrics to add.
            metrics (dict[str, float] | float): The performance metrics to add. If `perf_type`
                is "comm", this should be a single float value representing the communication cost.
            round (int, optional): The current round. Defaults to 0.
            client_id (int, optional): The client ID for local performance metrics. Defaults to
                None for global metrics.
        """
        if perf_type not in self._performance:
            raise ValueError(f"Unknown performance type: {perf_type}")

        if perf_type == "comm" or perf_type == "mem":
            if not isinstance(metrics, (float, int)):
                raise ValueError(f"Metrics for {perf_type} must be a float, got {type(metrics)}")
            if round not in self._performance[perf_type]:
                self._performance[perf_type][round] = metrics
            else:
                self._performance[perf_type][round] += metrics
        else:
            if round not in self._performance[perf_type]:
                self._performance[perf_type][round] = {}
            if perf_type != "global":
                self._performance[perf_type][round][client_id] = metrics
            else:
                self._performance[perf_type][round] = metrics

    def get(
        self,
        perf_type: Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem"],
        round: int,
    ) -> Union[dict, float]:
        """Get performance metrics for a specific type and round.

        Args:
            perf_type (Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem"]):
                The type of performance metrics to retrieve.
            round (int): The round for which to retrieve the metrics.

        Raises:
            ValueError: If the `perf_type` is unknown.

        Returns:
            Union[dict, float]: The performance metrics for the specified type and round.
                If `perf_type` is "comm" or "mem", returns a float; otherwise, returns a dict.
        """
        if perf_type not in self._performance:
            raise ValueError(f"Unknown performance type: {perf_type}")

        if round not in self._performance[perf_type]:
            return {} if perf_type not in ["comm", "mem"] else 0.0

        if perf_type in ["comm", "mem"]:
            return self._performance[perf_type][round]
        else:
            return self._performance[perf_type][round].copy()

    def __getitem__(self, item: str) -> dict:
        """Get performance metrics for a specific type.

        Args:
            item (str): The type of performance metrics to retrieve.

        Raises:
            ValueError: If the `item` is unknown.

        Returns:
            dict: The performance metrics for the specified type.
        """
        if item not in self._performance:
            raise ValueError(f"Unknown performance type: {item}")
        return self._performance[item].copy()

    def summary(
        self,
        perf_type: Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem"],
        round: int,
        include_round: bool = True,
        force_round: bool = True,
    ) -> Union[dict, float]:
        """Get the summary of the performance metrics for a specific type.

        Summary metrics are computed as the mean of the metrics for the specified type
        and round. If `perf_type` is "comm", the total communication cost is returned.
        If `perf_type` is "mem", the memory usage is returned.
        If `perf_type` is "global", the metrics are returned as they are.

        Args:
            perf_type (Literal["global", "locals", "pre-fit", "post-fit", "comm", "mem"]):
                The type of performance metrics to retrieve.
            round (int): The round for which to compute the summary of the metrics.
            include_round (bool, optional): Whether to include the round number in the returned
                metrics. Defaults to `True`.
            force_round (bool, optional): If `True`, the method will return the metrics for the
                specified round if it exists, otherwise it will return the metrics for the
                latest round. Defaults to `False`.

        Raises:
            ValueError: If the `perf_type` is unknown or if there are no metrics for the specified
                type and round.

        Returns:
            Union[dict, float]: The summary performance metrics for the specified type.
        """
        if perf_type not in self._performance:
            raise ValueError(f"Unknown performance type: {perf_type}")

        if not self._performance[perf_type]:
            return {} if perf_type not in ["comm", "mem"] else 0.0

        if force_round:
            the_round = max(self._performance[perf_type].keys())
        elif round not in self._performance[perf_type]:
            return {}
        else:
            the_round = round

        if perf_type == "mem":
            return self._performance[perf_type][the_round]

        if perf_type == "comm":
            return sum(list(self._performance[perf_type].values()))

        if perf_type == "global":
            metrics = self._performance[perf_type][the_round].copy()
        else:
            metrics = _compute_mean(self._performance[perf_type][the_round])
            metrics["support"] = len(self._performance[perf_type][the_round])

        if include_round and metrics:
            metrics["round"] = the_round
        return metrics
