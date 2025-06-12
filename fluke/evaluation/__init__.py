"""This module contains the definition of the evaluation classes used to perform the evaluation
of the model client-side and server-side."""

import sys
from typing import Collection, Literal, Optional, Union

import numpy as np
import torch
from pandas import DataFrame
from torch.nn import Module
from torchmetrics import Metric

sys.path.append(".")
sys.path.append("..")

from ..data import FastDataLoader  # NOQA
from ..utils import num_accepted_args  # NOQA

__all__ = ["classification", "fairness", "Evaluator", "CompoundEvaluator", "PerformanceTracker"]


class Evaluator:
    """This class is the base class for all evaluators in :mod:`fluke`.
    An evaluator object should be used to perform the evaluation of a (federated) model.

    Args:
        eval_every (int): The evaluation frequency expressed as the number of rounds between
            two evaluations. Defaults to 1, i.e., evaluate the model at each round.

    Attributes:
        eval_every (int): The evaluation frequency.
    """

    def __init__(self, eval_every: int = 1, **kwargs):
        self.eval_every: int = eval_every

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
        from ..utils import clear_cuda_cache  # NOQA

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

        if additional_metrics is None:
            additional_metrics = {}

        add_metric_values = {k: [] for k in additional_metrics.keys()}

        loss_n_args = (num_accepted_args(loss_fn) - 2) if loss_fn is not None else 2
        metrics_n_args = {k: (num_accepted_args(v.update) - 2) for k, v in self.metrics.items()}
        add_metrics_n_args = (
            {k: (num_accepted_args(v.update) - 2) for k, v in additional_metrics.items()}
            if additional_metrics
            else {}
        )

        for metric in self.metrics.values():
            metric.reset()

        for metric in additional_metrics.values():
            metric.reset()

        loss = 0
        for X, y, *z in eval_data_loader:
            X, y = X.to(device), y.to(device)
            if len(z) > 0:
                z = [item.to(device) for item in z]

            with torch.no_grad():
                y_hat = model(X)
                if loss_fn is not None:
                    loss += loss_fn(y_hat, y, *(z[:loss_n_args] if z else [])).item()

            for name, metric in self.metrics.items():
                metric.update(
                    y_hat.cpu(),
                    y.cpu(),
                    *([item.cpu() for item in z[: metrics_n_args[name]]] if z else []),
                )

            if additional_metrics:
                for name, metric in additional_metrics.items():
                    metric.update(
                        y_hat.cpu(),
                        y.cpu(),
                        *([item.cpu() for item in z[: add_metrics_n_args[name]]] if z else []),
                    )

        for k, v in self.metrics.items():
            matrics_values[k].append(v.compute().item())

        if additional_metrics:
            for k, v in additional_metrics.items():
                add_metric_values[k].append(v.compute().item())

        losses.append(loss / len(eval_data_loader))

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
            additional_metrics=additional_metrics,
            **kwargs,
        )


class CompoundEvaluator(Evaluator):
    """This class is a compound evaluator that combines multiple evaluators.

    Args:
        eval_every (int): The evaluation frequency expressed as the number of rounds between
            two evaluations. Defaults to 1, i.e., evaluate the model at each round.
        evaluators (list[Evaluator]): A list of evaluators to combine.

    Attributes:
        evaluators (list[Evaluator]): The list of evaluators to combine.
    """

    def __init__(self, eval_every: int = 1, *evaluators: Evaluator):
        super().__init__(eval_every)
        self.evaluators = list(evaluators)

    def add_evaluator(self, evaluator: Evaluator) -> None:
        """Add an evaluator to the compound evaluator."""
        self.evaluators.append(evaluator)

    def evaluate(
        self,
        round: int,
        model: torch.nn.Module,
        eval_data_loader: Union[FastDataLoader, Collection[FastDataLoader]],
        loss_fn: Optional[torch.nn.Module] = None,
        additional_metrics: Optional[dict[str, Metric]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """Evaluate the model using all evaluators."""
        results = {}
        for evaluator in self.evaluators:
            results.update(
                evaluator.evaluate(
                    round=round,
                    model=model,
                    eval_data_loader=eval_data_loader,
                    loss_fn=loss_fn,
                    additional_metrics=additional_metrics,
                    device=device,
                )
            )
        return results

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(eval_every={self.eval_every}, "
            + f"n_evaluators={len(self.evaluators)})"
            + f"[{', '.join([str(e) for e in self.evaluators])}]"
        )


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
