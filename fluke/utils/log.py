"""This submodule provides logging utilities."""
import json
import os
import sys
import time
from typing import Any, Literal, Union

import numpy as np
import pandas as pd
import psutil
import rich
from clearml import Task
from rich.panel import Panel
from rich.pretty import Pretty
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

import wandb

sys.path.append(".")
sys.path.append("..")

from .. import DDict  # NOQA
from ..comm import ChannelObserver, Message  # NOQA
from . import ClientObserver, ServerObserver, get_class_from_str  # NOQA


__all__ = [
    "Log",
    "TensorboardLog",
    "WandBLog",
    "ClearMLLog",
    "get_logger"
]


class Log(ServerObserver, ChannelObserver, ClientObserver):
    """Basic logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process. The logging happens in the console.

    Attributes:
        global_eval (dict): The global evaluation metrics.
        locals_eval (dict): The clients local model evaluation metrics on the server's test set.
        prefit_eval (dict): The clients' pre-fit evaluation metrics.
        postfit_eval (dict): The clients' post-fit evaluation metrics.
        locals_eval_summary (dict): The mean of the clients local model evaluation metrics.
        prefit_eval_summary (dict): The mean of the clients pre-fit evaluation metrics.
        postfit_eval_summary (dict): The mean of the clients post-fit evaluation metrics.
        comm_costs (dict): The communication costs.
        current_round (int): The current round.
    """

    def __init__(self, **kwargs: dict[str, Any]):
        self.global_eval: dict = {}  # round -> evals
        self.locals_eval: dict = {}  # round -> {client_id -> evals}
        self.prefit_eval: dict = {}  # round -> {client_id -> evals}
        self.postfit_eval: dict = {}  # round -> {client_id -> evals}
        self.locals_eval_summary: dict = {}  # round -> evals (mean across clients)
        self.prefit_eval_summary: dict = {}  # round -> evals (mean across clients)
        self.postfit_eval_summary: dict = {}  # round -> evals (mean across clients)
        self.comm_costs: dict = {0: 0}
        self.current_round: int = 0

    def log(self, message: str) -> None:
        """Log a message.

        Args:
            message (str): The message to log.
        """
        rich.print(message)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        """Add a scalar to the logger.

        Args:
            key (Any): The key of the scalar.
            value (float): The value of the scalar.
            round (int): The round.
        """
        pass

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        """Add scalars to the logger.

        Args:
            key (Any): The main key of the scalars.
            values (dict[str, float]): The key-value pairs of the scalars.
            round (int): The round.
        """
        pass

    def pretty_log(self, data: Any, title: str) -> None:
        """Log a pretty-printed data.

        Args:
            data (Any): The data to log.
            title (str): The title of the data.
        """
        rich.print(Panel(Pretty(data, expand_all=True), title=title))

    def init(self, **kwargs: dict[str, Any]) -> None:
        """Initialize the logger.
        The initialization is done by printing the configuration in the console.

        Args:
            **kwargs: The configuration.
        """
        if kwargs:
            rich.print(Panel(Pretty(kwargs, expand_all=True), title="Configuration"))

    def start_round(self, round: int, global_model: Module) -> None:
        self.comm_costs[round] = 0
        self.current_round = round

        if round == 1 and self.comm_costs[0] > 0:
            rich.print(Panel(Pretty({"comm_costs": self.comm_costs[0]}), title=f"Round: {round-1}"))

    def end_round(self, round: int) -> None:
        stats = {}
        # Pre-fit summary
        if self.prefit_eval and round in self.prefit_eval and self.prefit_eval[round]:
            client_mean = pd.DataFrame(self.prefit_eval[round].values()).mean(
                numeric_only=True).to_dict()
            client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
            self.prefit_eval_summary[round] = client_mean
            stats['pre-fit'] = client_mean

        # Post-fit summary
        if self.postfit_eval and round in self.postfit_eval and self.postfit_eval[round]:
            client_mean = pd.DataFrame(self.postfit_eval[round].values()).mean(
                numeric_only=True).to_dict()
            client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
            self.postfit_eval_summary[round] = client_mean
            stats['post-fit'] = client_mean

        # Locals summary
        if self.locals_eval and round in self.locals_eval and self.locals_eval[round]:
            client_mean = pd.DataFrame(list(self.locals_eval[round].values())).mean(
                numeric_only=True).to_dict()
            client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
            self.locals_eval_summary[round] = client_mean
            stats['locals'] = self.locals_eval_summary[round]

        # Global summary
        if self.global_eval and round in self.global_eval and self.global_eval[round]:
            stats['global'] = self.global_eval[round]

        stats['comm_cost'] = self.comm_costs[round]

        if stats:
            rich.print(Panel(Pretty(stats, expand_all=True), title=f"Round: {round}"))
            rich.print(f"  Memory usage: {psutil.Process(os.getpid()).memory_percent():.2f} %")

    def client_evaluation(self,
                          round: int,
                          client_id: int,
                          phase: Literal['pre-fit', 'post-fit'],
                          evals: dict[str, float],
                          **kwargs: dict[str, Any]) -> None:

        if round == -1:
            round = self.current_round + 1
        dict_ref = self.prefit_eval if phase == 'pre-fit' else self.postfit_eval
        dict_ref[round] = {client_id: evals}

    def server_evaluation(self,
                          round: int,
                          type: Literal['global', 'locals'],
                          evals: Union[dict[str, float], dict[int, dict[str, float]]],
                          **kwargs: dict[str, Any]) -> None:

        if type == 'global':
            self.global_eval[round] = evals
        elif type == "locals" and evals:
            self.locals_eval[round] = evals

    def message_received(self, message: Message) -> None:
        """Update the communication costs.

        Args:
            message (Message): The message received.
        """
        self.comm_costs[self.current_round] += message.get_size()

    def finished(self, round: int) -> None:
        stats = {}

        # Pre-fit summary
        if self.prefit_eval and round in self.prefit_eval and self.prefit_eval[round]:
            client_mean = pd.DataFrame(self.prefit_eval[round].values()).mean(
                numeric_only=True).to_dict()
            client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
            self.prefit_eval_summary[round] = client_mean
            stats['pre-fit'] = client_mean

        # Locals summary
        if self.locals_eval:
            stats['locals'] = self.locals_eval_summary[max(self.locals_eval.keys())]

        # Post-fit summary
        if self.postfit_eval:
            stats['post-fit'] = self.postfit_eval_summary[max(self.global_eval.keys())]

        # Global summary
        if self.global_eval:
            stats['global'] = self.global_eval[max(self.global_eval.keys())]

        if stats:
            rich.print(Panel(Pretty(stats, expand_all=True), title="Overall Performance"))

        rich.print(Panel(Pretty({"comm_costs": sum(self.comm_costs.values())}, expand_all=True),
                         title="Total communication cost"))

    def save(self, path: str) -> None:
        """Save the logger's history to a JSON file.

        Args:
            path (str): The path to the JSON file.
        """
        json_to_save = {
            "perf_global": self.global_eval,
            "comm_costs": self.comm_costs,
            "perf_locals": self.locals_eval_summary,
            "perf_prefit": self.prefit_eval_summary,
            "perf_postfit": self.postfit_eval_summary
        }
        with open(path, 'w') as f:
            json.dump(json_to_save, f, indent=4)


class TensorboardLog(Log):
    """TensorBoard logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process on TensorBoard

    See Also:
        For more information on TensorBoard, see the `official documentation
        <https://www.tensorflow.org/tensorboard>`_.

    Args:
        **config: The configuration for TensorBoard.
    """

    def __init__(self, **config):
        super().__init__(**config)
        ts_config = DDict(**config).exclude("name")
        if "log_dir" not in ts_config:
            exp_name = config['name']
            if exp_name.startswith("fluke.algorithms."):
                exp_name = ".".join(str(exp_name).split(".")[3:])
            ts_config.log_dir = f"./runs/{exp_name}" + "_" + time.strftime("%Y%m%dh%H%M%S")
        self._writer = SummaryWriter(**ts_config)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        return self._writer.add_scalars(key, value, round)

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        return self._writer.add_scalars(key, values, round)

    def start_round(self, round: int, global_model: Module) -> None:
        super().start_round(round, global_model)
        if round == 1 and self.comm_costs[0] > 0:
            self._writer.add_scalar("comm_costs", self.comm_costs[0], round)
        self._writer.flush()

    def _report(self, prefix: str, evals: dict[str, float], round: int) -> None:
        for metric, value in evals.items():
            self._writer.add_scalar(f"{prefix}/{metric}", value, round)
            self._writer.flush()

    def end_round(self, round: int) -> None:
        super().end_round(round)
        self._report("global", self.global_eval[round], round)
        self._writer.add_scalar("comm_costs", self.comm_costs[round], round)
        self._writer.flush()

        if self.prefit_eval_summary and round in self.prefit_eval_summary:
            self._report("pre-fit", self.prefit_eval_summary[round], round)

        if self.postfit_eval_summary and round in self.postfit_eval_summary:
            self._report("post-fit", self.postfit_eval_summary[round], round)

        if self.locals_eval_summary and round in self.locals_eval_summary:
            self._report("locals", self.locals_eval_summary[round], round)

        self._writer.flush()

    def finished(self, round: int) -> None:
        super().finished(round)
        if self.prefit_eval_summary and round in self.prefit_eval_summary:
            self._report("pre-fit", self.prefit_eval_summary[round], round)

        if self.locals_eval_summary and round in self.locals_eval_summary:
            self._report("locals", self.locals_eval_summary[round], round)

        self._writer.flush()
        self._writer.close()

    def save(self, path: str) -> None:
        super().save(path)


class WandBLog(Log):
    """Weights and Biases logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process on Weights and Biases.

    See Also:
        For more information on Weights and Biases, see the `Weights and Biases documentation
        <https://docs.wandb.ai/>`_.

    Args:
        **config: The configuration for Weights and Biases.
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config

    def init(self, **kwargs: dict[str, Any]) -> None:
        super().init(**kwargs)
        self.config["config"] = kwargs
        self.run = wandb.init(**self.config)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        return self.run.log({key: value}, step=round)

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        return self.run.log({f"{key}/{k}": v for k, v in values.items()}, step=round)

    def start_round(self, round: int, global_model: Module) -> None:
        super().start_round(round, global_model)
        if round == 1 and self.comm_costs[0] > 0:
            self.run.log({"comm_costs": self.comm_costs[0]})

    def end_round(self, round: int) -> None:
        super().end_round(round)
        self.run.log({"global": self.global_eval[round]}, step=round)
        self.run.log({"comm_cost": self.comm_costs[round]}, step=round)

        if self.prefit_eval_summary and round in self.prefit_eval_summary:
            self.run.log({"prefit": self.prefit_eval_summary[round]}, step=round)

        if self.postfit_eval_summary and round in self.postfit_eval_summary:
            self.run.log({"postfit": self.postfit_eval_summary[round]}, step=round)

        if self.locals_eval_summary and round in self.locals_eval_summary:
            self.run.log({"locals": self.locals_eval_summary[round]}, step=round)

    def finished(self, round: int) -> None:
        super().finished(round)

        if self.prefit_eval_summary and round in self.prefit_eval_summary:
            self.run.log({"prefit": self.prefit_eval_summary[round]}, step=round)

        if self.locals_eval_summary and round in self.locals_eval_summary:
            self.run.log({"locals": self.locals_eval_summary[round]}, step=round)

    def save(self, path: str) -> None:
        super().save(path)
        self.run.finish()


class ClearMLLog(TensorboardLog):
    """ClearML logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process on ClearML.

    Note:
        The ClearML logger takes advantage of the TensorBoard logger, thus the logging also happens
        on TensorBoard. The logging folder is "./runs/{experiment_name}_{timestamp}".

    See Also:
        For more information on ClearML, see the `official documentation
        <https://clear.ml/docs/latest/docs/>`_.

    Args:
        **config: The configuration for ClearML.
    """

    def __init__(self, **config):
        super().__init__(name=config['name'])
        self.config = DDict(**config)

    def init(self, **kwargs: dict[str, Any]) -> None:
        super().init(**kwargs)
        self.task = Task.init(task_name=self.config.name, **self.config.exclude("name"))
        self.task.connect(kwargs)


def get_logger(lname: str, **kwargs: dict[str, Any]) -> Log:
    """Get a logger from its name.
    This function is used to get a logger from its name. It is used to dynamically import loggers.
    The supported loggers are the ones defined in the ``fluke.utils.log`` module.

    Args:
        lname (str): The name of the logger.
        **kwargs: The keyword arguments to pass to the logger's constructor.

    Returns:
        Log | WandBLog | ClearMLLog | TensorboardLog: The logger.
    """
    return get_class_from_str("fluke.utils.log", lname)(**kwargs)
