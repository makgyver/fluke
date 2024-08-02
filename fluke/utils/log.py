from rich.pretty import Pretty
from rich.panel import Panel
import rich
import wandb
from torch.utils.tensorboard import SummaryWriter
from clearml import Task
import psutil
import pandas as pd
import numpy as np
import json
import time
import os
from torch.nn import Module
from typing import Any, Sequence

from ..comm import ChannelObserver, Message  # NOQA
from . import ServerObserver, get_class_from_str  # NOQA
from .. import DDict


class Log(ServerObserver, ChannelObserver):
    """Basic logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process. The logging happens in the console.

    Attributes:
        history (dict): The history of the global model's performance.
        client_history (dict): The history of the clients' performance.
        comm_costs (dict): The history of the communication costs.
        current_round (int): The current round.
    """

    def __init__(self, **kwargs):
        self.history: dict = {}
        self.client_history: dict = {}
        self.comm_costs: dict = {0: 0}
        self.current_round: int = 0

    def init(self, **kwargs):
        """Initialize the logger.
        The initialization is done by printing the configuration in the console.

        Args:
            **kwargs: The configuration.
        """
        if kwargs:
            rich.print(Panel(Pretty(kwargs, expand_all=True), title="Configuration"))

    def start_round(self, round: int, global_model: Module):
        self.comm_costs[round] = 0
        self.current_round = round

        if round == 1 and self.comm_costs[0] > 0:
            rich.print(Panel(Pretty({"comm_costs": self.comm_costs[0]}), title=f"Round: {round-1}"))

    def end_round(self,
                  round: int,
                  global_eval: dict[str, float],
                  client_evals: Sequence[Any]):
        self.history[round] = global_eval
        stats = {'global': self.history[round]}

        if client_evals:
            client_mean = pd.DataFrame(client_evals).mean(numeric_only=True).to_dict()
            client_mean = {k: np.round(float(v), 5) for k, v in client_mean.items()}
            self.client_history[round] = client_mean
            stats['local'] = client_mean

        stats['comm_cost'] = self.comm_costs[round]
        if stats['global'] or ('local' in stats and stats['local']):
            rich.print(Panel(Pretty(stats, expand_all=True), title=f"Round: {round}"))
            rich.print(f"  MEMORY USAGE: {psutil.Process(os.getpid()).memory_percent():.2f} %")

    def message_received(self, message: Message):
        self.comm_costs[self.current_round] += message.get_size()

    def finished(self, client_evals: Sequence[Any]):
        if client_evals:
            client_mean = pd.DataFrame(client_evals).mean(numeric_only=True).to_dict()
            client_mean = {k: np.round(float(v), 5) for k, v in client_mean.items()}
            self.client_history[self.current_round + 1] = client_mean
            rich.print(Panel(Pretty(client_mean, expand_all=True),
                             title="Overall local performance"))

        if self.history[self.current_round]:
            rich.print(Panel(Pretty(self.history[self.current_round], expand_all=True),
                             title="Overall global performance"))

        rich.print(Panel(Pretty({"comm_costs": sum(self.comm_costs.values())}, expand_all=True),
                         title="Total communication cost"))

    def save(self, path: str):
        """Save the logger's history to a JSON file.

        Args:
            path (str): The path to the JSON file.
        """
        json_to_save = {
            "perf_global": self.history,
            "comm_costs": self.comm_costs,
            "perf_local": self.client_history
        }
        with open(path, 'w') as f:
            json.dump(json_to_save, f, indent=4)

    def error(self, error: str):
        """Log an error.

        Args:
            error (str): The error message.
        """
        rich.print(f"[bold red]Error: {error}[/bold red]")


class TensorBoardLog(Log):
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
        print(config)
        if "log_dir" not in ts_config:
            exp_name = config['name']
            if exp_name.startswith("fluke.algorithms."):
                exp_name = ".".join(str(exp_name).split(".")[3:])
            ts_config.log_dir = f"./runs/{exp_name}" + "_" + time.strftime("%Y%m%dh%H%M%S")
        self._writer = SummaryWriter(**ts_config)

    def start_round(self, round: int, global_model: Module):
        super().start_round(round, global_model)
        if round == 1 and self.comm_costs[0] > 0:
            self._writer.add_scalar("comm_costs", self.comm_costs[0], round)
        self._writer.flush()

    def end_round(self, round: int, global_eval: dict[str, float], client_evals: Sequence[Any]):
        super().end_round(round, global_eval, client_evals)
        self._writer.add_scalars("global", self.history[round], round)
        self._writer.add_scalar("comm_costs", self.comm_costs[round], round)
        if client_evals:
            self._writer.add_scalars("local", self.client_history[round], round)
        self._writer.flush()

    def finished(self, client_evals: Sequence[Any]):
        super().finished(client_evals)
        if client_evals:
            self._writer.add_scalars("local",
                                     self.client_history[self.current_round+1],
                                     self.current_round+1)
            self._writer.flush()
        self._writer.close()

    def save(self, path: str):
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

    def init(self, **kwargs):
        super().init(**kwargs)
        self.config["config"] = kwargs
        self.run = wandb.init(**self.config)

    def start_round(self, round: int, global_model: Module):
        super().start_round(round, global_model)
        if round == 1 and self.comm_costs[0] > 0:
            self.run.log({"comm_costs": self.comm_costs[0]})

    def end_round(self, round: int, global_eval: dict[str, float], client_evals: Sequence[Any]):
        super().end_round(round, global_eval, client_evals)
        self.run.log({"global": self.history[round]}, step=round)
        self.run.log({"comm_cost": self.comm_costs[round]}, step=round)
        if client_evals:
            self.run.log({"local": self.client_history[round]}, step=round)

    def finished(self, client_evals: Sequence[Any]):
        super().finished(client_evals)
        if client_evals:
            self.run.log(
                {"local": self.client_history[self.current_round+1]}, step=self.current_round+1)

    def save(self, path: str):
        super().save(path)
        self.run.finish()


class ClearMLLog(TensorBoardLog):
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

    def init(self, **kwargs):
        super().init(**kwargs)
        self.task = Task.init(**self.config.exclude("name"))
        self.task.connect(kwargs)


def get_logger(lname: str, **kwargs) -> Log:
    """Get a logger from its name.
    This function is used to get a logger from its name. It is used to dynamically import loggers.
    The supported loggers are the ones defined in the ``fluke.utils.log`` module.

    Args:
        lname (str): The name of the logger.
        **kwargs: The keyword arguments to pass to the logger's constructor.

    Returns:
        Log | WandBLog | TensorBoardLog: The logger.
    """
    return get_class_from_str("fluke.utils.log", lname)(**kwargs)
