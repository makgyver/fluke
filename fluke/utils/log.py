"""This submodule provides logging utilities."""

import logging
import os
import sys
from typing import Any, Collection, Literal, Union

import json
import clearml
import wandb
from psutil import Process
from rich import print as rich_print
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import Pretty
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
sys.path.append("..")

from .. import DDict  # NOQA
from ..comm import ChannelObserver, Message  # NOQA
from ..evaluation import PerformanceTracker  # NOQA
from ..utils import bytes2human, get_class_from_qualified_name  # NOQA
from . import ClientObserver, ServerObserver, get_class_from_str  # NOQA

__all__ = ["Log", "DebugLog", "TensorboardLog", "WandBLog", "ClearMLLog", "get_logger"]


class Log(ServerObserver, ChannelObserver, ClientObserver):
    """Basic logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process. The logging happens in the console.

    Attributes:
        tracker (PerformanceTracker): The performance tracker.
        current_round (int): The current round.
    """

    def __init__(self, **kwargs):
        self.tracker = PerformanceTracker()
        self.current_round: int = 0
        self.custom_fields: dict = {}

    def log(self, message: str) -> None:
        """Log a message.

        Args:
            message (str): The message to log.
        """
        rich_print(message)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        """Add a scalar to the logger.

        Args:
            key (Any): The key of the scalar.
            value (float): The value of the scalar.
            round (int): The round.
        """
        if round not in self.custom_fields:
            self.custom_fields[round] = {}
        self.custom_fields[round][key] = value

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        """Add scalars to the logger.

        Args:
            key (Any): The main key of the scalars.
            values (dict[str, float]): The key-value pairs of the scalars.
            round (int): The round.
        """
        if round not in self.custom_fields:
            self.custom_fields[round] = {}
        for k, v in values.items():
            self.custom_fields[round][f"{key}/{k}"] = v

    def pretty_log(self, data: Any, title: str) -> None:
        """Log a pretty-printed data.

        Args:
            data (Any): The data to log.
            title (str): The title of the data.
        """
        rich_print(Panel(Pretty(data, expand_all=True), title=title, width=100))

    def init(self, **kwargs) -> None:
        """Initialize the logger.
        The initialization is done by printing the configuration in the console.

        Args:
            **kwargs: The configuration.
        """
        if kwargs:
            rich_print(Panel(Pretty(kwargs, expand_all=True), title="Configuration", width=100))

    def start_round(self, round: int, global_model: Module) -> None:
        self.tracker.add(perf_type="comm", metrics=0, round=round)
        self.current_round = round

        if round == 1 and self.tracker.get("comm", round=0) > 0:
            rich_print(
                Panel(
                    Pretty({"comm_costs": self.tracker.get("comm", round=0)}),
                    title=f"Round: {round - 1}",
                    width=100,
                )
            )

    def end_round(self, round: int) -> None:
        stats = {
            "pre-fit": self.tracker.summary("pre-fit", round=round, include_round=False),
            "locals": self.tracker.summary("locals", round=round, include_round=False),
            "post-fit": self.tracker.summary("post-fit", round=round, include_round=False),
            "global": self.tracker.summary("global", round=round),
            "comm_cost": self.tracker.summary("comm", round=round),
        }

        proc = Process(os.getpid())

        self.tracker.add(
            perf_type="mem",
            metrics=proc.memory_full_info().uss,
            round=round,
        )

        if self.custom_fields and round in self.custom_fields:
            stats.update(self.custom_fields[round])

        to_skip = [k for k, v in stats.items() if v is None or (isinstance(v, dict) and not v)]
        stats = DDict(stats).exclude(*to_skip)

        rich_print(Panel(Pretty(stats, expand_all=True), title=f"Round: {round}", width=100))
        rich_print(
            f"  Memory usage: {bytes2human(self.tracker.get('mem', round=round))} "
            + f"[{proc.memory_percent():.2f} %]"
        )

    def client_evaluation(
        self,
        round: int,
        client_id: int,
        phase: Literal["pre-fit", "post-fit"],
        evals: dict[str, float],
        **kwargs,
    ) -> None:
        if round == -1:
            round = self.current_round + 1
        self.tracker.add(perf_type=phase, metrics=evals, round=round, client_id=client_id)

    def server_evaluation(
        self,
        round: int,
        eval_type: Literal["global", "locals"],
        evals: Union[dict[str, float], dict[int, dict[str, float]]],
        **kwargs,
    ) -> None:

        self.tracker.add(perf_type=eval_type, metrics=evals, round=round)

    def message_received(self, by: Any, message: Message) -> None:
        """Update the communication costs.

        Args:
            by (Any): The sender of the message.
            message (Message): The message received.
        """
        self.tracker.add(perf_type="comm", metrics=message.size, round=self.current_round)

    def finished(self, round: int) -> None:
        stats = {
            "pre-fit": self.tracker.summary("pre-fit", round=round),
            "locals": self.tracker.summary("locals", round=round),
            "post-fit": self.tracker.summary("post-fit", round=round),
            "global": self.tracker.summary("global", round=round),
            "comm_costs": self.tracker.summary("comm", round),
        }

        to_skip = [k for k, v in stats.items() if v is None or (isinstance(v, dict) and not v)]
        stats = DDict(stats).exclude(*to_skip)

        rich_print(
            Panel(
                Pretty(stats, expand_all=True),
                title="Overall Performance",
                width=100,
            )
        )

    def interrupted(self) -> None:
        rich_print("\n[bold italic yellow]The experiment has been interrupted by the user.")

    def early_stop(self, round: int) -> None:
        return self.end_round(round)

    def track_item(self, round: int, item: str, value: Any) -> None:
        self.add_scalar(item, value, round)

    def save(self, path: str) -> None:
        """Save the logger's history to a JSON file.

        Args:
            path (str): The path to the JSON file.
        """
        json_to_save = {
            "perf_global": self.tracker["global"],
            "comm_costs": self.tracker["comm"],
            "perf_locals": self.tracker["locals"],
            "perf_prefit": self.tracker["pre-fit"],
            "perf_postfit": self.tracker["post-fit"],
            "custom_fields": self.custom_fields,
        }
        with open(path, "w") as f:
            json.dump(json_to_save, f, indent=4)

    def close(self) -> None:
        """Close the logger."""
        pass


class DebugLog(Log):
    """Debug Logger.
    This type of logger extends the basic logger by adding debug information.
    It can be seen as a more verbose version of the basic logger.

    Args:
        **kwargs: The configuration for the logger.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            # datefmt="[%X]",
            datefmt="[%Y-%m-%d %H:%M:%S]",
            handlers=[RichHandler(rich_tracebacks=False, show_path=False, markup=True)],
        )

        self.logger = logging.getLogger("rich")

        # Example logs
        # logger.debug("This is a [bold blue]debug[/] message.")
        # logger.info("This is a [green]info[/] message.")
        # logger.warning("This is a [yellow]warning[/] message.")
        # logger.error("This is a [red]error[/] message.")
        # logger.critical("This is a [bold red]critical[/] message.")

    def init(self, **kwargs) -> None:
        self.logger.debug("Debug logging enabled")
        super().init(**kwargs)

    def start_round(self, round: int, global_model: Module) -> None:
        self.logger.debug("Starting round %d", round)
        super().start_round(round, global_model)

    def end_round(self, round: int) -> None:
        self.logger.debug("Ending round %d", round)
        super().end_round(round)

    def selected_clients(self, round: int, clients: Collection) -> None:
        clients_idx = [client.index for client in clients]
        self.logger.debug(f"Selected {len(clients_idx)} clients for round {round}: {clients_idx}")
        super().selected_clients(round, clients)

    def server_evaluation(
        self,
        round: int,
        eval_type: Literal["global", "locals"],
        evals: Union[dict[str, float], dict[int, dict[str, float]]],
        **kwargs,
    ) -> None:
        if eval_type == "global":
            self.logger.debug(f"Global evaluation for round {round}")
        elif eval_type == "locals":
            self.logger.debug(f"Local models evaluated on server's test set for round {round}")
        super().server_evaluation(round, eval_type, evals, **kwargs)

    def finished(self, round: int) -> None:
        self.logger.debug(f"Federation ended successfully after round {round - 1}")
        return super().finished(round)

    def interrupted(self) -> None:
        self.logger.debug("Experiment interrupted by the user")
        return super().interrupted()

    def early_stop(self, round: int) -> None:
        self.logger.debug(f"Early stopping fired for round {round}")
        return super().early_stop(round)

    def start_fit(self, round: int, client_id: int, model: Module, **kwargs) -> None:
        self.logger.debug(f"Starting fit for client {client_id}")
        return super().start_fit(round, client_id, model, **kwargs)

    def end_fit(self, round: int, client_id: int, model: Module, loss: float, **kwargs) -> None:
        self.logger.debug(f"Fit for Client[{client_id}] ended with loss {loss}")
        return super().end_fit(round, client_id, model, loss, **kwargs)

    def client_evaluation(
        self,
        round: int,
        client_id: int,
        phase: Literal["pre-fit", "post-fit"],
        evals: dict[str, float],
        **kwargs,
    ) -> None:
        if round >= 0:
            self.logger.debug(f"Client[{client_id}] {phase} evaluation for round {round}")
        else:
            self.logger.debug(f"Client[{client_id}] {phase} evaluation after final round")
        self.logger.debug(evals)
        return super().client_evaluation(round, client_id, phase, evals, **kwargs)

    def message_received(self, by: Any, message: Message) -> None:
        sender = str(message.sender).split("(")[0]
        receiver = str(by).split("(")[0]
        self.logger.debug(f"Message {message.id} ({sender} -> {receiver}) received")
        return super().message_received(by, message)

    def message_sent(self, to: Any, message: Message) -> None:
        sender = str(message.sender).split("(")[0]
        receiver = str(to).split("(")[0]
        self.logger.debug(f"Message {message.id} ({sender} -> {receiver}) sent")
        return super().message_sent(to, message)

    def message_broadcasted(self, to: list[Any], message: Message) -> None:
        sender = str(message.sender).split("(")[0]
        self.logger.debug(f"Message {message.id} from {sender} broadcasted ")
        return super().message_broadcasted(to, message)


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
        self._config: dict = DDict(**config)
        if "log_dir" not in self._config:
            self._config["log_dir"] = "./logs"
        self._writer: SummaryWriter | None = None

    def init(self, **kwargs) -> None:
        exp_name = self._config.name
        self._config["log_dir"] = os.path.join(self._config["log_dir"], f"{exp_name}")
        self._writer = SummaryWriter(**self._config.exclude("name"))

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        super().add_scalar(key, value, round)
        return self._writer.add_scalar(key, value, round)

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        super().add_scalars(key, values, round)
        return self._writer.add_scalars(key, values, round)

    def start_round(self, round: int, global_model: Module) -> None:
        super().start_round(round, global_model)
        if round == 1 and self.tracker.get("comm", round) > 0:
            self._writer.add_scalar("comm_costs", self.tracker.get("comm", round), round)
        self._writer.flush()

    def _report(self, prefix: str, evals: dict[str, float], round: int) -> None:
        for metric, value in evals.items():
            self._writer.add_scalar(f"{prefix}/{metric}", value, round)
            self._writer.flush()

    def end_round(self, round: int) -> None:
        super().end_round(round)
        global_perf = self.tracker.get("global", round)
        if global_perf is not None:
            self._report("global", global_perf, round)
        self._writer.add_scalar("comm_costs", self.tracker.get("comm", round), round)
        self._writer.flush()

        prefit_perf = self.tracker.summary("pre-fit", round=round, include_round=True)
        if prefit_perf:
            self._report("pre-fit", prefit_perf, prefit_perf["round"])

        postfit_perf = self.tracker.summary("post-fit", round=round, include_round=True)
        if postfit_perf:
            self._report("post-fit", postfit_perf, postfit_perf["round"])

        locals_perf = self.tracker.summary("locals", round=round, include_round=True)
        if locals_perf:
            self._report("locals", locals_perf, locals_perf["round"])

        self._writer.flush()

    def finished(self, round: int) -> None:
        super().finished(round)
        prefit_perf = self.tracker.summary("pre-fit", round=round, include_round=True)
        if prefit_perf and round == prefit_perf["round"]:
            self._report("pre-fit", prefit_perf, round)

        locals_perf = self.tracker.summary("locals", round=round, include_round=True)
        if locals_perf and round == locals_perf["round"]:
            self._report("locals", locals_perf, round)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


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
        self.run = None

    def init(self, **kwargs) -> None:
        super().init(**kwargs)
        self.config["config"] = kwargs
        self.run = wandb.init(**self.config)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        super().add_scalar(key, value, round)
        return self.run.log({key: value}, step=round)

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        super().add_scalars(key, values, round)
        return self.run.log({f"{key}/{k}": v for k, v in values.items()}, step=round)

    def start_round(self, round: int, global_model: Module) -> None:
        super().start_round(round, global_model)
        if round == 1 and self.tracker.get("comm", round) > 0:
            self.run.log({"comm_costs": self.tracker.get("comm", round)})

    def end_round(self, round: int) -> None:
        super().end_round(round)
        global_perf = self.tracker.get("global", round)
        if global_perf is not None:
            self.run.log({"global": global_perf}, step=round)
        self.run.log({"comm_cost": self.tracker.get("comm", round)}, step=round)

        prefit_perf = self.tracker.summary("pre-fit", round=round, include_round=True)
        if prefit_perf:
            self.run.log({"pre-fit": prefit_perf}, step=prefit_perf["round"])

        postfit_perf = self.tracker.summary("post-fit", round=round, include_round=True)
        if postfit_perf:
            self.run.log({"post-fit": postfit_perf}, step=postfit_perf["round"])

        locals_perf = self.tracker.summary("locals", round=round, include_round=True)
        if locals_perf:
            self.run.log({"locals": locals_perf}, step=locals_perf["round"])

    def finished(self, round: int) -> None:
        super().finished(round)

        server_last_round = max(self.tracker["global"])
        self.run.log(
            {"global": self.tracker.get("global", server_last_round)}, step=server_last_round
        )

        postfit_eval_summary = self.tracker.summary("post-fit", round=round, include_round=True)
        if postfit_eval_summary:
            last_round = postfit_eval_summary["round"]
            # avoid warning
            if server_last_round <= last_round:
                self.run.log({"post-fit": postfit_eval_summary}, step=last_round)

        prefit_eval_summary = self.tracker.summary("pre-fit", round=round, include_round=True)
        if prefit_eval_summary:
            last_round = prefit_eval_summary["round"]
            # avoid warning
            if server_last_round <= last_round:
                self.run.log({"pre-fit": prefit_eval_summary}, step=last_round)

        locals_eval_summary = self.tracker.summary("locals", round=round, include_round=True)
        if locals_eval_summary:
            last_round = locals_eval_summary["round"]
            # avoid warning
            if server_last_round <= last_round:
                self.run.log({"locals": locals_eval_summary}, step=last_round)

    def close(self) -> None:
        self.run.finish()

    def save(self, path: str) -> None:
        super().save(path)


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
        super().__init__(name=config["name"])
        self.config: DDict = DDict(**config)
        self.task: clearml.task.Task | None = None

    def init(self, **kwargs) -> None:
        super().init(**kwargs)
        # imported here to avoid issues with requests
        from clearml import Task

        self.task = Task.init(task_name=self.config.name, **self.config.exclude("name"))
        self.task.connect(kwargs)

    def close(self) -> None:
        super().close()
        if self.task is not None:
            self.task.close()
            self.task = None


def get_logger(lname: str, **kwargs) -> Log:
    """Get a logger from its name.
    This function is used to get a logger from its name. It is used to dynamically import loggers.
    The supported loggers are the ones defined in the ``fluke.utils.log`` module, but it can handle
    any logger defined by the user.

    Note:
        To use a custom logger, it must be defined in a module and the full model name must be
        provided in the configuration file. For example, if the logger ``MyLogger`` is defined in
        the module ``my_module`` (i.e., a file called ``my_module.py``), the logger name must be
        ``my_module.MyLogger``. The other logger's parameters must be passed as in the following
        example:

        .. code-block:: yaml

            logger:
                name: my_module.MyLogger
                param1: value1
                param2: value2
                ...


    Args:
        lname (str): The name of the logger.
        **kwargs: The keyword arguments to pass to the logger's constructor.

    Returns:
        Log | DebugLog | WandBLog | ClearMLLog | TensorboardLog: The logger.
    """
    if "." in lname and not lname.startswith("fluke.utils.log"):
        return get_class_from_qualified_name(lname)(**kwargs)
    return get_class_from_str("fluke.utils.log", lname)(**kwargs)
