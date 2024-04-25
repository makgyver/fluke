"""This module contains utility functions and classes used throughout the package."""
from __future__ import annotations
from rich.pretty import Pretty
from rich.panel import Panel
import rich
from torch.optim import Optimizer
from torch.nn import Module
import torch
from typing import Any, Dict, Sequence
from enum import Enum
import psutil
import pandas as pd
import numpy as np
import importlib
import wandb
import json
import yaml
import os
import sys
sys.path.append(".")
sys.path.append("..")


from .. import DDict  # NOQA
from ..comm import ChannelObserver, Message  # NOQA
from ..server import ServerObserver  # NOQA
from ..data.datasets import DatasetsEnum  # NOQA
from ..data import DistributionEnum  # NOQA


__all__ = [
    'model',
    'OptimizerConfigurator',
    'LogEnum',
    'Log',
    'WandBLog',
    'Configuration',
    'import_module_from_str',
    'get_class_from_str',
    'get_loss',
    'get_model',
    'get_scheduler',
    'clear_cache',
    'get_full_classname'
]


class OptimizerConfigurator:
    """Optimizer configurator.

    This class is used to configure the optimizer and the learning rate scheduler.
    To date, only the `StepLR` scheduler is supported.

    Attributes:
        optimizer (type[Optimizer]): The optimizer class.
        scheduler_kwargs (DDict): The scheduler keyword arguments.
        optimizer_kwargs (DDict): The optimizer keyword arguments.
    """

    def __init__(self,
                 optimizer_class: type[Optimizer],
                 scheduler_kwargs: dict = None,
                 **optimizer_kwargs):
        self.optimizer: type[Optimizer] = optimizer_class
        self.scheduler_kwargs: DDict = (DDict(**scheduler_kwargs) if scheduler_kwargs is not None
                                        else DDict(name="StepLR", step_size=1, gamma=1))
        self.scheduler = get_scheduler(self.scheduler_kwargs.name)
        self.optimizer_kwargs: DDict = DDict(**optimizer_kwargs)

    def __call__(self, model: Module, **override_kwargs):
        """Creates the optimizer and the scheduler.

        Args:
            model (Module): The model whose parameters will be optimized.
            override_kwargs (dict): The optimizer's keyword arguments to override the default ones.

        Returns:
            tuple[Optimizer, StepLR]: The optimizer and the scheduler.
        """
        if override_kwargs:
            self.optimizer_kwargs.update(override_kwargs)
        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs.exclude("name"))
        return optimizer, scheduler

    def __str__(self) -> str:
        strsched = self.scheduler.__name__
        sch_params = self.scheduler_kwargs.exclude("name")
        to_str = f"OptCfg({self.optimizer.__name__},"
        to_str += ",".join([f"{k}={v}" for k, v in self.optimizer_kwargs.items()])
        to_str += f",{strsched}(" + ",".join([f"{k}={v}" for k, v in sch_params.items()])
        to_str += "))"
        return to_str


class LogEnum(Enum):
    """Log enumerator."""
    LOCAL = "local"  # : Local logging
    WANDB = "wandb"  # : Weights and Biases logging

    def logger(self,
               **wandb_config):
        """Returns a new logger according to the value of the enumerator.

        Args:
            **wandb_config (dict): The configuration for Weights and Biases.

        Returns:
            Log: The logger.
        """
        if self == LogEnum.LOCAL:
            return Log()
        else:
            return WandBLog(**wandb_config)


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

    def __init__(self):
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
                  global_eval: Dict[str, float],
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


class WandBLog(Log):
    """Weights and Biases logger.

    This class is used to log the performance of the global model and the communication costs during
    the federated learning process on Weights and Biases.

    See Also:
        For more information on Weights and Biases, see the `Weights and Biases documentation
        <https://docs.wandb.ai/>`_.

    Args:
        evaluator (Evaluator): The evaluator that will be used to evaluate the model.
        eval_every (int): The number of rounds between evaluations.
        **config: The configuration for Weights and Biases.
    """

    def __init__(self, **config):
        super().__init__()
        self.config = config

    def init(self, **kwargs):
        super().init(**kwargs)
        self.config["config"] = kwargs
        self.run = wandb.init(**self.config)

    def start_round(self, round: int, global_model: Module):
        super().start_round(round, global_model)
        if round == 1 and self.comm_costs[0] > 0:
            self.run.log({"comm_costs": self.comm_costs[0]})

    def end_round(self, round: int, global_eval: Dict[str, float], client_evals: Sequence[Any]):
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


def import_module_from_str(name: str) -> Any:
    """Import a module from its name.

    Args:
        name (str): The name of the module.

    Returns:
        Any: The module.
    """
    components = name.split('.')
    mod = importlib.import_module(".".join(components[:-1]))
    mod = getattr(mod, components[-1])
    return mod


def get_class_from_str(module_name: str, class_name: str) -> Any:
    """Get a class from its name.

    This function is used to get a class from its name and the name of the module where it is
    defined. It is used to dynamically import classes.

    Args:
        module_name (str): The name of the module where the class is defined.
        class_name (str): The name of the class.

    Returns:
        Any: The class.
    """
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_class_from_qualified_name(qualname: str) -> Any:
    """Get a class from its fully qualified name.

    Args:
        qualname (str): The fully qualified name of the class.

    Returns:
        Any: The class.
    """
    module_name = ".".join(qualname.split(".")[:-1])
    class_name = qualname.split(".")[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_loss(lname: str) -> Module:
    """Get a loss function from its name.

    The supported loss functions are the ones defined in the `torch.nn` module.

    Args:
        lname (str): The name of the loss function.

    Returns:
        Module: The loss function.
    """
    return get_class_from_str("torch.nn", lname)()


def get_model(mname: str, **kwargs) -> Module:
    """Get a model from its name.

    This function is used to get a model from its name and the name of the module where it is
    defined. It is used to dynamically import models.

    Args:
        mname (str): The name of the model.
        **kwargs: The keyword arguments to pass to the model's constructor.

    Returns:
        Module: The model.
    """
    if "." in mname:
        module_name = ".".join(mname.split(".")[:-1])
        mname = mname.split(".")[-1]
    else:
        module_name: str = "fluke.nets"
    return get_class_from_str(module_name, mname)(**kwargs)


def get_full_classname(classtype: type) -> str:
    """Get the fully qualified name of a class.

    Args:
        classtype (type): The class.

    Returns:
        str: The fully qualified name of the class.
    """
    return f"{classtype.__module__}.{classtype.__name__}"


def get_scheduler(sname: str) -> torch.nn.Module:
    """Get a learning rate scheduler from its name.

    This function is used to get a learning rate scheduler from its name. It is used to dynamically
    import learning rate schedulers. The supported schedulers are the ones defined in the
    `torch.optim.lr_scheduler` module.

    Args:
        sname (str): The name of the scheduler.

    Returns:
        torch.nn.Module: The learning rate scheduler.
    """
    return get_class_from_str("torch.optim.lr_scheduler", sname)


def clear_cache(ipc: bool = False):
    """Clear the CUDA cache.

    Args:
        ipc (bool, optional): Whether to force collecting GPU memory after it has been released by
            CUDA IPC.
    """
    torch.cuda.empty_cache()
    if ipc and torch.cuda.is_available():
        torch.cuda.ipc_collect()


class Configuration(DDict):
    """FL-Bench configuration class.

    This class is used to store the configuration of an experiment. The configuration must adhere to
    a specific structure. The configuration is validated when the class is instantiated.

    Args:
        config_exp_path (str): The path to the experiment configuration file.
        config_alg_path (str): The path to the algorithm configuration file.

    Raises:
        ValueError: If the configuration is not valid.
    """

    def __init__(self, config_exp_path: str, config_alg_path: str):
        with open(config_exp_path) as f:
            config_exp = yaml.safe_load(f)
        with open(config_alg_path) as f:
            config_alg = yaml.safe_load(f)

        self.update(**config_exp)
        self.update(method=config_alg)
        self._validate()

    def _validate(self) -> bool:

        EXP_OPT_KEYS = {
            "average": "micro",
            "device": "cpu",
            "seed": 42
        }

        LOG_OPT_KEYS = {
            "name": "local",
            "eval_every": 1
        }

        FIRST_LVL_KEYS = ["data", "protocol", "method"]
        FIRST_LVL_OPT_KEYS = {
            "exp": EXP_OPT_KEYS,
            "logger": LOG_OPT_KEYS
        }
        PROTO_REQUIRED_KEYS = ["n_clients", "eligible_perc", "n_rounds"]
        DATA_REQUIRED_KEYS = ["distribution", "dataset"]
        DATA_OPT_KEYS = {
            "sampling_perc": 1.0,
            "client_split": 0.0,
            "standardize": False
        }

        ALG_1L_REQUIRED_KEYS = ["name", "hyperparameters"]
        HP_REQUIRED_KEYS = ["model", "client", "server"]
        CLIENT_HP_REQUIRED_KEYS = ["loss", "batch_size", "local_epochs", "optimizer"]
        WANDB_REQUIRED_KEYS = ["project", "entity", ]

        error = False
        for k in FIRST_LVL_KEYS:
            if k not in self:
                f = "experiment" if k != "method" else "algorithm"
                rich.print(f"Error: {k} is required in the {f} configuration file")
                error = True

        for k, v in FIRST_LVL_OPT_KEYS.items():
            if k not in self:
                self[k] = DDict(v)

        for k in PROTO_REQUIRED_KEYS:
            if k not in self.protocol:
                rich.print(f"Error: {k} is required for key 'protocol'.")
                error = True

        for k in DATA_REQUIRED_KEYS:
            if k not in self.data:
                rich.print(f"Error: {k} is required for key 'data'.")
                error = True

        for k, v in DATA_OPT_KEYS.items():
            if k not in self.data:
                self.data[k] = v

        for k, v in EXP_OPT_KEYS.items():
            if k not in self.exp:
                self.exp[k] = v

        for k in ALG_1L_REQUIRED_KEYS:
            if k not in self.method:
                rich.print(f"Error: {k} is required for key 'method'.")
                error = True

        for k in HP_REQUIRED_KEYS:
            if k not in self.method.hyperparameters:
                rich.print(f"Error: {k} is required for key 'hyperparameters' in 'method'.")
                error = True

        for k in CLIENT_HP_REQUIRED_KEYS:
            if k not in self.method.hyperparameters.client:
                rich.print(f"Error: {k} is required as hyperparameter of 'client'.")
                error = True

        if 'logger' in self and self.logger.name == "wandb":
            for k in WANDB_REQUIRED_KEYS:
                if k not in WANDB_REQUIRED_KEYS:
                    rich.print(f"Error: {k} is required for key 'logger' when using 'wandb'.")
                    error = True

        if not error:
            self.data.distribution.name = DistributionEnum(self.data.distribution.name)
            self.data.dataset.name = DatasetsEnum(self.data.dataset.name)
            # self.exp.device = DeviceEnum(self.exp.device) if self.exp.device else DeviceEnum.CPU
            self.logger.name = LogEnum(self.logger.name)

        if error:
            raise ValueError("Configuration validation failed.")

    def __str__(self) -> str:
        return f"{self.method.name}_data({self.data.dataset.name.value}, " + \
               f"{self.data.distribution.name.value}" + \
               f"{',std' if self.data.standardize else ''})" + \
               f"_proto(C{self.protocol.n_clients}, R{self.protocol.n_rounds}," + \
               f"E{self.protocol.eligible_perc})" + \
               f"_seed({self.exp.seed})"

    def __repr__(self) -> str:
        return self.__str__()
