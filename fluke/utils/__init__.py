"""This module contains utility functions and classes used in ``fluke``."""
from __future__ import annotations
from rich.pretty import Pretty
from rich.panel import Panel
import rich
from torch.optim import Optimizer
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
import torch
from typing import Any, Sequence
# from enum import Enum
import psutil
import pandas as pd
import numpy as np
import importlib
import inspect
import wandb
import json
import yaml
import os
import sys
sys.path.append(".")
sys.path.append("..")


from .. import DDict  # NOQA
from ..comm import ChannelObserver, Message  # NOQA
# from ..data.datasets import DatasetsEnum  # NOQA


__all__ = [
    'model',
    'Configuration',
    'Log',
    'OptimizerConfigurator',
    'ServerObserver',
    'WandBLog',
    'clear_cache',
    'get_class_from_str',
    'get_class_from_qualified_name',
    'get_full_classname',
    'get_loss',
    'get_model',
    'get_scheduler',
    'import_module_from_str'
]


class ServerObserver():
    """Server observer interface.
    This interface is used to observe the server during the federated learning process.
    For example, it can be used to log the performance of the global model and the communication
    costs, as it is done in the ``Log`` class.
    """

    def start_round(self, round: int, global_model: Any):
        """This method is called when a new round starts.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
        """
        pass

    def end_round(self,
                  round: int,
                  evals: dict[str, float],
                  client_evals: Sequence[Any]):
        """This method is called when a round ends.

        Args:
            round (int): The round number.
            evals (dict[str, float]): The evaluation results of the global model.
            client_evals (Sequence[Any]): The evaluation rstuls of the clients.
        """
        pass

    def selected_clients(self, round: int, clients: Sequence):
        """This method is called when the clients have been selected for the current round.

        Args:
            round (int): The round number.
            clients (Sequence): The clients selected for the current round.
        """
        pass

    def error(self, error: str):
        """This method is called when an error occurs.

        Args:
            error (str): The error message.
        """
        pass

    def finished(self,  client_evals: Sequence[Any]):
        """This method is called when the federated learning process has ended.

        Args:
            client_evals (Sequence[Any]): The evaluation metrics of the clients.
        """
        pass


class OptimizerConfigurator:
    """This class is used to configure the optimizer and the learning rate scheduler.

    Attributes:
        optimizer (type[Optimizer]): The optimizer class.
        scheduler (type[LRScheduler]): The learning rate scheduler class.
        optimizer_cfg (DDict): The optimizer keyword arguments.
        scheduler_cfg (DDict): The scheduler keyword arguments.
    """

    def __init__(self,
                 optimizer_cfg: DDict | dict,
                 scheduler_cfg: DDict | dict = None):
        """Initialize the optimizer configurator. In both the ``optimizer_cfg`` and the
        ``scheduler_cfg`` dictionaries, the key "name" is used to specify the optimizer and the
        scheduler, respectively. If not present, the default optimizer is the ``SGD`` optimizer, and
        the default scheduler is the ``StepLR`` scheduler. The other keys are the keyword arguments
        for the optimizer and the scheduler, respectively.

        Note:
            In the key "name" of the ``optimizer_cfg`` dictionary, you can specify the optimizer
            class directly, instead of its string name. Same for the scheduler.

        Args:
            optimizer_cfg (dict, DDict): The optimizer class.
            scheduler_cfg (dict or DDict, optional): The scheduler keyword arguments. If not
                specified, the default scheduler is the ``StepLR`` scheduler with a step size of 1
                and a gamma of 1. Defaults to ``None``.
        """

        if isinstance(optimizer_cfg, DDict):
            self.optimizer_cfg = optimizer_cfg
        elif isinstance(optimizer_cfg, dict):
            self.optimizer_cfg = DDict(**optimizer_cfg)
        else:
            raise ValueError("Invalid optimizer configuration.")

        if scheduler_cfg is None:
            self.scheduler_cfg = DDict(name="StepLR", step_size=1, gamma=1)
        elif isinstance(scheduler_cfg, DDict):
            self.scheduler_cfg = scheduler_cfg
        elif isinstance(scheduler_cfg, dict):
            self.scheduler_cfg = DDict(**scheduler_cfg)
        else:
            raise ValueError("Invalid scheduler configuration.")

        if isinstance(self.optimizer_cfg.name, str):
            self.optimizer: type[Optimizer] = get_optimizer(self.optimizer_cfg.name)
        elif inspect.isclass(self.optimizer_cfg.name) and \
                issubclass(self.optimizer_cfg.name, Optimizer):
            self.optimizer = self.optimizer_cfg.name
        else:
            raise ValueError("Invalid optimizer name. Must be a string or an optimizer class.")

        if "name" not in self.scheduler_cfg:
            self.scheduler = get_scheduler("StepLR")
        elif isinstance(self.scheduler_cfg.name, str):
            self.scheduler: type[LRScheduler] = get_scheduler(self.scheduler_cfg.name)
        elif inspect.isclass(self.scheduler_cfg.name) and \
                issubclass(self.scheduler_cfg.name, LRScheduler):
            self.scheduler = self.scheduler_cfg.name
        else:
            raise ValueError("Invalid scheduler name. Must be a string or a scheduler class.")

        self.scheduler_cfg = self.scheduler_cfg.exclude("name")
        self.optimizer_cfg = self.optimizer_cfg.exclude("name")

    def __call__(self, model: Module, **override_kwargs):
        """Creates the optimizer and the scheduler.

        Args:
            model (Module): The model whose parameters will be optimized.
            override_kwargs (dict): The optimizer's keyword arguments to override the default ones.

        Returns:
            tuple[Optimizer, StepLR]: The optimizer and the scheduler.
        """
        if override_kwargs:
            self.optimizer_cfg.update(**override_kwargs)
        optimizer = self.optimizer(model.parameters(), **self.optimizer_cfg)
        scheduler = self.scheduler(optimizer, **self.scheduler_cfg)
        return optimizer, scheduler

    def __str__(self) -> str:
        strsched = self.scheduler.__name__
        to_str = f"OptCfg({self.optimizer.__name__},"
        to_str += ",".join([f"{k}={v}" for k, v in self.optimizer_cfg.items()])
        to_str += f",{strsched}(" + ",".join([f"{k}={v}" for k, v in self.scheduler_cfg.items()])
        to_str += "))"
        return to_str

    def __repr__(self) -> str:
        return str(self)


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
    The supported loss functions are the ones defined in the ``torch.nn`` module.

    Args:
        lname (str): The name of the loss function.

    Returns:
        Module: The loss function.
    """
    return get_class_from_str("torch.nn", lname)()


def get_model(mname: str, **kwargs) -> Module:
    """Get a model from its name.
    This function is used to get a torch model from its name and the name of the module where it is
    defined. It is used to dynamically import models. If ``mname`` is not a fully qualified name,
    the model is assumed to be defined in the ``fluke.nets`` module.

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

    Example:
        Let ``A`` be a class defined in the module ``fluke.utils``

        .. code-block:: python
            :linenos:

            # This is the content of the file fluke/utils.py
            class A:
                pass

            get_full_classname(A) # 'fluke.utils.A'

        If the class is defined in the ``__main__`` module, then:

        .. code-block:: python
            :linenos:

            if __name__ == "__main__":
                class B:
                    pass

                get_full_classname(B) # '__main__.B'
    """
    return f"{classtype.__module__}.{classtype.__name__}"


def get_logger(lname: str, **kwargs) -> Log | WandBLog:
    """Get a logger from its name.
    This function is used to get a logger from its name. It is used to dynamically import loggers.
    The supported loggers are the ones defined in the ``fluke.utils`` module.

    Args:
        lname (str): The name of the logger.
        **kwargs: The keyword arguments to pass to the logger's constructor.

    Returns:
        Log | WandBLog: The logger.
    """
    return get_class_from_str("fluke.utils", lname)(**kwargs)


def get_optimizer(oname: str) -> type[Optimizer]:
    """Get an optimizer from its name.
    This function is used to get an optimizer from its name. It is used to dynamically import
    optimizers. The supported optimizers are the ones defined in the ``torch.optim`` module.

    Args:
        oname (str): The name of the optimizer.

    Returns:
        type[Optimizer]: The optimizer class.
    """
    return get_class_from_str("torch.optim", oname)


def get_scheduler(sname: str) -> type[LRScheduler]:
    """Get a learning rate scheduler from its name.
    This function is used to get a learning rate scheduler from its name. It is used to dynamically
    import learning rate schedulers. The supported schedulers are the ones defined in the
    ``torch.optim.lr_scheduler`` module.

    Args:
        sname (str): The name of the scheduler.

    Returns:
        torch.nn.Module: The learning rate scheduler.
    """
    return get_class_from_str("torch.optim.lr_scheduler", sname)


def clear_cache(ipc: bool = False):
    """Clear the CUDA cache. This function should be used to free the GPU memory after the training
    process has ended. It is usually used after the local training of the clients.

    Args:
        ipc (bool, optional): Whether to force collecting GPU memory after it has been released by
            CUDA IPC.
    """
    torch.cuda.empty_cache()
    if ipc and torch.cuda.is_available():
        torch.cuda.ipc_collect()


class Configuration(DDict):
    """Fluke configuration class.
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
            "device": "cpu",
            "seed": 42
        }

        LOG_OPT_KEYS = {
            "name": "Log"
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
            "keep_test": True,
            "server_test": True,
            "server_split": 0.0,
            "uniform_test": False
        }

        ALG_1L_REQUIRED_KEYS = ["name", "hyperparameters"]
        HP_REQUIRED_KEYS = ["model", "client", "server"]
        CLIENT_HP_REQUIRED_KEYS = ["loss", "batch_size", "local_epochs", "optimizer"]
        WANDB_REQUIRED_KEYS = ["project", "entity"]

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
                    rich.print(f"Error: {k} is required for key 'logger' when using 'WandBLog'.")
                    error = True

        # if not error:
        #     self.data.dataset.name = DatasetsEnum(self.data.dataset.name)
            # self.exp.device = DeviceEnum(self.exp.device) if self.exp.device else DeviceEnum.CPU
            # self.logger.name = LogEnum(self.logger.name)

        if error:
            raise ValueError("Configuration validation failed.")

    def __str__(self) -> str:
        return f"{self.method.name}_data({self.data.dataset.name}, " + \
               f"{self.data.distribution.name}" + \
               f"{',std' if self.data.standardize else ''})" + \
               f"_proto(C{self.protocol.n_clients}, R{self.protocol.n_rounds}," + \
               f"E{self.protocol.eligible_perc})" + \
               f"_seed({self.exp.seed})"

    def __repr__(self) -> str:
        return str(self)
