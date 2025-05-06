"""This module contains utility functions and classes used in ``fluke``."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import warnings
from itertools import product
from typing import (TYPE_CHECKING, Any, Generator, Iterable, Literal, Optional,
                    Union)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from cerberus import Validator
from rich import print as rich_print
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

sys.path.append(".")
sys.path.append("..")

if TYPE_CHECKING:
    from client import Client  # NOQA
    from server import Server  # NOQA

from .. import DDict, FlukeCache, FlukeENV  # NOQA

__all__ = [
    'log',
    'model',
    'Configuration',
    'ConfigurationError',
    'ClientObserver',
    'OptimizerConfigurator',
    'ServerObserver',
    'bytes2human',
    'cache_obj',
    'clear_cuda_cache',
    'get_class_from_str',
    'get_class_from_qualified_name',
    'get_full_classname',
    'get_loss',
    'get_model',
    'get_optimizer',
    'get_scheduler',
    'flatten_dict',
    'import_module_from_str',
    'memory_usage',
    'plot_distribution',
    'retrieve_obj'
]


class ClientObserver():
    """Client observer interface.
    This interface is used to observe the client during the federated learning process.
    For example, it can be used to log the performance of the local model, as it is done by the
    :class:`fluke.utils.log.Log` class.
    """

    def start_fit(self, round: int, client_id: int, model: Module, **kwargs: dict[str, Any]):
        """This method is called when the client starts the local training process.

        Args:
            round (int): The round number.
            client_id (int):  The client ID.
            model (Module): The local model before training.
            **kwargs (dict): Additional keyword arguments.
        """
        pass

    def end_fit(self,
                round: int,
                client_id: int,
                model: Module,
                loss: float,
                **kwargs: dict[str, Any]):
        """This method is called when the client ends the local training process.

        Args:
            round (int): The round number.
            client_id (int): The client ID.
            model (Module): The local model after the local training.
            loss (float): The average loss incurred by the local model during training.
            **kwargs (dict): Additional keyword arguments.
        """
        pass

    def client_evaluation(self,
                          round: int,
                          client_id: int,
                          phase: Literal["pre-fit", "post-fit"],
                          evals: dict[str, float],
                          **kwargs: dict[str, Any]):
        """This method is called when the client evaluates the local model.
        The evaluation can be done before ('pre-fit') and/or after ('post-fit') the local
        training process. The 'pre-fit' evlauation is usually the evaluation of the global model on
        the local test set, and the 'post-fit' evaluation is the evaluation of the just updated
        local model on the local test set.

        Args:
            round (int): The round number.
            client_id (int): The client ID.
            phase (Literal['pre-fit', 'post-fit']): Whether the evaluation is done before or after
                the local training process.
            evals (dict[str, float]): The evaluation results.
            **kwargs (dict): Additional keyword arguments.
        """
        pass

    def track_item(self,
                   round: int,
                   client_id: int,
                   item: str,
                   value: float,
                   **kwargs: dict[str, Any]) -> None:
        """This method is called when the client aims to log an item.

        Args:
            round (int): The round number.
            client_id (int): The client ID.
            item (str): The name of the log item.
            value (float): The value of the log item.
            **kwargs (dict): Additional keyword arguments.
        """
        pass


class ServerObserver():
    """Server observer interface.
    This interface is used to observe the server during the federated learning process.
    For example, it can be used to log the performance of the global model and the communication
    costs, as it is done by the ``Log`` class.
    """

    def start_round(self, round: int, global_model: Any) -> None:
        """This method is called when a new round starts.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
        """
        pass

    def end_round(self, round: int) -> None:
        """This method is called when a round ends.

        Args:
            round (int): The round number.
        """
        pass

    def selected_clients(self, round: int, clients: Iterable) -> None:
        """This method is called when the clients have been selected for the current round.

        Args:
            round (int): The round number.
            clients (Iterable): The clients selected for the current round.
        """
        pass

    def server_evaluation(self,
                          round: int,
                          type: Literal["global", "locals"],
                          evals: Union[dict[str, float], dict[int, dict[str, float]]],
                          **kwargs: dict[str, Any]) -> None:
        """This method is called when the server evaluates the global or the local models on its
        test set.

        Args:
            round (int): The round number.
            type (Literal['global', 'locals']): The type of evaluation. If 'global', the evaluation
                is done on the global model. If 'locals', the evaluation is done on the local models
                of the clients on the test set of the server.
            evals (dict[str, float] | dict[int, dict[str, float]]): The evaluation metrics. In case
                of 'global' evaluation, it is a dictionary with the evaluation metrics. In case of
                'locals' evaluation, it is a dictionary of dictionaries where the keys are the
                client IDs and the values are the evaluation metrics.
        """
        pass

    def finished(self, round: int) -> None:
        """This method is called when the federated learning process has ended.

        Args:
            round (int): The last round number.
        """
        pass

    def interrupted(self) -> None:
        """This method is called when the federated learning process has been interrupted."""
        pass

    def early_stop(self, round: int) -> None:
        """This method is called when the federated learning process has been stopped due to an
        early stopping criterion.

        Args:
            round (int): The last round number.
        """
        pass

    def track_item(self,
                   round: int,
                   item: str,
                   value: float) -> None:
        """This method is called when the server aims log an item.

        Args:
            round (int): The round number.
            item (str): The name of the log item.
            value (float): The value of the log item.
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

        self.optimizer_cfg: DDict = None
        self.scheduler_cfg: DDict = None
        self.optimizer: type[Optimizer] = None
        self.scheduler: type[LRScheduler] = None

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
            self.optimizer = get_optimizer(self.optimizer_cfg.name)
        elif inspect.isclass(self.optimizer_cfg.name) and \
                issubclass(self.optimizer_cfg.name, Optimizer):
            self.optimizer = self.optimizer_cfg.name
        else:
            raise ValueError("Invalid optimizer name. Must be a string or an optimizer class.")

        if "name" not in self.scheduler_cfg:
            self.scheduler = get_scheduler("StepLR")
        elif isinstance(self.scheduler_cfg.name, str):
            self.scheduler = get_scheduler(self.scheduler_cfg.name)
        elif inspect.isclass(self.scheduler_cfg.name) and \
                issubclass(self.scheduler_cfg.name, LRScheduler):
            self.scheduler = self.scheduler_cfg.name
        else:
            raise ValueError("Invalid scheduler name. Must be a string or a scheduler class.")

        self.scheduler_cfg = self.scheduler_cfg.exclude("name")
        self.optimizer_cfg = self.optimizer_cfg.exclude("name")

    def __call__(self, model: Module, filter_fun: Optional[callable] = None, **override_kwargs):
        """Creates the optimizer and the scheduler.

        Args:
            model (Module): The model whose parameters will be optimized.
            filter_fun (callable): This must be a function of the model and it must returns the set
              of parameters that the optimizer will consider.
            override_kwargs (dict): The optimizer's keyword arguments to override the default ones.

        Returns:
            tuple[Optimizer, StepLR]: The optimizer and the scheduler.
        """
        if override_kwargs:
            self.optimizer_cfg.update(**override_kwargs)

        if filter_fun is None:
            optimizer = self.optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                       **self.optimizer_cfg)
        else:
            optimizer = self.optimizer(filter_fun(model),
                                       **self.optimizer_cfg)
        scheduler = self.scheduler(optimizer, **self.scheduler_cfg)
        return optimizer, scheduler

    def __str__(self, indent: int = 0) -> str:
        strsched = self.scheduler.__name__
        indentstr = " " * (indent + 7)
        to_str = f"OptCfg({self.optimizer.__name__},\n{indentstr}"
        to_str += f",\n{indentstr}".join([f"{k}={v}" for k, v in self.optimizer_cfg.items()])
        to_str += f",\n{indentstr}{strsched}("
        indentstr = indentstr + " " * (len(strsched) + 1)
        to_str += f",\n{indentstr}".join([f"{k}={v}" for k, v in self.scheduler_cfg.items()])
        to_str += "))"
        return to_str

    def __repr__(self, indent: int = 0) -> str:
        return self.__str__(indent=indent)

    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


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


def get_model(mname: str, **kwargs: dict[str, Any]) -> Module:
    """Get a model from its name.
    This function is used to get a torch model from its name and the name of the module where it is
    defined. It is used to dynamically import models. If ``mname`` is not a fully qualified name,
    the model is assumed to be defined in the :mod:`fluke.nets` module.

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


def get_optimizer(oname: str) -> type[Optimizer]:
    """Get an optimizer from its name.
    This function is used to get an optimizer from its name. It is used to dynamically import
    optimizers. The supported optimizers are the ones defined in the :mod:`torch.optim` module.

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
    :mod:`torch.optim.lr_scheduler` module.

    Args:
        sname (str): The name of the scheduler.

    Returns:
        torch.nn.Module: The learning rate scheduler.
    """
    return get_class_from_str("torch.optim.lr_scheduler", sname)


def clear_cuda_cache(ipc: bool = False):
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
        force_validation (bool, optional): Whether to force the validation of the configuration.
            Defaults to ``True``.

    Raises:
        ValueError: If the configuration is not valid.
    """

    def __init__(self,
                 config_exp_path: str = None,
                 config_alg_path: str = None,
                 force_validation: bool = True):

        if config_exp_path is not None and os.path.exists(config_exp_path):
            cfg_exp = OmegaConf.load(config_exp_path)
            self.update(DDict(**cfg_exp))

        if config_alg_path is not None and os.path.exists(config_alg_path):
            cfg_alg = OmegaConf.load(config_alg_path)
            self.update(method=DDict(**cfg_alg))

        if force_validation:
            self._validate()

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> Configuration:
        """Create a configuration from a dictionary.

        Args:
            cfg_dict (dict): The dictionary.

        Returns:
            Configuration: The configuration.
        """
        cfg = Configuration(force_validation=False)
        cfg.update(**cfg_dict)
        print(cfg)
        cfg._validate()
        return cfg

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            dict: The dictionary.
        """
        def _to_dict(ddict: DDict) -> dict:
            if not isinstance(ddict, dict):
                if isinstance(ddict, type):
                    return ddict.__name__
                return ddict
            return {k: _to_dict(v) for k, v in ddict.items()}

        return _to_dict(self)

    @classmethod
    def sweep(cls,
              config_exp_path: str,
              config_alg_path: str) -> Generator[Configuration, None, None]:
        """Generate configurations from a sweep.
        This method is used to generate configurations from a sweep. The sweep is defined by the
        experiment configuration file. The method yields a configuration for each combination of
        hyperparameters.

        Args:
            config_exp_path (str): The path to the experiment configuration file.
            config_alg_path (str): The path to the algorithm configuration file.

        Yields:
            Configuration: A configuration.
        """
        cfgs = Configuration(config_exp_path, config_alg_path, force_validation=False)

        for cfg in Configuration.__sweep(cfgs):
            yield Configuration.from_dict(cfg)

    @staticmethod
    def __sweep(cfgs: DDict) -> Generator[DDict, None, None]:
        """Generate configurations from a sweep.
        This method is used to generate configurations from a sweep. The sweep is defined by the
        experiment configuration file. The method yields a configuration for each combination of
        hyperparameters.

        Args:
            cfgs (DDict): The configuration.

        Yields:
            DDict: A configuration.
        """
        normalized = {
            k: v if isinstance(v, (list, dict, ListConfig)) else [
                v]  # strings of numbers treated as is
            for k, v in cfgs.items()
        }

        for k, v in normalized.items():
            if isinstance(v, dict):
                nested_combos = Configuration.__sweep(v)
                normalized[k] = nested_combos

        keys = normalized.keys()
        values = normalized.values()

        all_combinations = []
        for combo in product(*values):
            combined = {k: v for k, v in zip(keys, combo)}
            all_combinations.append(combined)

        return all_combinations

    @property
    def client(self) -> DDict:
        """Get quick access to the client's hyperparameters.

        Returns:
            DDict: The client's hyperparameters.
        """
        return self.method.hyperparameters.client

    @property
    def server(self) -> DDict:
        """Get quick access to the server's hyperparameters.

        Returns:
            DDict: The server's hyperparameters.
        """
        return self.method.hyperparameters.server

    @property
    def model(self) -> DDict:
        """Get quick access to the model hyperparameters.

        Returns:
            DDict: The model hyperparameters.
        """
        return self.method.hyperparameters.model

    __SCHEMA = {
        "data": {
            "type": "dict",
            "schema": {
                "dataset": {
                    "type": "dict",
                    "required": True,
                    "schema": {
                        "name": {"type": "string", "required": True},
                        "path": {"type": "string", "required": False, "default": "./data"}
                    }
                },
                "distribution": {
                    "type": "dict",
                    "required": True,
                    "schema": {
                        "name": {"type": "string", "required": True}
                    }
                },
                "sampling_perc": {"type": "float", "required": False,
                                  "min": 0.001, "max": 1.0, "default": 1.0},
                "client_split": {"type": "float", "required": False, "min": 0.0, "max": 1.0,
                                 "default": 0.0},
                "keep_test": {"type": "boolean", "required": False, "default": True},
                "server_test": {"type": "boolean", "required": False, "default": True},
                "server_split": {"type": "float", "required": False, "min": 0.0, "max": 1.0,
                                 "default": 0.0},
                "uniform_test": {"type": "boolean", "required": False, "default": False}
            }
        },
        "exp": {
            "type": "dict",
            "schema": {
                "device": {
                    "type": "string",
                    "required": False,
                    "default": "cpu",
                    "anyof": [{"allowed": ["cpu", "cuda", "mps"]},
                              {"regex": "^cuda:[0-9]+$"}]
                },
                "seed": {"type": "integer", "required": False, "default": 42},
                "inmemory": {"type": "boolean", "required": False, "default": True}
            }
        },
        "eval": {
            "type": "dict",
            "schema": {
                "task": {"type": "string", "required": False,
                         "default": "classification", "allowed": ["classification"]},
                "eval_every": {"type": "integer", "required": False, "default": 1, "min": 1},
                "pre_fit": {"type": "boolean", "required": False, "default": False},
                "post_fit": {"type": "boolean", "required": False, "default": True},
                "locals": {"type": "boolean", "required": False, "default": False},
                "server": {"type": "boolean", "required": False, "default": True}
            }
        },
        "logger": {
            "type": "dict",
            "schema": {
                "name": {"type": "string", "required": False, "default": "Log"}
            }
        },
        "protocol": {
            "type": "dict",
            "schema": {
                "eligible_perc": {"type": "float", "required": True, "min": 0.0, "max": 1.0},
                "n_clients": {"type": "integer", "required": True, "min": 1},
                "n_rounds": {"type": "integer", "required": True, "min": 1}
            }
        },
        "method": {
            "type": "dict",
            "schema": {
                "hyperparameters": {
                    "type": "dict",
                    "schema": {
                        "client": {
                            "type": "dict",
                            "schema": {
                                "batch_size": {"type": "integer", "required": True, "min": 0},
                                "local_epochs": {"type": "integer", "required": True, "min": 1},
                                "loss": {"type": "string", "required": True},
                                "optimizer": {
                                    "type": "dict",
                                    "schema": {
                                        "name": {"type": "string", "required": False,
                                                 "default": "SGD"},
                                        "lr": {"type": "float", "required": False, "default": 0.01},
                                    }
                                },
                                "scheduler": {
                                    "type": "dict",
                                    "schema": {
                                        "name": {"type": "string", "required": False,
                                                 "default": "StepLR"}
                                    }
                                }
                            }
                        },
                        "server": {"type": "dict"},
                        "model": {"type": "string", "required": True}
                    }
                },
                "name": {"type": "string", "required": True}
            }
        }
    }

    def __repair_save(self, data: dict) -> None:
        if "save" not in data:
            return {}, []

        save_valid = Validator()
        save_valid.schema = {
            "save_every": {"type": "integer", "default": 1, "min": 1},
            "path": {"type": "string", "default": "./models"},
            "global_only": {"type": "boolean", "default": False}
        }
        save_valid.allow_unknown = False
        valid_result = save_valid.validate(data["save"])
        if not valid_result:
            return None, save_valid.errors

        return save_valid.document, []

    def _validate(self) -> bool:

        validator = Validator()
        validator.schema = self.__SCHEMA
        validator.allow_unknown = True

        cfg_dict = self.to_dict()
        valid_result = validator.validate(cfg_dict)
        save_valid_result, save_errors = self.__repair_save(cfg_dict)

        errors = validator.errors
        if save_errors:
            errors.update(save=save_errors)

        if not valid_result:
            rich_print("[red]Invalid configuration:[/red]")
            rich_print(errors)
            raise ConfigurationError()

        clean_cfg = validator.document
        clean_cfg["save"] = save_valid_result

        self.update(clean_cfg)

    def verbose(self) -> str:
        return super().__str__()


def plot_distribution(clients: list[Client],
                      train: bool = True,
                      type: str = "ball") -> None:
    """Plot the distribution of classes for each client.
    This function is used to plot the distribution of classes for each client. The plot can be a
    scatter plot, a heatmap, or a bar plot. The scatter plot (``type='ball'``) shows filled circles
    whose size is proportional to the number of examples of a class. The heatmap (``type='mat'``)
    shows a matrix where the rows represent the classes and the columns represent the clients with
    a color intensity proportional to the number of examples of a class. The bar plot
    (``type='bar'``) shows a stacked bar plot where the height of the bars is proportional to the
    number of examples of a class.

    Warning:
        If the number of clients is greater than 30, the type is automatically switched to
        ``'bar'`` for better visualization.

    Args:
        clients (list[Client]): The list of clients.
        train (bool, optional): Whether to plot the distribution on the training set. If ``False``,
            the distribution is plotted on the test set. Defaults to ``True``.
        type (str, optional): The type of plot. It can be ``'ball'``, ``'mat'``, or ``'bar'``.
            Defaults to ``'ball'``.
    """
    assert type in ["bar", "ball", "mat"], "Invalid plot type. Must be 'bar', 'ball' or 'mat'."
    if len(clients) > 30 and type != "bar":
        warnings.warn("Too many clients to plot. Switching to 'bar' plot.")
        type = "bar"

    client = {}
    for c in clients:
        client[c.index] = c.train_set.tensors[1] if train else c.test_set.tensors[1]

    # Count the occurrences of each class for each client
    class_counts = {client_idx: torch.bincount(client_data).tolist()
                    for client_idx, client_data in enumerate(client.values())}

    # Maximum number of classes
    num_classes = max(len(counts) for counts in class_counts.values())

    _, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(range(len(client)))

    class_matrix = np.zeros((num_classes, len(client)))
    for client_idx, counts in class_counts.items():
        for class_idx, count in enumerate(counts):
            class_matrix[class_idx, client_idx] = count
            # Adjusting size based on the count
            if type == "ball":
                size = count * 1  # Adjust the scaling factor as needed
                ax.scatter(client_idx, class_idx, s=size, alpha=0.6)
                ax.set_yticks(range(num_classes))
                ax.text(client_idx, class_idx, str(count), va='center',
                        ha='center', color='black', fontsize=9)
    plt.title('Number of Examples per Class for Each Client', fontsize=12)
    ax.grid(False)
    if type == "mat":
        ax.set_yticks(range(num_classes))
        sns.heatmap(class_matrix, ax=ax, cmap="viridis", annot=class_matrix, fmt='g',
                    cbar=False, annot_kws={"fontsize": 6})
    elif type == "bar":
        df = pd.DataFrame(class_matrix.T, index=[f'{i}' for i in range(len(client))],
                          columns=[f'{i}' for i in range(num_classes)])
        step = int(max(1, np.ceil(len(clients) / 50)))
        df.plot(ax=ax,
                kind='bar',
                stacked=True,
                xticks=[x for x in range(0, len(clients), step)],
                color=sns.color_palette('viridis', num_classes)).legend(loc='upper right')
    plt.xlabel('clients')
    plt.ylabel('classes')
    plt.show()


class ConfigurationError(Exception):
    """Exception raised when the configuration is not valid."""
    pass


def bytes2human(n: int) -> str:
    """Convert bytes to human-readable format.

    See:
        https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

    Args:
        n (int): The number of bytes.

    Returns:
        str: The number of bytes in human-readable format.

    Example:
        .. code-block:: python
            :linenos:

            >>> bytes2human(10000)
            '9.8K'
            >>> bytes2human(100001221)
            '95.4M'
            >>> bytes2human(1024 ** 3)
            '1.0G'

    """
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f %sB' % (value, s)
    return "%s B" % n


def memory_usage() -> tuple[int, int, int]:
    """Get the memory usage of the current process.

    Returns:
        tuple[int, int, int]: The resident set size (RSS), the virtual memory size (VMS), and the
            current CUDA reserved memory. If the device is not a CUDA device, the reserved memory is
            0.
    """
    proc = psutil.Process(os.getpid())

    if FlukeENV().get_device().type == "cuda":
        cuda_device = torch.device(FlukeENV().get_device())
        current_reserved = torch.cuda.memory_reserved(cuda_device)
    else:
        current_reserved = 0
    return proc.memory_info().rss, proc.memory_info().vms, current_reserved


def retrieve_obj(key: str,
                 party: Client | Server | None = None,
                 pop: bool = True) -> Any:
    """Load an object from the cache (disk).
    If the object is not found in the cache, it returns ``None``.

    Warning:
        This method assumes the cache is already initialized and opened.
        If it is not, it will raise an exception.

    Args:
        key (str, optional): specific key to add to the default filename.
        party (Client | Server, optional): the party for which to load the object. Defaults to
            ``None``.
        pop (bool, optional): Whether to remove the object from the cache after loading it.
            Defaults to ``True``.

    Returns:
        Any: The object retrieved from the cache.
    """
    prefix = ""
    if party is not None:
        prefix = f"c{party.index}_" if hasattr(party, "index") else "server_"
    cache = FlukeENV().get_cache()
    obj = cache.pop(f"{prefix}{key}", copy=(party is None)) if pop else cache.get(f"{prefix}{key}")
    return obj


def cache_obj(obj: Any,
              key: str,
              party: Server | Client | None = None) -> FlukeCache._ObjectRef | None:
    """Move the object from the RAM to the disk cache to free up memory.
    If the object is ``None``, it returns ``None`` without caching it.

    Warning:
        This method assumes the cache is already initialized and opened.
        If it is not, it will raise an exception.

    Args:
        obj (Any): The object to cache.
        key (str, optional): The key associated with the object.
        party (Client | Server): the party for which to unload the model. Defaults to ``None``.

    Returns:
        FlukeCache._ObjectRef: The object reference identifier. If the object is ``None``,
        it returns ``None``.
    """
    if obj is None:
        return None
    prefix = ""
    if party is not None:
        prefix = f"c{party.index}_" if hasattr(party, "index") else "server_"
    cache = FlukeENV().get_cache()
    return cache.push(f"{prefix}{key}", obj)


def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_dict(nested_dict: dict, sep: str = '.') -> dict:
    """Flatten a nested dictionary.
    The flatten dictionary is a dictionary where the keys are the concatenation of the keys of the
    nested dictionary separated by a separator.

    Args:
        d (dict): Nested dictionary.
        sep (str, optional): Separator. Defaults to '.'.

    Returns:
        dict: Flattened dictionary.

    Example:
        .. code-block:: python
            :linenos:

            d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
            flatten_dict(d)
            # Output: {'a': 1, 'b.c': 2, 'b.d.e': 3}

    """
    return _flatten_dict(nested_dict)
