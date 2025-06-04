from __future__ import annotations

import inspect
import os
import sys
from itertools import product
from typing import Any, Optional

from cerberus import Validator
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from rich import print as rich_print
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from . import DDict  # NOQA
from .utils import get_optimizer, get_scheduler  # NOQA

sys.path.append(".")


__all__ = ["Configuration", "ConfigurationError", "OptimizerConfigurator"]


class ConfigurationError(Exception):
    """Exception raised when the configuration is not valid."""

    def __init__(self, message: str = "Invalid configuration.", errors_dict: Optional[dict] = None):
        super().__init__(message)
        rich_print(f"[red]ConfigurationError:[/red] {message}")
        if errors_dict is not None:
            rich_print(errors_dict)


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

    def __init__(
        self,
        config_exp_path: str = None,
        config_alg_path: str = None,
        force_validation: bool = True,
    ):
        super().__init__()

        if config_exp_path is not None and os.path.exists(config_exp_path):
            cfg_exp = OmegaConf.load(config_exp_path)
            self.update(DDict(**cfg_exp))

        if config_alg_path is not None and os.path.exists(config_alg_path):
            cfg_alg = OmegaConf.load(config_alg_path)
            self.update(method=DDict(**cfg_alg))

        if force_validation:
            self._validate()

    @classmethod
    def from_dict(cls, cfg_dict: dict | DictConfig) -> Configuration:
        """Create a configuration from a dictionary.

        Args:
            cfg_dict (dict | DictConfig): The dictionary.

        Returns:
            Configuration: The configuration.
        """
        cfg = Configuration(force_validation=False)
        cfg.update(**cfg_dict)
        cfg._validate()
        return cfg

    @classmethod
    def fromkeys(cls, *args, **kwargs) -> Configuration:
        # Hides the fromkeys method of dict
        raise AttributeError("'Configuration' class has no method 'fromkeys'")

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary.

        Returns:
            dict: The dictionary.
        """

        def _to_dict(ddict: DDict) -> Any:
            if not isinstance(ddict, dict):
                if isinstance(ddict, type):
                    return ddict.__name__
                return ddict
            return {k: _to_dict(v) for k, v in ddict.items()}

        return _to_dict(self)

    @classmethod
    def sweep(cls, config_exp_path: str, config_alg_path: str) -> list[Configuration]:
        """Generate configurations from a sweep.
        This method is used to generate configurations from a sweep. The sweep is defined by the
        experiment configuration file. The method yields a configuration for each combination of
        hyperparameters.

        Args:
            config_exp_path (str): The path to the experiment configuration file.
            config_alg_path (str): The path to the algorithm configuration file.

        Returns:
            list[Configuration]: A list of configurations.
        """
        cfgs = Configuration(config_exp_path, config_alg_path, force_validation=False)
        all_configs = Configuration.__sweep(cfgs)
        return [Configuration.from_dict(cfg) for cfg in all_configs]

    @staticmethod
    def __sweep(cfgs: DDict | dict) -> list[Configuration]:
        """Generate configurations from a sweep.
        This method is used to generate configurations from a sweep. The sweep is defined by the
        experiment configuration file. The method yields a configuration for each combination of
        hyperparameters.

        Args:
            cfgs (DDict | dict): The configuration.

        Returns:
            list[Configuration]: A list of configurations.
        """
        normalized = {
            k: v if isinstance(v, (list, dict, ListConfig)) else [v] for k, v in cfgs.items()
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
                        "path": {
                            "type": "string",
                            "required": False,
                            "default": "./data",
                        },
                    },
                },
                "distribution": {
                    "type": "dict",
                    "required": True,
                    "schema": {"name": {"type": "string", "required": True}},
                },
                "sampling_perc": {
                    "type": "float",
                    "required": False,
                    "min": 0.001,
                    "max": 1.0,
                    "default": 1.0,
                },
                "client_split": {
                    "type": "float",
                    "required": False,
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                },
                "keep_test": {"type": "boolean", "required": False, "default": True},
                "server_test": {"type": "boolean", "required": False, "default": True},
                "server_split": {
                    "type": "float",
                    "required": False,
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                },
                "uniform_test": {
                    "type": "boolean",
                    "required": False,
                    "default": False,
                },
            },
        },
        "exp": {
            "type": "dict",
            "schema": {
                "device": {
                    "required": False,
                    "default": "cpu",
                    "anyof": [
                        {
                            "type": "string",
                            "anyof": [
                                {"allowed": ["cpu", "cuda", "mps"]},
                                {"regex": "^cuda:[0-9]+$"},
                            ],
                        },
                        {"type": "list"},
                    ],
                },
                "seed": {"type": "integer", "required": True, "default": 42},
                "inmemory": {"type": "boolean", "required": True, "default": True},
            },
        },
        "eval": {
            "type": "dict",
            "schema": {
                "task": {
                    "type": "string",
                    "required": False,
                    "default": "classification",
                    "allowed": ["classification"],
                },
                "eval_every": {
                    "type": "integer",
                    "required": False,
                    "default": 1,
                    "min": 1,
                },
                "pre_fit": {"type": "boolean", "required": False, "default": False},
                "post_fit": {"type": "boolean", "required": False, "default": True},
                "locals": {"type": "boolean", "required": False, "default": False},
                "server": {"type": "boolean", "required": False, "default": True},
            },
        },
        "logger": {
            "type": "dict",
            "schema": {"name": {"type": "string", "required": False, "default": "Log"}},
        },
        "protocol": {
            "type": "dict",
            "schema": {
                "eligible_perc": {
                    "type": "float",
                    "required": True,
                    "min": 0.0,
                    "max": 1.0,
                },
                "n_clients": {"type": "integer", "required": True, "min": 1},
                "n_rounds": {"type": "integer", "required": True, "min": 1},
            },
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
                                "batch_size": {
                                    "type": "integer",
                                    "required": True,
                                    "min": 0,
                                },
                                "local_epochs": {
                                    "type": "integer",
                                    "required": True,
                                    "min": 1,
                                },
                                "loss": {"type": "string", "required": True},
                                "optimizer": {
                                    "type": "dict",
                                    "schema": {
                                        "name": {
                                            "type": "string",
                                            "required": False,
                                            "default": "SGD",
                                        },
                                        "lr": {
                                            "type": "float",
                                            "required": False,
                                            "default": 0.01,
                                        },
                                    },
                                },
                                "scheduler": {
                                    "type": "dict",
                                    "schema": {
                                        "name": {
                                            "type": "string",
                                            "required": False,
                                            "default": "StepLR",
                                        }
                                    },
                                },
                            },
                        },
                        "server": {"type": "dict"},
                        "model": {"type": "string", "required": True},
                    },
                },
                "name": {"type": "string", "required": True},
            },
        },
    }

    @staticmethod
    def __repair_save(data: dict) -> tuple:
        if "save" not in data:
            return {}, []

        save_valid = Validator()
        save_valid.schema = {
            "save_every": {"type": "integer", "default": 1, "min": 1},
            "path": {"type": "string", "default": "./models"},
            "global_only": {"type": "boolean", "default": False},
        }
        save_valid.allow_unknown = False
        valid_result = save_valid.validate(data["save"])
        if not valid_result:
            return None, save_valid.errors

        return save_valid.document, []

    def _validate(self) -> None:

        validator = Validator()
        validator.schema = self.__SCHEMA
        validator.allow_unknown = True

        cfg_dict = self.to_dict()
        valid_result = validator.validate(cfg_dict)
        save_valid_result, save_errors = Configuration.__repair_save(cfg_dict)

        errors = validator.errors
        if save_errors:
            errors.update(save=save_errors)

        if not valid_result:
            raise ConfigurationError(errors_dict=errors)

        clean_cfg = validator.document
        clean_cfg["save"] = save_valid_result

        self.update(clean_cfg)

    def verbose(self) -> str:
        return super().__str__()


class OptimizerConfigurator:
    """This class is used to configure the optimizer and the learning rate scheduler.

    Attributes:
        optimizer (type[Optimizer]): The optimizer class.
        scheduler (type[LRScheduler]): The learning rate scheduler class.
        optimizer_cfg (DDict): The optimizer keyword arguments.
        scheduler_cfg (DDict): The scheduler keyword arguments.
    """

    def __init__(self, optimizer_cfg: DDict | dict, scheduler_cfg: DDict | dict = None):
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

        self.optimizer_cfg: DDict | None = None
        self.scheduler_cfg: DDict | None = None
        self.optimizer: type[Optimizer] | None = None
        self.scheduler: type[LRScheduler] | None = None

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
        elif inspect.isclass(self.optimizer_cfg.name) and issubclass(
            self.optimizer_cfg.name, Optimizer
        ):
            self.optimizer = self.optimizer_cfg.name
        else:
            raise ValueError("Invalid optimizer name. Must be a string or an optimizer class.")

        if "name" not in self.scheduler_cfg:
            self.scheduler = get_scheduler("StepLR")
        elif isinstance(self.scheduler_cfg.name, str):
            self.scheduler = get_scheduler(self.scheduler_cfg.name)
        elif inspect.isclass(self.scheduler_cfg.name) and issubclass(
            self.scheduler_cfg.name, LRScheduler
        ):
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
            optimizer = self.optimizer(
                filter(lambda p: p.requires_grad, model.parameters()),
                **self.optimizer_cfg,
            )
        else:
            optimizer = self.optimizer(filter_fun(model), **self.optimizer_cfg)
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
