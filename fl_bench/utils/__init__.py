from __future__ import annotations
import os
import json
import typer
import wandb
import importlib
import numpy as np
import pandas as pd
import psutil
from enum import Enum
from typing import Any, Sequence

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

import rich
from rich.panel import Panel
from rich.pretty import Pretty

from fl_bench.data import DistributionEnum, FastTensorDataLoader
from fl_bench.evaluation import Evaluator
from fl_bench.data.datasets import DatasetsEnum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fl_bench import Message

class DeviceEnum(Enum):
    """Device enumerator."""
    CPU: str = "cpu"    #: CPU
    CUDA: str = "cuda"  #: CUDA
    AUTO: str = "auto"  #: AUTO - automatically selects CUDA if available, otherwise CPU
    MPS: str = "mps"    #: MPS - for Apple M1/M2 GPUs


class OptimizerConfigurator:
    """Optimizer configurator.

    This class is used to configure the optimizer and the learning rate scheduler.
    To date, only the `StepLR` scheduler is supported.

    Attributes:
        optimizer (type[Optimizer]): The optimizer class.
        scheduler_kwargs (dict): The scheduler keyword arguments.
        optimizer_kwargs (dict): The optimizer keyword arguments.
    
    Todo: 
        * Add support for more schedulers.
    """ 
    def __init__(self,
                 optimizer_class: type[Optimizer], 
                 scheduler_kwargs: dict=None,
                 **optimizer_kwargs):
        self.optimizer: type[Optimizer] = optimizer_class
        self.scheduler_kwargs: DDict = (DDict(scheduler_kwargs) if scheduler_kwargs is not None
                                        else DDict({"step_size": 1, "gamma": 1}))
        self.optimizer_kwargs: DDict = DDict(optimizer_kwargs)
    
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
        scheduler = StepLR(optimizer, **self.scheduler_kwargs)
        return optimizer, scheduler
    
    def __str__(self) -> str:
        to_str = f"OptCfg({self.optimizer.__name__},"
        to_str += ",".join([f"{k}={v}" for k, v in self.optimizer_kwargs.items()])
        to_str += "," + ",".join([f"{k}={v}" for k, v in self.scheduler_kwargs.items()])
        to_str += ")"
        return to_str


class LogEnum(Enum):
    """Log enumerator."""
    LOCAL = "local" #: Local logging
    WANDB = "wandb" #: Weights and Biases logging

    def logger(self, 
               classification_eval: Evaluator, 
               eval_every: int, 
               **wandb_config):
        """Returns a new logger according to the enumerator.

        Args:
            classification_eval (Evaluator): The evaluator that will be used to evaluate the model.
            eval_every (int): The number of rounds between evaluations.
            **wandb_config (dict): The configuration for Weights and Biases.
        
        Returns:
            Log: The logger.
        """
        if self == LogEnum.LOCAL:
            return Log(classification_eval, 
                       eval_every if eval_every else 1)
        else:
            return WandBLog(
                classification_eval,
                eval_every if eval_every else 1,
                **wandb_config)

class ServerObserver():
    """Server observer interface.

    This interface is used to observe the server during the federated learning process.
    For example, it can be used to log the performance of the global model and the communication 
    costs, as it is done in the :class:`Log` class.
    """
    def start_round(self, round: int, global_model: Any):
        pass

    def end_round(self, 
                  round: int, 
                  global_model: Any, 
                  data: FastTensorDataLoader, 
                  client_evals: Sequence[Any]):
        pass

    def selected_clients(self, round: int, clients: Sequence):
        pass

    def error(self, error: str):
        pass

    def finished(self,  client_evals: Sequence[Any]):
        pass


class ChannelObserver():
    """Channel observer interface.

    This interface is used to observe the communication channel during the federated learning 
    process.
    """
    
    def message_received(self, message: Message):
        pass


class Log(ServerObserver, ChannelObserver):
    """Default logger.

    This class is used to log the performance of the global model and the communication costs during
    the federated learning process. The logging happens in the console.

    Attributes:
        evaluator (Evaluator): The evaluator that will be used to evaluate the model.
        history (dict): The history of the global model's performance.
        client_history (dict): The history of the clients' performance.
        comm_costs (dict): The history of the communication costs.
        current_round (int): The current round.
        eval_every (int): The number of rounds between evaluations.
    """

    def __init__(self, evaluator: Evaluator, eval_every: int=1):
        self.evaluator: Evaluator = evaluator
        self.history: dict = {}
        self.client_history: dict= {}
        self.comm_costs: dict = {0: 0}
        self.current_round: int = 0
        self.eval_every: int = eval_every
    
    def init(self, **kwargs):
        """Initialize the logger.

        The initialization is done by printing the configuration in the console.

        Args:
            **kwargs: The configuration.
        """
        rich.print(Panel(Pretty(kwargs, expand_all=True), title=f"Configuration"))
    
    def start_round(self, round: int, global_model: Module):
        self.comm_costs[round] = 0
        self.current_round = round

        if round == 1 and self.comm_costs[0] > 0:
            rich.print(Panel(Pretty({"comm_costs": self.comm_costs[0]}), title=f"Round: {round-1}"))

    def end_round(self, 
                  round: int, 
                  global_model: Module, 
                  data: FastTensorDataLoader, 
                  client_evals: Sequence[Any]):
        if round % self.eval_every == 0:
            self.history[round] = self.evaluator(global_model, data)
            stats = { 'global': self.history[round] }

            if client_evals:
                client_mean = pd.DataFrame(client_evals).mean(numeric_only=True).to_dict()
                client_mean = {k: np.round(float(v), 5) for k, v in client_mean.items()}
                self.client_history[round] = client_mean
                stats['local'] = client_mean
            
            stats['comm_cost'] = self.comm_costs[round]

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
                             title=f"Overall local performance"))
        
        if self.history[self.current_round]:
            rich.print(Panel(Pretty(self.history[self.current_round], expand_all=True), 
                             title=f"Overall global performance"))
        
        rich.print(Panel(Pretty({"comm_costs": sum(self.comm_costs.values())}, expand_all=True), 
                         title=f"Total communication cost"))
    
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
    def __init__(self, evaluator: Evaluator, eval_every: int, **config):
        super().__init__(evaluator, eval_every)
        self.config = config
        
    def init(self, **kwargs):
        super().init(**kwargs)
        self.config["config"] = kwargs
        self.run = wandb.init(**self.config)
    
    def start_round(self, round: int, global_model: Module):
        super().start_round(round, global_model)
        if round == 1 and self.comm_costs[0] > 0:
            self.run.log({"comm_costs": self.comm_costs[0]})

    def end_round(self, round: int, global_model: Module, data: FastTensorDataLoader, client_evals: Sequence[Any]):
        super().end_round(round, global_model, data, client_evals)
        if round % self.eval_every == 0:
            self.run.log({ "global": self.history[round]}, step=round)
            self.run.log({ "comm_cost": self.comm_costs[round]}, step=round)
            if client_evals:
                self.run.log({ "local": self.client_history[round]}, step=round)
    
    def finished(self, client_evals: Sequence[Any]):
        super().finished(client_evals)
        if client_evals:
            self.run.log({"local" : self.client_history[self.current_round+1]}, step=self.current_round+1)
    
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

def get_loss(lname: str) -> Module:
    """Get a loss function from its name.

    The supported loss functions are the ones defined in the `torch.nn` module.

    Args:
        lname (str): The name of the loss function.
    
    Returns:
        Module: The loss function.
    """
    return get_class_from_str("torch.nn", lname)()

def get_model(mname:str, module_name: str="net", **kwargs) -> Module:
    """Get a model from its name.

    This function is used to get a model from its name and the name of the module where it is
    defined. It is used to dynamically import models.

    Args:
        mname (str): The name of the model.
        module_name (str, optional): The name of the module where the model is defined. 
            Defaults to "net".
        **kwargs: The keyword arguments to pass to the model's constructor.

    Returns:
        Module: The model.
    """
    return get_class_from_str(module_name, mname)(**kwargs)

def get_scheduler(sname:str) -> torch.nn.Module:
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

def clear_cache(ipc: bool=False):
    """Clear the CUDA cache.
    
    Args:
        ipc (bool, optional): Whether to force collecting GPU memory after it has been released by 
            CUDA IPC.
    """
    torch.cuda.empty_cache()
    if ipc:
        torch.cuda.ipc_collect()


class DDict(dict):
    """A dictionary that can be accessed with dot notation recursively."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

    def __init__(self, d: dict):
        self.update(d)
    
    def update(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                self[k] = DDict(v)
            else:
                self[k] = v
    
    def exclude(self, *keys: str):
        """Create a new DDict excluding the specified keys.

        Returns:
            DDict: The new DDict.
        """
        return DDict({k: v for k, v in self.items() if k not in keys})

                
class Configuration(DDict):
    """FL-Bench configuration class.



    Args:
        DDict (_type_): _description_
    """
    def __init__(self, config_exp_path: str, config_alg_path: str):
        with open(config_exp_path) as f:
            config_exp = json.load(f)
        with open(config_alg_path) as f:
            config_alg = json.load(f)

        self.update(config_exp)
        self.update({"method": config_alg})
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
                rich.print(f"Error: {k} is required for key 'logger' when using 'wandb'.")
                error = True
        
        if not error:
            self.data.distribution.name = DistributionEnum(self.data.distribution.name)
            self.data.dataset.name = DatasetsEnum(self.data.dataset.name)
            self.exp.device = DeviceEnum(self.exp.device) if self.exp.device else DeviceEnum.CPU
            self.logger.name = LogEnum(self.logger.name)

        if error:
            raise ValueError("Configuration validation failed.")





