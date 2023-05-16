from collections import OrderedDict
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json
import importlib
from enum import Enum
from typing import Any, Optional, Iterable

import typer
import wandb
from fl_bench import Message
from fl_bench.data import DistributionEnum
from fl_bench.data.datasets import DatasetsEnum
from fl_bench.evaluation import Evaluator


from rich.pretty import Pretty
from rich.console import Console
from rich.panel import Panel
import rich

console = Console()

class DeviceEnum(Enum):
    CPU: str = "cpu"
    CUDA: str = "cuda"
    AUTO: str = "auto"


def set_seed(seed: int) -> None:
    """Set seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class OptimizerConfigurator:
    """Optimizer configurator.

    Parameters
    ----------
    optimizer_class : type[Optimizer]
        The optimizer class.
    scheduler_kwargs : dict, optional
        The scheduler keyword arguments, by default None. If None, the scheduler
        is set to StepLR with step_size=1 and gamma=1.
    **optimizer_kwargs
        The optimizer keyword arguments.
    """ 
    def __init__(self,
                 optimizer_class: type[Optimizer], 
                 scheduler_kwargs: dict=None,
                 **optimizer_kwargs):
        self.optimizer = optimizer_class
        if scheduler_kwargs is not None:
            self.scheduler_kwargs = scheduler_kwargs
        else:
            self.scheduler_kwargs = {"step_size": 1, "gamma": 1}
        self.optimizer_kwargs = optimizer_kwargs
    
    def __call__(self, model: Module):
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
    LOCAL = "local"
    WANDB = "wandb"

    def logger(self, classification_eval, **wandb_config):
        if self == LogEnum.LOCAL:
            return Log(classification_eval)
        else:
            return WandBLog(
                classification_eval,
                **wandb_config)

class ServerObserver():
    
    def start_round(self, round: int, global_model: Module):
        pass

    def end_round(self, round: int, global_model: Module, client_evals: Iterable[Any]):
        pass

    def selected_clients(self, round:int, clients: Iterable):
        pass

    def error(self, error: str):
        pass

class Log(ServerObserver):
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.history = {}
        self.client_history = {}
        self.comm_costs = {}
        self.current_round = 0
    
    def start_round(self, round: int, global_model: Module):
        self.comm_costs[round] = 0
        self.current_round = round

    def end_round(self, round: int, global_model: Module, client_evals: Iterable[Any]):
        self.history[round] = self.evaluator(global_model)
        stats = { 'global': self.history[round] }

        if client_evals:
            client_mean = pd.DataFrame(client_evals).mean().to_dict()
            client_mean = {k: np.round(float(v), 5) for k, v in client_mean.items()}
            self.client_history[round] = client_mean
            stats['local'] = client_mean
        
        stats['comm_cost'] = self.comm_costs[round]

        rich.print(Panel(Pretty(stats), title=f"Round: {round}"))

    def send(self, message: Message):
        self.comm_costs[self.current_round] += message.get_size()
    
    def receive(self, message: Message):
        self.comm_costs[self.current_round] += message.get_size()
    
    def save(self, path: str):
        json_to_save = {
            "global": self.history,
            "local": self.client_history
        }
        with open(path, 'w') as f:
            json.dump(json_to_save, f, indent=4)
        
    def error(self, error: str):
        console.print(f"[bold red]Error: {error}[/bold red]")


class WandBLog(Log):
    def __init__(self, evaluator: Evaluator, **config):
        super().__init__(evaluator)
        self.run = wandb.init(**config)
    
    def end_round(self, round: int, global_model: Module, client_evals: Iterable[Any]):
        super().end_round(round, global_model, client_evals)
        self.run.log(self.history[round], step=round)
        if client_evals:
            self.run.log(self.client_history[round], step=round)
    
    def save(self, path: str):
        super().save(path)
        self.run.finish()


def plot_comparison(*log_paths: str, 
                    local: bool=False, 
                    metric: str='accuracy', 
                    show_loss: bool=True) -> None:
    iidness = os.path.basename(log_paths[0]).split("_")[-1].split(".")[0]

    fig = plt.figure(figsize=(10, 6))
    if show_loss:
        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('round')
        plt.ylabel('loss')
        for path in log_paths:
            with open(path, 'r') as f:
                history = json.load(f)
            rounds = list(history["global"].keys())
            values = [history["local" if local else "global"][round]['loss'] for round in rounds]
            plt.plot(list(map(int, rounds)), values)
    
        plt.subplot(1, 2, 2)
    
    plt.title(metric.capitalize())
    plt.xlabel('round')
    plt.ylabel(metric)
    leg_curves = []
    leg_labels = []
    for path in log_paths:
        with open(path, 'r') as f:
            history = json.load(f)
        rounds = list(history["global"].keys())
        values = [history["local" if local else "global"][round][metric] for round in rounds]
        leg_labels.append(os.path.basename(path).split(")_")[0] + ")")
        leg_curves.append(plt.plot(list(map(int, rounds)), values, label=leg_labels[-1])[0])
    fig.legend(tuple(leg_curves), tuple(leg_labels), 'center left')
    plt.get_current_fig_manager().set_window_title(f"{iidness}")
    plt.show()


def load_defaults(console, config_path: Optional[str]=None):
    defaults = {
        "name": "no_name",
        "seed": 987654,
        "device": "auto",
        "n_clients": 100,
        "n_rounds": 100,
        "batch_size": 20,
        "n_epochs": 5,
        "eligibility_percentage": 0.5,
        "loss": "CrossEntropyLoss",
        "distribution": "iid",
        "model": "MLP",
        "dataset": "mnist",
        "standardize": False,
        "validation": 0.0,
        "sampling": 1.0,
        "checkpoint": {
            "save": 0,
            "load": 0,
            "path": "./checkpoints/checkpoint.pt"
        },
        "logger": "local",
        "wandb_params": {
            "project": "fl-bench",
            "entity": "mlgroup",
            "tags": []
        }
    }

    config = {}

    if config_path is not None:
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            console.print(f'Could not load config.json: {e}')

        for key in defaults.keys():
            if not key in config:
                console.log(f"[bold yellow]Warn:[/] key {key} not found in config.json, using default value {defaults[key]}")
                config[key] = defaults[key]

        return config
    else:
        return defaults

def _get_class_from_str(module_name: str, class_name: str) -> Any:
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def get_loss(lname: str) -> torch.nn.Module:
    return _get_class_from_str("torch.nn", lname)()

def get_model(mname:str, **kwargs) -> torch.nn.Module:
    return _get_class_from_str("net", mname)(**kwargs)

def get_scheduler(sname:str) -> torch.nn.Module:
    return _get_class_from_str("torch.optim.lr_scheduler", sname)

def cli_option(default: Any, help: str) -> Any:
    return typer.Option(default=None, show_default=default, help=help)

class Config(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

    def __init__(self, defaults: dict, config_fname: str, cli_args: dict):
        self.update(defaults)
        self.update(self._read_defaults(config_fname))
        self._fix_enums()
        self.update({k:v for k,v in cli_args.items() if v is not None})
        self._read_alg_cfg()
    
    def _read_defaults(self, config_fname: str) -> dict:
        with open(config_fname) as f:
            return json.load(f)
    
    def _fix_enums(self):
        self["distribution"] = DistributionEnum(self["distribution"])
        self["dataset"] = DatasetsEnum(self["dataset"])
        self["device"] = DeviceEnum(self["device"])
        self["logger"] = LogEnum(self["logger"])
    
    def _read_alg_cfg(self):
        with open(self["alg_cfg"]) as f:
            self["method"] = json.load(f)


def diff_model(model_dict1: dict, model_dict2: dict):
    assert model_dict1.keys() == model_dict2.keys(), "Models have not the same architecture"
    return OrderedDict({key: model_dict1[key] - model_dict2[key] for key in model_dict1.keys()})