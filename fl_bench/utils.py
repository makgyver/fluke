from __future__ import annotations

import os
import json
import typer
import wandb
import importlib
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from typing import Any, Iterable
from collections import OrderedDict

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

import rich
from rich.panel import Panel
from rich.pretty import Pretty
from rich.console import Console

from fl_bench.data import DistributionEnum, FastTensorDataLoader
from fl_bench.evaluation import Evaluator
from fl_bench.data.datasets import DatasetsEnum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fl_bench import Message

console = Console()

class DeviceEnum(Enum):
    CPU: str = "cpu"
    CUDA: str = "cuda"
    AUTO: str = "auto"


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
    
    def __call__(self, model: Module, **override_kwargs):
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
    
    def start_round(self, round: int, global_model: Any):
        pass

    def end_round(self, round: int, global_model: Any, data: FastTensorDataLoader, client_evals: Iterable[Any]):
        pass

    def selected_clients(self, round: int, clients: Iterable):
        pass

    def error(self, error: str):
        pass

    def finished(self):
        pass


class ChannelObserver():
    
    def message_received(self, message: Message):
        pass


class Log(ServerObserver, ChannelObserver):

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.history = {}
        self.client_history = {}
        self.comm_costs = {0: 0}
        self.current_round = 0
    
    def init(self, **kwargs):
        rich.print(Panel(Pretty(kwargs, expand_all=True), title=f"Configuration"))
    
    def start_round(self, round: int, global_model: Module):
        self.comm_costs[round] = 0
        self.current_round = round

        if round == 1 and self.comm_costs[0] > 0:
            rich.print(Panel(Pretty({"comm_costs": self.comm_costs[0]}), title=f"Round: {round-1}"))

    def end_round(self, round: int, global_model: Module, data: FastTensorDataLoader, client_evals: Iterable[Any]):
        self.history[round] = self.evaluator(global_model, data)
        stats = { 'global': self.history[round] }

        if client_evals:
            client_mean = pd.DataFrame(client_evals).mean(numeric_only=True).to_dict()
            client_mean = {k: np.round(float(v), 5) for k, v in client_mean.items()}
            self.client_history[round] = client_mean
            stats['local'] = client_mean
        
        stats['comm_cost'] = self.comm_costs[round]

        rich.print(Panel(Pretty(stats, expand_all=True), title=f"Round: {round}"))
    
    def message_received(self, message: Message):
        self.comm_costs[self.current_round] += message.get_size()
    
    def finished(self):
        rich.print(Panel(Pretty({"comm_costs": sum(self.comm_costs.values())}), 
                         title=f"Total communication cost"))
    
    def save(self, path: str):
        json_to_save = {
            "perf_global": self.history,
            "comm_costs": self.comm_costs,
            "perf_local": self.client_history
        }
        with open(path, 'w') as f:
            json.dump(json_to_save, f, indent=4)
        
    def error(self, error: str):
        console.print(f"[bold red]Error: {error}[/bold red]")


class WandBLog(Log):
    def __init__(self, evaluator: Evaluator, **config):
        super().__init__(evaluator)
        self.config = config
        
    def init(self, **kwargs):
        super().init(**kwargs)
        self.config["config"] = kwargs
        self.run = wandb.init(**self.config)
    
    def start_round(self, round: int, global_model: Module):
        super().start_round(round, global_model)
        if round == 1 and self.comm_costs[0] > 0:
            self.run.log({"comm_costs": self.comm_costs[0]})

    def end_round(self, round: int, global_model: Module, data: FastTensorDataLoader, client_evals: Iterable[Any]):
        super().end_round(round, global_model, data, client_evals)
        self.run.log(self.history[round], step=round)
        self.run.log({"comm_cost": self.comm_costs[round]}, step=round)
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

def clear_cache(ipc: bool=False):
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
        return DDict({k: v for k, v in self.items() if k not in keys})

                
class Configuration(DDict):
    def __init__(self, config_exp_path: str, config_alg_path: str):
        with open(config_exp_path) as f:
            config_exp = json.load(f)
        with open(config_alg_path) as f:
            config_alg = json.load(f)

        self.update(config_exp)
        self.update({"method": config_alg})
        self._fix_enums()
    
    def _fix_enums(self):
        self.data.distribution = DistributionEnum(self.data.distribution)
        self.data.dataset = DatasetsEnum(self.data.dataset)
        self.exp.device = DeviceEnum(self.exp.device) if self.exp.device else DeviceEnum.CPU
        self.log.logger = LogEnum(self.log.logger)
    
    def __str__(self) -> str:
        return f"{self.method.name}_data({self.data.dataset.value},{self.data.distribution.value}{',std' if self.data.standardize else ''})" + \
               f"_proto(C{self.protocol.n_clients},R{self.protocol.n_rounds},E{self.protocol.eligible_perc})" + \
               f"_seed({self.exp.seed})"

    def __repr__(self) -> str:
        return self.__str__()


def diff_model(model_dict1: dict, model_dict2: dict):
    assert model_dict1.keys() == model_dict2.keys(), "Models have not the same architecture"
    return OrderedDict({key: model_dict1[key] - model_dict2[key] for key in model_dict1.keys()})

def import_module_from_str(name: str) -> Any:
    components = name.split('.')
    mod = importlib.import_module(".".join(components[:-1]))
    mod = getattr(mod, components[-1])
    return mod