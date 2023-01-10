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



import wandb
from fl_bench.evaluation import Evaluator
from fl_bench.data import Distribution

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

    def logger(self, classification_eval, wandb_config):
        if self == LogEnum.LOCAL:
            return Log(classification_eval)
        else:
            return WandBLog(
                classification_eval,
                **wandb_config)


class Log():
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.history = {}
        self.client_history = {}
    
    def update(self, model, round, client_evals):
        self.history[round] = self.evaluator(model)
        print(f"[Round {round}]")
        print(f"\tglobal: {self.history[round]}")
        if client_evals:
            client_mean = pd.DataFrame(client_evals).mean().to_dict()
            client_mean = {k: np.round(float(v), 5) for k, v in client_mean.items()}
            self.client_history[round] = client_mean
            print(f"\tlocal: {client_mean}")
    
    def __call__(self, model, round, client_evals=None):
        self.update(model, round, client_evals)
    
    def save(self, path: str):
        json_to_save = {
            "global": self.history,
            "local": self.client_history
        }
        with open(path, 'w') as f:
            json.dump(json_to_save, f, indent=4)


class WandBLog(Log):
    def __init__(self, evaluator: Evaluator, project: str, entity:str, name: str, config: dict):
        super().__init__(evaluator)
        self.run = wandb.init(project=project, #FIXME: load from config file 
                              entity=entity,     #FIXME: load from config file
                              name=name,
                              config=config)
    
    def update(self, model, round):
        super().update(model, round)
        self.run.log(self.history[round], step=round)
    
    def save(self, path: str):
        super().save(path)
        self.run.finish()


def plot_comparison(*log_paths: str, 
                    local: bool=False, 
                    metric: str='accuracy', 
                    show_loss: bool=True) -> None:
    iidness = os.path.basename(log_paths[0]).split("_")[-1].split(".")[0]

    if show_loss:
        plt.figure(figsize=(10, 5))
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
    for path in log_paths:
        with open(path, 'r') as f:
            history = json.load(f)
        rounds = list(history["global"].keys())
        values = [history["local" if local else "global"][round][metric] for round in rounds]
        plt.plot(list(map(int, rounds)), values, label=os.path.basename(path).split("_")[0])
    plt.legend()
    plt.get_current_fig_manager().set_window_title(f"{Distribution(iidness).name}")
    plt.show()


def load_defaults(console):
    defaults = {
        "name": "NAME OF THE EXP",
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
        "method": {
            "name": "fedavg",
            "optimizer_parameters": {
                "lr": 0.0001,
                "scheduler_kwargs": {
                    "step_size":10, 
                    "gamma":0.9
                }
            },
            "hyperparameters": {

            }
        },
        "dataset": "mnist",
        "validation": 0.1,
        "sampling": 0.1,
        "logger": "local",
        "wandb_params": {
            "project": "fl-bench",
            "entity": "mlgroup",
            "tags": []
        }
    }

    config = {}

    try:
        with open('config.json') as f:
            config = json.load(f)
    except Exception as e:
        console.print(f'Could not load config.json: {e}')

    for key in defaults.keys():
        if not key in config:
            console.log(f"[bold yellow]Warn:[/] key {key} not found in config.json, using default value {defaults[key]}")
            config[key] = defaults[key]

    return config

def get_loss(lname:str) -> torch.nn.Module:
    module = importlib.import_module("torch.nn")
    class_ = getattr(module, lname)
    return class_()

def get_model(mname:str) -> torch.nn.Module:
    module = importlib.import_module("net")
    class_ = getattr(module, mname)
    return class_()