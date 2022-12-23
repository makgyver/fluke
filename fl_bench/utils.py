import os
import json
import random
import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer

import wandb
from rich.pretty import pprint
from evaluation import Evaluator
from fl_bench.data import Distribution, INV_IIDNESS_MAP

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class OptimizerConfigurator:
    def __init__(self, optimizer_class: type[Optimizer], **optimizer_kwargs):
        self.optimizer = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
    
    def __call__(self, model: Module):
        return self.optimizer(model.parameters(), **self.optimizer_kwargs)
    
    def learning_rate(self) -> float:
        return self.optimizer_kwargs['lr']
    
    def weight_decay(self) -> float:
        return self.optimizer_kwargs['weight_decay']
    
    def __str__(self) -> str:
        to_str = f"OptCfg({self.optimizer.__name__},"
        to_str += ",".join([f"{k}={v}" for k, v in self.optimizer_kwargs.items()])
        to_str += ")"
        return to_str


class Log():
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.history = {}
    
    def update(self, model, round):
        self.history[round] = self.evaluator(model)
        print(f"Round {round}",)
        pprint(self.history[round])
    
    def __call__(self, model, round):
        self.update(model, round)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)


class WandBLog(Log):
    def __init__(self, evaluator: Evaluator, project: str, entity:str, name: str, config: dict):
        super().__init__(evaluator)
        self.run = wandb.init(project=project, #FIXME: load from config file 
                            entity=entity, #FIXME: load from config file
                            name=name,
                            config=config)
    
    def update(self, model, round):
        super().update(model, round)
        self.run.log(self.history[round], step=round)
    
    def save(self, path: str):
        super().save(path)
        self.run.finish()


def print_params(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")


def plot_comparison(*log_paths: str, metric: str='accuracy', show_loss: bool=True):
    import matplotlib.pyplot as plt
    import json
    
    iidness = os.path.basename(log_paths[0]).split("_")[-1].split(".")[0]
    iidness = INV_IIDNESS_MAP[iidness]

    #if len(iidness) > 1:
    #    raise ValueError("Cannot compare algorithms on different data distributions")

    if show_loss:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.xlabel('round')
        plt.ylabel('loss')
        for path in log_paths:
            with open(path, 'r') as f:
                history = json.load(f)
            rounds = list(history.keys())
            values = [history[round]['loss'] for round in rounds]
            plt.plot(list(map(int, rounds)), values)
    
        plt.subplot(1, 2, 2)
    
    plt.title(metric.capitalize())
    plt.xlabel('round')
    plt.ylabel(metric)
    for path in log_paths:
        with open(path, 'r') as f:
            history = json.load(f)
        rounds = list(history.keys())
        values = [history[round][metric] for round in rounds]
        plt.plot(list(map(int, rounds)), values, label=os.path.basename(path).split("_")[0])
    plt.legend()
    plt.get_current_fig_manager().set_window_title(f"{Distribution(iidness).name}")
    plt.show()