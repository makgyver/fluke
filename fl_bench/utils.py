import os
import json

from torch.nn import Module
from torch.optim import Optimizer

from rich.pretty import pprint
from evaluation import Evaluator

class OptimizerConfigurator:
    def __init__(self, optimizer_class: type[Optimizer], **optimizer_kwargs):
        self.optimizer = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
    
    def __call__(self, model: Module):
        return self.optimizer(model.parameters(), **self.optimizer_kwargs)
    
    def learning_rate(self):
        return self.optimizer_kwargs['lr']
    
    def weight_decay(self):
        return self.optimizer_kwargs['weight_decay']


class Log():
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.history = {}
    
    def update(self, model, round):
        self.history[round] = self.evaluator(model)
        print(f"Round {round} test:",)
        pprint(self.history[round])
    
    def __call__(self, model, round):
        self.update(model, round)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)


def print_params(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")


def plot_comparison(*log_paths: str, metric: str='accuracy', show_loss: bool=True):
    import matplotlib.pyplot as plt
    import json

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
    plt.show()