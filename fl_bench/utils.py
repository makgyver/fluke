import json

from torch.nn import Module
from torch.optim import Optimizer

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
        print(f"Round {round} test: {self.history[round]}")
    
    def __call__(self, model, round):
        self.update(model, round)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)


def print_params(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")