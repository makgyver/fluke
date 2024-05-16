from rich.pretty import Pretty
from rich.panel import Panel
from rich.progress import track
import rich
from typing import Any
import typer
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append(".")

from . import GlobalSettings  # NOQA
from .utils import (Configuration, OptimizerConfigurator,  # NOQA
                    get_class_from_qualified_name, get_loss, get_model, get_logger)  # NOQA
from .data import DataSplitter, FastDataLoader  # NOQA
from .data.datasets import Datasets  # NOQA
from .evaluation import ClassificationEval  # NOQA

app = typer.Typer()

# CONST
CONFIG_FNAME = ""


@app.command()
def centralized(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
                epochs: int = typer.Option(0, help='Number of epochs to run')) -> None:

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_container = Datasets.get(**cfg.data.dataset)

    device = GlobalSettings().get_device()

    train_loader = FastDataLoader(*data_container.train,
                                  batch_size=cfg.method.hyperparameters.client.batch_size,
                                  num_labels=data_container.num_classes,
                                  shuffle=True)
    test_loader = FastDataLoader(*data_container.test,
                                 batch_size=10,
                                 num_labels=data_container.num_classes,
                                 shuffle=False)

    # , **cfg.method.hyperparameters.net_args)
    model = get_model(mname=cfg.method.hyperparameters.model)
    sch_args = cfg.method.hyperparameters.client.scheduler
    cfg.method.hyperparameters.client.optimizer.name = torch.optim.SGD
    optimizer_cfg = OptimizerConfigurator(optimizer_cfg=cfg.method.hyperparameters.client.optimizer,
                                          scheduler_cfg=sch_args)
    optimizer, scheduler = optimizer_cfg(model)
    criterion = get_loss(cfg.method.hyperparameters.client.loss)
    evaluator = ClassificationEval(criterion, data_container.num_classes, device)
    history = []

    model.to(device)
    epochs = epochs if epochs > 0 else int(
        max(1, cfg.protocol.n_rounds * cfg.protocol.eligible_perc))
    for e in range(epochs):
        model.train()
        rich.print(f"Epoch {e+1}")
        loss = None
        for _, (X, y) in track(enumerate(train_loader), total=train_loader.n_batches):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_eval = evaluator.evaluate(model, test_loader)
        history.append(epoch_eval)
        rich.print(Panel(Pretty(epoch_eval, expand_all=True), title="Performance"))
        rich.print()
    model.to("cpu")


@app.command()
def federation(alg_cfg: str = typer.Argument(...,
                                             help='Config file for the algorithm to run')) -> None:

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_container = Datasets.get(**cfg.data.dataset)
    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg.data.distribution.name,
                                 dist_args=cfg.data.distribution.exclude("name"),
                                 **cfg.data.exclude('dataset', 'distribution'))

    fl_algo_class = get_class_from_qualified_name(cfg.method.name)
    fl_algo = fl_algo_class(cfg.protocol.n_clients,
                            data_splitter,
                            cfg.method.hyperparameters)

    log = get_logger(cfg.logger.name, name=str(cfg), **cfg.logger.exclude('name'))
    log.init(**cfg)
    fl_algo.set_callbacks(log)
    rich.print(Panel(Pretty(fl_algo), title="FL algorithm"))
    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)


@app.command()
def clients_only(alg_cfg: str = typer.Argument(...,
                                               help='Config file for \
                                                the algorithm to run')) -> None:

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)

    data_container = Datasets.get(**cfg.data.dataset)
    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg.data.distribution.name,
                                 dist_args=cfg.data.distribution.exclude("name"),
                                 **cfg.data.exclude('dataset', 'distribution'))

    device = GlobalSettings().get_device()

    hp = cfg.method.hyperparameters
    (clients_tr_data, clients_te_data), _ = \
        data_splitter.assign(cfg.protocol.n_clients, hp.client.batch_size)

    criterion = get_loss(hp.client.loss)
    client_evals = []
    epochs = max(200, int(cfg.protocol.n_rounds *
                          hp.client.local_epochs * cfg.protocol.eligible_perc))
    progress = track(enumerate(zip(clients_tr_data, clients_te_data)),
                     total=len(clients_tr_data),
                     description="Clients training...")
    for i, (train_loader, test_loader) in progress:
        rich.print(f"Client [{i}]")
        model = get_model(mname=hp.model)  # , **hp.net_args)
        hp.client.optimizer.name = torch.optim.SGD
        optimizer_cfg = OptimizerConfigurator(optimizer_cfg=hp.client.optimizer,
                                              scheduler_cfg=hp.client.scheduler)
        optimizer, scheduler = optimizer_cfg(model)
        evaluator = ClassificationEval(criterion,
                                       data_splitter.data_container.num_classes,
                                       device)
        model.to(device)
        for _ in range(epochs):
            model.train()
            loss = None
            for _, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_hat = model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                scheduler.step()

        client_eval = evaluator.evaluate(model, test_loader)
        rich.print(Panel(Pretty(client_eval, expand_all=True), title=f"Client [{i}] Performance"))
        client_evals.append(client_eval)
        model.to("cpu")

    client_mean = pd.DataFrame(client_evals).mean(numeric_only=True).to_dict()
    client_mean = {k: np.round(float(v), 5) for k, v in client_mean.items()}
    rich.print(Panel(Pretty(client_mean, expand_all=True),
                     title="Overall local performance"))


@app.callback()
def run(config: str = typer.Option(CONFIG_FNAME, help="Configuration file")) -> None:
    global CONFIG_FNAME
    CONFIG_FNAME = config


def main() -> Any:
    return app()
