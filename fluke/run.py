"""`fluke` command line interface."""
import sys
from typing import Any

import numpy as np
import pandas as pd
import rich
import torch
import typer
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import track

sys.path.append(".")

from . import GlobalSettings  # NOQA
from .data import DataSplitter, FastDataLoader  # NOQA
from .data.datasets import Datasets  # NOQA
from .evaluation import ClassificationEval  # NOQA
from .utils import (Configuration, OptimizerConfigurator,  # NOQA
                    get_class_from_qualified_name, get_loss, get_model)
from .utils.log import get_logger  # NOQA

__version__ = "0.3.4"


def version_callback(value: bool):
    if value:
        print(f"fluke: {__version__}")
        raise typer.Exit()


app = typer.Typer()

# CONST
CONFIG_FNAME = ""


@app.command()
def centralized(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
                epochs: int = typer.Option(0, help='Number of epochs to run')) -> None:
    """Run a centralized learning experiment."""

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_container = Datasets.get(**cfg.data.dataset)

    device = GlobalSettings().get_device()

    train_loader = FastDataLoader(*data_container.train,
                                  batch_size=cfg.client.batch_size,
                                  num_labels=data_container.num_classes,
                                  shuffle=True)
    test_loader = FastDataLoader(*data_container.test,
                                 batch_size=10,
                                 num_labels=data_container.num_classes,
                                 shuffle=False)

    model = get_model(mname=cfg.model)
    if "name" not in cfg.client.optimizer:
        cfg.client.optimizer.name = torch.optim.SGD
    optimizer_cfg = OptimizerConfigurator(optimizer_cfg=cfg.client.optimizer,
                                          scheduler_cfg=cfg.client.scheduler)
    optimizer, scheduler = optimizer_cfg(model)
    criterion = get_loss(cfg.client.loss)
    evaluator = ClassificationEval(eval_every=cfg.eval.eval_every,
                                   n_classes=data_container.num_classes)
    history = []

    model.to(device)
    epochs = epochs if epochs > 0 else int(
        max(1, cfg.protocol.n_rounds * cfg.protocol.eligible_perc))

    exp_name = f"Centralized_{cfg.data.dataset.name}_E{epochs}_S{cfg.exp.seed}"
    log = get_logger(cfg.logger.name, name=exp_name, **cfg.logger.exclude('name'))
    log.init(**cfg)
    log.log(f"Centralized Learning [ #Epochs = {epochs} ]\n")

    for e in range(epochs):
        model.train()
        rich.print(f"Epoch {e+1}")
        for _, (X, y) in track(enumerate(train_loader),
                               total=train_loader.n_batches,
                               transient=True):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        epoch_eval = evaluator.evaluate(e+1, model, test_loader, criterion)
        history.append(epoch_eval)
        for k, v in epoch_eval.items():
            log.add_scalar(k, v, e+1)
        log.pretty_log(epoch_eval, title=f"Performance [Epoch {e+1}]")
        rich.print()
    model.to("cpu")


@app.command()
def federation(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
               resume: str = typer.Option(None, help='Path to the checkpoint file to load.'),
               save: str = typer.Option(None, help='Path to the checkpoint file to save.'),
               seed: int = typer.Option(None, help='Seed for reproducibility.')) -> None:
    """Run a federated learning experiment."""

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    if seed is not None:
        cfg.exp.seed = seed

    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_container = Datasets.get(**cfg.data.dataset)
    evaluator = ClassificationEval(eval_every=cfg.eval.eval_every,
                                   n_classes=data_container.num_classes)
    GlobalSettings().set_evaluator(evaluator)
    GlobalSettings().set_eval_cfg(cfg.eval)

    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg.data.distribution.name,
                                 dist_args=cfg.data.distribution.exclude("name"),
                                 **cfg.data.exclude('dataset', 'distribution'))

    fl_algo_class = get_class_from_qualified_name(cfg.method.name)
    fl_algo = fl_algo_class(cfg.protocol.n_clients,
                            data_splitter,
                            cfg.method.hyperparameters)
    # plot_distribution(fl_algo.clients)
    log = get_logger(cfg.logger.name, name=str(cfg), **cfg.logger.exclude('name'))
    log.init(**cfg)
    fl_algo.set_callbacks(log)
    rich.print(Panel(Pretty(fl_algo), title="FL algorithm"))

    if resume is not None:
        fl_algo.load(resume)

    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)

    if save is not None:
        fl_algo.save(save)


@app.command()
def clients_only(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
                 epochs: int = typer.Option(0, help='Number of epochs to run')) -> None:
    """Run a local training (for all clients) experiment."""

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
    if epochs == 0:
        epochs = max(100, int(cfg.protocol.n_rounds *
                              hp.client.local_epochs * cfg.protocol.eligible_perc))
    progress = track(enumerate(zip(clients_tr_data, clients_te_data)),
                     total=len(clients_tr_data),
                     description="Clients training...")
    exp_name = "Clients-only_" + "_".join(str(cfg).split("_")[1:])
    log = get_logger(cfg.logger.name, name=exp_name, **cfg.logger.exclude('name'))
    log.init(**cfg)

    running_evals = {c: [] for c in range(cfg.protocol.n_clients)}
    for i, (train_loader, test_loader) in progress:
        log.log(f"Client [{i}/{cfg.protocol.n_clients}]")
        model = get_model(mname=hp.model)  # , **hp.net_args)
        hp.client.optimizer.name = torch.optim.SGD
        optimizer_cfg = OptimizerConfigurator(optimizer_cfg=hp.client.optimizer,
                                              scheduler_cfg=hp.client.scheduler)
        optimizer, scheduler = optimizer_cfg(model)
        evaluator = ClassificationEval(eval_every=cfg.eval.eval_every,
                                       n_classes=data_container.num_classes)
        model.to(device)
        for e in range(epochs):
            model.train()
            for _, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_hat = model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            client_eval = evaluator.evaluate(e+1, model, test_loader, criterion)
            running_evals[i].append(client_eval)

        log.pretty_log(client_eval, title=f"Client [{i}] Performance")
        client_evals.append(client_eval)
        model.to("cpu")

    for e in range(epochs):
        for c in running_evals:
            log.add_scalars(f"Client[{c}]", running_evals[c][e], e+1)

    client_mean = pd.DataFrame(client_evals).mean(numeric_only=True).to_dict()
    client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
    log.pretty_log(client_mean, title="Overall local performance")


@app.callback()
def run(config: str = typer.Option(help="Configuration file"),
        version: bool = typer.Option(None, "--version", help="Show the installed version of fluke",
                                     callback=version_callback)) -> None:
    global CONFIG_FNAME
    CONFIG_FNAME = config


def main() -> Any:
    return app()


if __name__ == "__main__":
    main()
