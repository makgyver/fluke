"""`fluke` command line interface."""

import os
import sys
import uuid
from typing import Any, List

import numpy as np
import typer
import yaml
from pandas import DataFrame
# from rich import print as console.print
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import track
from torch.optim import SGD

sys.path.append(".")

from . import FlukeENV  # NOQA
from .data import DataSplitter, FastDataLoader  # NOQA
from .data.datasets import Datasets  # NOQA
from .evaluation import ClassificationEval  # NOQA
from .utils import (Configuration, OptimizerConfigurator,  # NOQA
                    get_class_from_qualified_name, get_loss, get_model)
from .utils.log import get_logger  # NOQA

__version__ = "0.7.2"

console = Console()


def fluke_banner():
    fluke_pretty = run.__doc__
    console.print(Panel(fluke_pretty,
                  subtitle=f"v{__version__}", style="bold white"), width=53)


def version_callback(value: bool):
    if value:
        print(f"fluke: {__version__}")
        raise typer.Exit()

app = typer.Typer()


@app.command()
def centralized(exp_cfg: str = typer.Argument(..., help="Configuration file"),
                alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
                epochs: int = typer.Option(0, help='Number of epochs to run')) -> None:
    """Run a centralized learning experiment."""

    cfg = Configuration(exp_cfg, alg_cfg)
    FlukeENV().configure(cfg)
    data_container = Datasets.get(**cfg.data.dataset)

    device = FlukeENV().get_device()
    train_loader = FastDataLoader(*data_container.train,
                                  batch_size=cfg.client.batch_size,
                                  num_labels=data_container.num_classes,
                                  shuffle=True)
    test_loader = FastDataLoader(*data_container.test,
                                 batch_size=10,
                                 num_labels=data_container.num_classes,
                                 shuffle=False)

    hp = cfg.method.hyperparameters
    model = get_model(mname=hp.model, **hp.net_args if "net_args" in hp else {})
    if "name" not in hp.client.optimizer:
        hp.client.optimizer.name = SGD
    optimizer_cfg = OptimizerConfigurator(optimizer_cfg=hp.client.optimizer,
                                          scheduler_cfg=hp.client.scheduler)
    optimizer, scheduler = optimizer_cfg(model)
    criterion = get_loss(hp.client.loss)
    evaluator = ClassificationEval(eval_every=cfg.eval.eval_every,
                                   n_classes=data_container.num_classes)
    history = []

    model.to(device)
    epochs = epochs if epochs > 0 else int(
        max(1, cfg.protocol.n_rounds * cfg.protocol.eligible_perc))

    exp_name = f"Centralized_{cfg.data.dataset.name}_E{epochs}_S{cfg.exp.seed}"
    log = get_logger(cfg.logger.name, name=exp_name, **cfg.logger.exclude('name'))
    log.init(**cfg, exp_id=uuid.uuid4().hex)
    log.log(f"Centralized Learning [ #Epochs = {epochs} ]\n")

    for e in range(epochs):
        model.train()
        console.print(f"Epoch {e+1}")
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

        epoch_eval = evaluator.evaluate(e+1, model, test_loader, criterion, device=device)
        history.append(epoch_eval)
        for k, v in epoch_eval.items():
            log.add_scalar(k, v, e+1)
        log.pretty_log(epoch_eval, title=f"Performance [Epoch {e+1}]")
        console.print()
    model.cpu()


@app.command()
def federation(exp_cfg: str = typer.Argument(..., help="Configuration file"),
               alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
               resume: str = typer.Option(None,
                                          help='Path to the checkpoint file to load.')) -> None:
    """Run a federated learning experiment."""

    cfg = Configuration(exp_cfg, alg_cfg)
    _run_federation(cfg, resume)


@app.command()
def sweep(exp_cfg: str = typer.Argument(..., help="Configuration file"),
          alg_cfgs: List[str] = typer.Argument(...,
                                               help='Config file(s) for the algorithm(s) to run')
          ) -> None:
    """Run a battery of federated learning experiments with different configurations."""

    for alg_cfg in alg_cfgs:
        for cfg in Configuration.sweep(exp_cfg, alg_cfg):
            try:
                _run_federation(cfg)
            except Exception as e:
                # print the error with rich
                console.print(f"[yellow]Error with experiment: {cfg.verbose()}")
                console.print(f"[red]{e}")
                console.print_exception(max_frames=10)
                continue


def _run_federation(cfg: Configuration, resume: str = None) -> None:
    FlukeENV().configure(cfg)
    data_container = Datasets.get(**cfg.data.dataset)
    evaluator = ClassificationEval(eval_every=cfg.eval.eval_every,
                                   n_classes=data_container.num_classes)
    FlukeENV().set_evaluator(evaluator)

    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg.data.distribution.name,
                                 dist_args=cfg.data.distribution.exclude("name"),
                                 **cfg.data.exclude('dataset', 'distribution'))

    fl_algo_class = get_class_from_qualified_name(cfg.method.name)
    fl_algo = fl_algo_class(cfg.protocol.n_clients,
                            data_splitter,
                            cfg.method.hyperparameters)
    # plot_distribution(fl_algo.clients)
    if cfg.save and cfg.save.path:
        path = f"{cfg.save.path}_{fl_algo.id}"
        if not os.path.exists(path):
            os.makedirs(path)
        yaml.dump(cfg.to_dict(), open(f"{path}/config.yaml", "w"))

    log_name = f"{fl_algo.__class__.__name__} [{fl_algo.id}]"
    log = get_logger(cfg.logger.name, name=log_name, **cfg.logger.exclude('name'))
    log.init(**cfg, exp_id=fl_algo.id)

    fl_algo.set_callbacks([log])
    console.print(Panel(Pretty(fl_algo), title="FL algorithm", width=100))

    if resume is not None:
        fl_algo.load(resume)

    try:
        fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
    except Exception as e:
        log.log(f"Error: {e}")
        FlukeENV().force_close()
        FlukeENV.clear()
        log.close()
        FlukeENV().close_cache()
        raise e

    FlukeENV().close_cache()
    log.close()


@app.command()
def clients_only(exp_cfg: str = typer.Argument(..., help="Configuration file"),
                 alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
                 epochs: int = typer.Option(0, help='Number of epochs to run')) -> None:
    """Run a local training (for all clients) experiment."""

    cfg = Configuration(exp_cfg, alg_cfg)
    FlukeENV().configure(cfg)

    data_container = Datasets.get(**cfg.data.dataset)
    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg.data.distribution.name,
                                 dist_args=cfg.data.distribution.exclude("name"),
                                 **cfg.data.exclude('dataset', 'distribution'))

    device = FlukeENV().get_device()

    hp = cfg.method.hyperparameters
    (clients_tr_data, clients_te_data), _ = \
        data_splitter.assign(cfg.protocol.n_clients, hp.client.batch_size)

    criterion = get_loss(hp.client.loss)
    client_evals = []
    if epochs == 0:
        epochs = int(cfg.protocol.n_rounds *
                     hp.client.local_epochs *
                     cfg.protocol.eligible_perc)
    exp_name = "Clients-only_" + "_".join(str(cfg).split("_")[1:])
    log = get_logger(cfg.logger.name, name=exp_name, **cfg.logger.exclude('name'))
    log.init(**cfg, exp_id=uuid.uuid4().hex)

    if "name" not in hp.client.optimizer:
        hp.client.optimizer.name = SGD

    running_evals = {c: [] for c in range(cfg.protocol.n_clients)}
    for i, (train_loader, test_loader) in enumerate(zip(clients_tr_data, clients_te_data)):
        log.log(f"Client [{i+1}/{cfg.protocol.n_clients}]")
        model = get_model(mname=hp.model, **hp.net_args if "net_args" in hp else {})
        model.to(device)
        optimizer_cfg = OptimizerConfigurator(optimizer_cfg=hp.client.optimizer,
                                              scheduler_cfg=hp.client.scheduler)
        optimizer, scheduler = optimizer_cfg(model)
        evaluator = ClassificationEval(eval_every=cfg.eval.eval_every,
                                       n_classes=data_container.num_classes)
        for e in track(range(epochs), description="Traning...", transient=True):
            model.to(device)
            model.train()
            for _, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_hat = model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            client_eval = evaluator.evaluate(e+1, model, test_loader, criterion, device=device)
            log.add_scalar(f"Client[{i}]", client_eval, e+1)
            running_evals[i].append(client_eval)

        log.pretty_log(client_eval, title=f"Client [{i}] Performance")
        client_evals.append(client_eval)
        model.cpu()

    client_mean = DataFrame(client_evals).mean(numeric_only=True).to_dict()
    client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
    log.pretty_log(client_mean, title="Overall local performance")


@app.callback()
def run(version: bool = typer.Option(None, "--version", help="Show the installed version of fluke",
                                     callback=version_callback)) -> None:
    """\
\b\
    ██████  ████             █████
   ███░░███░░███            ░░███
  ░███ ░░░  ░███  █████ ████ ░███ █████  ██████
 ███████    ░███ ░░███ ░███  ░███░░███  ███░░███
░░░███░     ░███  ░███ ░███  ░██████░  ░███████
  ░███      ░███  ░███ ░███  ░███░░███ ░███░░░
  █████     █████ ░░████████ ████ █████░░██████
 ░░░░░     ░░░░░   ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░\
    """

    fluke_banner()


def main() -> Any:
    return app()


if __name__ == "__main__":
    main()
