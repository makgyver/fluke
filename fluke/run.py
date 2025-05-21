"""`fluke` command line interface."""

import os
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional

import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

sys.path.append(".")

from . import __version__  # NOQA
from .config import (Configuration, ConfigurationError,  # NOQA
                     OptimizerConfigurator)
from .utils import (get_class_from_qualified_name, get_loss, get_model,  # NOQA
                    plot_distribution)

console = Console()
app = typer.Typer()


def fluke_banner() -> None:
    from rich.panel import Panel

    fluke_pretty = run.__doc__
    console.print(Panel(fluke_pretty,
                  subtitle=f"v{__version__}", style="bold white"), width=53)


def version_callback(value: bool) -> None:
    if value:
        print(f"fluke: {__version__}")
        raise typer.Exit()


def _compose_config(cfg_base: str, overrides: Optional[List[str]]) -> DictConfig:
    abs_config_path = (Path.cwd() / cfg_base).resolve()
    cfg_base_folder = abs_config_path.parent
    cfg_base_name = abs_config_path.stem
    with initialize_config_dir(config_dir=str(cfg_base_folder),
                               job_name="fluke_cli",
                               version_base=None):
        return compose(config_name=cfg_base_name, overrides=overrides)


@app.command()
def centralized(exp_cfg: str = typer.Argument(..., help="Configuration file"),
                alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
                epochs: int = typer.Option(0, help='Number of epochs to run')) -> None:
    """Run a centralized learning experiment."""

    from rich.progress import track
    from torch.optim import SGD

    from . import FlukeENV  # NOQA
    from .data import FastDataLoader  # NOQA
    from .data.datasets import Datasets  # NOQA
    from .evaluation import ClassificationEval  # NOQA
    from .utils.log import get_logger  # NOQA

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

    exp_id = uuid.uuid4().hex
    exp_name = f"Centralized [{exp_id}]"
    log = get_logger(cfg.logger.name, name=exp_name, **cfg.logger.exclude('name'))
    log.init(**cfg, exp_id=exp_id)
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
        log.add_scalars(f"Epoch {e+1}", epoch_eval, e+1)
        log.pretty_log(epoch_eval, title=f"Performance [Epoch {e+1}]")
        console.print()
    model.cpu()


@app.command()
def federation(exp_cfg: str = typer.Argument(..., help="Configuration file"),
               alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
               overrides: Optional[List[str]] =
               typer.Argument(None,
                              help='Overrides for the configuration, e.g. "exp.seed=10"'),
               resume: str = typer.Option(None,
                                          help='Path to the checkpoint file to load.')) -> None:
    """Run a federated learning experiment."""

    try:

        if overrides is not None:
            overrides_exp = [v for v in overrides if not v.startswith('method.')]
            overrides_alg = [v for v in overrides if v.startswith('method.')]
            exp_cfg = _compose_config(exp_cfg, overrides_exp)
            alg_cfg = _compose_config(alg_cfg, overrides_alg)
            OmegaConf.set_struct(exp_cfg, False)
            alg_cfg = OmegaConf.create({"method": alg_cfg})
            cfg = Configuration.from_dict(OmegaConf.merge(exp_cfg, alg_cfg))
        else:
            cfg = Configuration(exp_cfg, alg_cfg)
    except ConfigurationError:
        exit(1)
    except Exception as e:
        raise e

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
    import yaml
    from rich.panel import Panel
    from rich.pretty import Pretty

    from . import FlukeENV  # NOQA
    from .data import DataSplitter  # NOQA
    from .data.datasets import Datasets  # NOQA
    from .evaluation import ClassificationEval  # NOQA
    from .utils.log import get_logger  # NOQA

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

    if cfg.save and cfg.save.path:
        path = f"{cfg.save.path}_{fl_algo.id}"
        if not os.path.exists(path):
            os.makedirs(path)
        yaml.dump(cfg.to_dict(), open(f"{path}/config.yaml", "w"))

    log_name = f"{fl_algo.__class__.__name__} [{fl_algo.id}]"
    log = get_logger(cfg.logger.name, name=log_name, **cfg.logger.exclude('name'))
    log.init(**cfg, exp_id=fl_algo.id)

    fl_algo.set_callbacks([log])
    FlukeENV().set_logger(log)
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

    import numpy as np
    from pandas import DataFrame
    from rich.progress import track
    from torch.optim import SGD

    # sys.path.append(".")
    from . import FlukeENV  # NOQA
    from .data import DataSplitter  # NOQA
    from .data.datasets import Datasets  # NOQA
    from .evaluation import ClassificationEval  # NOQA
    from .utils.log import get_logger  # NOQA

    cfg = Configuration(exp_cfg, alg_cfg)
    FlukeENV().configure(cfg)

    data_container = Datasets.get(**cfg.data.dataset)
    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg.data.distribution.name,
                                 dist_args=cfg.data.distribution.exclude("name"),
                                 **cfg.data.exclude('dataset', 'distribution'))

    device = FlukeENV().get_device()

    hp = cfg.method.hyperparameters
    if "name" not in hp.client.optimizer:
        hp.client.optimizer.name = SGD

    (clients_tr_data, clients_te_data), shared_test = \
        data_splitter.assign(cfg.protocol.n_clients, hp.client.batch_size)

    criterion = get_loss(hp.client.loss)
    client_local_evals = []
    client_shared_evals = []
    if epochs == 0:
        epochs = int(cfg.protocol.n_rounds *
                     hp.client.local_epochs *
                     cfg.protocol.eligible_perc)
    cfg.exp_id = uuid.uuid4().hex
    exp_name = f"Clients-only [{cfg.exp_id}]"

    if "tags" in cfg.logger:
        cfg.logger.tags = [f"{cfg.exp_id}"] + cfg.logger.tags
    elif cfg.logger.name in ["WandBLog", "ClearMLLog"]:
        cfg.logger.tags = [f"{cfg.exp_id}"]

    # running_local_evals = {c: [] for c in range(cfg.protocol.n_clients)}
    # running_shared_evals = {c: [] for c in range(cfg.protocol.n_clients)}
    for i, (train_loader, test_loader) in enumerate(zip(clients_tr_data, clients_te_data)):
        name = f"Client [{i}]_{cfg.exp_id}"
        if cfg.logger.name == "WandBLog":
            cfg.logger.group = exp_name
            cfg.logger.reinit = True

        log = get_logger(cfg.logger.name, name=name, **cfg.logger.exclude('name'))
        log.init(**cfg)

        log.log(f"Client [{i+1}/{cfg.protocol.n_clients}]")
        model = get_model(mname=hp.model, **hp.net_args if "net_args" in hp else {})
        model.to(device)
        optimizer_cfg = OptimizerConfigurator(optimizer_cfg=hp.client.optimizer,
                                              scheduler_cfg=hp.client.scheduler)
        optimizer, scheduler = optimizer_cfg(model)
        evaluator = ClassificationEval(eval_every=cfg.eval.eval_every,
                                       n_classes=data_container.num_classes)
        for e in track(range(epochs), description="Training...", transient=True):
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

            if test_loader is not None:
                client_local_eval = evaluator.evaluate(
                    e+1, model, test_loader, criterion, device=device)
                client_local_eval["epoch"] = e+1
                log.add_scalars(f"Client[{i}].local_test", client_local_eval, e+1)
                # running_local_evals[i].append(client_local_eval)
            if shared_test is not None:
                client_shared_eval = evaluator.evaluate(
                    e+1, model, shared_test, criterion, device=device)
                client_shared_eval["epoch"] = e + 1
                log.add_scalars(f"Client[{i}].shared_test", client_shared_eval, e+1)
                # running_shared_evals[i].append(client_shared_eval)

        perf = {}
        if test_loader is not None:
            perf["local_test"] = client_local_eval
            client_local_evals.append(client_local_eval)
        if shared_test is not None:
            perf["shared_test"] = client_shared_eval
            client_shared_evals.append(client_shared_eval)

        log.pretty_log(perf, title=f"Client [{i}] Performance")
        model.cpu()
        log.close()

    client_mean = {}
    if test_loader is not None:
        client_mu = DataFrame(client_local_evals).mean(numeric_only=True).to_dict()
        client_mean["local_test"] = {k: float(np.round(float(v), 5))
                                     for k, v in client_mu.items()}
    if shared_test is not None:
        client_mu = DataFrame(client_shared_evals).mean(numeric_only=True).to_dict()
        client_mean["shared_test"] = {k: float(np.round(float(v), 5))
                                      for k, v in client_mu.items()}

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
