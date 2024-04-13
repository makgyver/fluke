from .evaluation import ClassificationEval
from .utils import (Configuration, OptimizerConfigurator,
                    get_class_from_qualified_name, get_loss, get_model)
from .data import DataSplitter, FastTensorDataLoader
from . import GlobalSettings
from rich.pretty import Pretty
from rich.panel import Panel
from rich.progress import track
import rich
import typer
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append(".")

# from .algorithms import FedAlgorithmsEnum

app = typer.Typer()

# CONST
CONFIG_FNAME = "configs/exp_settings.json"


@app.command()
def centralized(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
                epochs: int = typer.Option(0, help='Number of epochs to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_container = cfg.data.dataset.name.klass()(**cfg.data.dataset.exclude('name'))

    device = GlobalSettings().get_device()

    train_loader = FastTensorDataLoader(*data_container.train,
                                        batch_size=cfg.method.hyperparameters.client.batch_size,
                                        shuffle=True)
    test_loader = FastTensorDataLoader(*data_container.test,
                                       batch_size=1,
                                       shuffle=False)

    # , **cfg.method.hyperparameters.net_args)
    model = get_model(mname=cfg.method.hyperparameters.model)
    sch_args = cfg.method.hyperparameters.client.scheduler
    optimizer_cfg = OptimizerConfigurator(torch.optim.SGD,
                                          **cfg.method.hyperparameters.client.optimizer,
                                          scheduler_kwargs=sch_args)
    optimizer, scheduler = optimizer_cfg(model)
    criterion = get_loss(cfg.method.hyperparameters.client.loss)
    evaluator = ClassificationEval(
        criterion, data_container.num_classes, cfg.exp.average, device=device)
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
def federation(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)

    fl_algo_class = get_class_from_qualified_name(cfg.method.name)
    fl_algo = fl_algo_class(cfg.protocol.n_clients,
                            data_splitter,
                            cfg.method.hyperparameters)

    log = cfg.logger.name.logger(name=str(cfg), **cfg.logger.exclude('name'))
    log.init(**cfg)
    fl_algo.set_callbacks(log)

    # if cfg.exp.checkpoint.load:
    #     fl_algo.load_checkpoint(cfg.exp.checkpoint.path)

    # if cfg.exp.checkpoint.save:
    #     fl_algo.activate_checkpoint(cfg.exp.checkpoint.path)

    rich.print(Panel(Pretty(fl_algo), title="FL algorithm"))
    # GlobalSettings().set_workers(8)
    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
    # log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


@app.command()
def clients_only(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)

    device = GlobalSettings().get_device()

    hp = cfg.method.hyperparameters
    (clients_tr_data, clients_te_data), _ = \
        data_splitter.assign(cfg.protocol.n_clients, hp.client.batch_size)

    criterion = get_loss(hp.client.loss)
    client_evals = []
    progress = track(enumerate(zip(clients_tr_data, clients_te_data)), total=len(clients_tr_data))
    for i, (train_loader, test_loader) in progress:
        rich.print(f"Client [{i}]")
        model = get_model(mname=hp.model)  # , **hp.net_args)
        optimizer_cfg = OptimizerConfigurator(torch.optim.SGD,
                                              **hp.client.optimizer,
                                              scheduler_kwargs=hp.client.scheduler)
        optimizer, scheduler = optimizer_cfg(model)
        evaluator = ClassificationEval(criterion,
                                       data_splitter.data_container.num_classes,
                                       cfg.exp.average,
                                       device=device)
        model.to(device)
        for _ in range(200):
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
def run(config: str = typer.Option(CONFIG_FNAME, help="Configuration file")):
    global CONFIG_FNAME
    CONFIG_FNAME = config


if __name__ == '__main__':
    app()
