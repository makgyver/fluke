import sys
sys.path.append(".")
import torch
import numpy as np
import pandas as pd
import typer
import rich
from rich.progress import track
from rich.panel import Panel
from rich.pretty import Pretty

from . import GlobalSettings
from .data import DataSplitter, FastTensorDataLoader
from .utils import Configuration, OptimizerConfigurator, get_loss, get_model, get_class_from_str
from .evaluation import ClassificationEval
from .algorithms import FedAlgorithmsEnum

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
                                            batch_size=1,#cfg.method.hyperparameters.client.batch_size, 
                                            shuffle=False)

    model = get_model(mname=cfg.method.hyperparameters.model)#, **cfg.method.hyperparameters.net_args)
    optimizer_cfg = OptimizerConfigurator(torch.optim.SGD, 
                                              **cfg.method.hyperparameters.client.optimizer,
                                              scheduler_kwargs=cfg.method.hyperparameters.client.scheduler)
    optimizer, scheduler = optimizer_cfg(model)
    criterion = get_loss(cfg.method.hyperparameters.client.loss)
    evaluator = ClassificationEval(criterion, data_container.num_classes, cfg.exp.average, device=device)
    history = []
    
    model.to(device)
    epochs = epochs if epochs > 0 else int(max(1, cfg.protocol.n_rounds * cfg.protocol.eligible_perc))
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
        rich.print(Panel(Pretty(epoch_eval, expand_all=True), 
                             title=f"Performance"))
        rich.print()
    model.to("cpu")

@app.command()
def federation(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed) 
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)

    if FedAlgorithmsEnum.contains(cfg.method.name):
        fl_algo_builder = FedAlgorithmsEnum(cfg.method.name)
        fl_algo = fl_algo_builder.algorithm()(cfg.protocol.n_clients, 
                                              data_splitter, 
                                              cfg.method.hyperparameters)
    else:
        module_name = ".".join(cfg.method.name.split(".")[:-1])
        class_name = cfg.method.name.split(".")[-1]
        fl_algo_class = get_class_from_str(module_name, class_name)
        print(fl_algo_class)
        fl_algo = fl_algo_class(cfg.protocol.n_clients, 
                                data_splitter, 
                                cfg.method.hyperparameters)


    log = cfg.logger.name.logger(ClassificationEval(fl_algo.loss, 
                                                   data_splitter.num_classes(),
                                                   cfg.exp.average,
                                                   GlobalSettings().get_device()), 
                                eval_every=cfg.logger.eval_every,
                                name=str(cfg),
                                **cfg.logger.exclude('name', 'eval_every'))
    log.init(**cfg)
    fl_algo.set_callbacks(log)
    
    # if cfg.exp.checkpoint.load:
    #     fl_algo.load_checkpoint(cfg.exp.checkpoint.path)
    
    # if cfg.exp.checkpoint.save:
    #     fl_algo.activate_checkpoint(cfg.exp.checkpoint.path)

    rich.print(Panel(Pretty(fl_algo), title=f"FL algorithm"))
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
    
    (clients_tr_data, clients_te_data), server_data = \
            data_splitter.assign(cfg.protocol.n_clients, cfg.method.hyperparameters.client.batch_size)

    criterion = get_loss(cfg.method.hyperparameters.client.loss)
    client_evals = []
    for i, (train_loader, test_loader) in track(enumerate(zip(clients_tr_data, clients_te_data)), total=len(clients_tr_data)):
        rich.print(f"Client [{i}]")
        model = get_model(mname=cfg.method.hyperparameters.model)#, **cfg.method.hyperparameters.net_args)
        optimizer_cfg = OptimizerConfigurator(torch.optim.SGD, 
                                                **cfg.method.hyperparameters.client.optimizer,
                                                scheduler_kwargs=cfg.method.hyperparameters.client.scheduler)
        optimizer, scheduler = optimizer_cfg(model)
        evaluator = ClassificationEval(criterion, data_splitter.data_container.num_classes, cfg.exp.average, device=device)
        model.to(device)
        for e in range(200):
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
        rich.print(Panel(Pretty(client_eval, expand_all=True), title=f"CLient [{i}] Performance"))
        client_evals.append(client_eval)
        model.to("cpu")

    client_mean = pd.DataFrame(client_evals).mean(numeric_only=True).to_dict()
    client_mean = {k: np.round(float(v), 5) for k, v in client_mean.items()}
    rich.print(Panel(Pretty(client_mean, expand_all=True), 
                             title=f"Overall local performance"))    



@app.callback()
def run(config: str=typer.Option(CONFIG_FNAME, help="Configuration file")):
    global CONFIG_FNAME
    CONFIG_FNAME = config


if __name__ == '__main__':
    app()