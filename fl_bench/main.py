import sys
sys.path.append(".")
import torch

import typer

import rich
from rich.progress import track
from rich.panel import Panel
from rich.pretty import Pretty

from . import GlobalSettings
from .data import DataSplitter, FastTensorDataLoader
from .utils import Configuration, OptimizerConfigurator, get_loss, get_model
from .evaluation import ClassificationEval
from .algorithms import FedAlgorithmsEnum

app = typer.Typer()

# CONST
CONFIG_FNAME = "configs/exp_settings.json"


@app.command()
def run_centralized(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
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
def run(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed) 
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)

    fl_algo_builder = FedAlgorithmsEnum(cfg.method.name)
    fl_algo = fl_algo_builder.algorithm()(cfg.protocol.n_clients, data_splitter, cfg.method.hyperparameters)

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
def validate(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):
    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    cfg._validate()
    rich.print(Panel(Pretty(cfg, expand_all=True), title=f"Configuration"))
    



@app.callback()
def main(config: str=typer.Option(CONFIG_FNAME, help="Configuration file")):
    global CONFIG_FNAME
    CONFIG_FNAME = config


if __name__ == '__main__':
    app()