import sys; sys.path.append(".")

import typer
from rich.pretty import Pretty

from fl_bench import GlobalSettings
from fl_bench.data import DataSplitter
from fl_bench.utils import *
from fl_bench.evaluation import ClassificationEval, ClassificationSklearnEval
from fl_bench.algorithms import FedAdaboostAlgorithmsEnum, FedAlgorithmsEnum

app = typer.Typer()

# CONST
CONFIG_FNAME = "configs/exp_settings.json"

@app.command()
def run(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed) 
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)

    fl_algo_builder = FedAlgorithmsEnum(cfg.method.name)
    fl_algo = fl_algo_builder.algorithm()(cfg.protocol.n_clients, data_splitter, cfg.method.hyperparameters)

    log = cfg.exp.logger.logger(ClassificationEval(fl_algo.loss, 
                                                   data_splitter.num_classes(), 
                                                   "macro",
                                                   GlobalSettings().get_device()), 
                                name=str(cfg),
                                **cfg.exp.wandb_params)
    log.init(**cfg)
    fl_algo.set_callbacks(log)
    
    if cfg.exp.checkpoint.load:
        fl_algo.load_checkpoint(cfg.exp.checkpoint.path)
    
    if cfg.exp.checkpoint.save:
        fl_algo.activate_checkpoint(cfg.exp.checkpoint.path)

    rich.print(Panel(Pretty(fl_algo), title=f"FL algorithm"))
    # GlobalSettings().set_workers(8)
    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
    log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


@app.command()
def run_boost(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed) 
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)
    
    fl_algo_builder = FedAdaboostAlgorithmsEnum(cfg.method.name)
    fl_algo = fl_algo_builder.algorithm()(cfg.protocol.n_clients, data_splitter, cfg.method.hyperparameters)

    log = cfg.exp.logger.logger(ClassificationSklearnEval("macro"), 
                                name=str(cfg),
                                **cfg.exp.wandb_params)
    log.init(**cfg)
    fl_algo.set_callbacks(log)

    rich.print(Panel(Pretty(fl_algo), title=f"FL algorithm"))

    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
    log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


@app.command()
def compare(paths: list[str]=typer.Argument(..., help='Log files to compare'),
            show_loss: bool=typer.Option(True, help='Show loss graph'),
            local: bool=typer.Option(False, help='Compare client-side results')):
    plot_comparison(*paths, local=local, show_loss=show_loss)


@app.callback()
def main(config: str=typer.Option(CONFIG_FNAME, help="Configuration file")): 
    global CONFIG_FNAME
    CONFIG_FNAME = config


if __name__ == '__main__':
    app()