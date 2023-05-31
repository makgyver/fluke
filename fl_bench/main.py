import sys; sys.path.append(".")

import typer
from rich.pretty import Pretty
from rich.console import Console

from fl_bench import GlobalSettings
from fl_bench.data import DataSplitter
from fl_bench.utils import *
from fl_bench.evaluation import ClassificationEval
from fl_bench.algorithms import FedAlgorithmsEnum


app = typer.Typer()
console = Console()

# DEFAULTS = load_defaults(console)
CONFIG_FNAME = "configs/exp_settings.json"

# cli argument
# file config
# default config

@app.command()
def run(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed) 
    GlobalSettings().set_device(cfg.exp.device)
    data_splitter = DataSplitter.from_config(cfg.data)

    fl_algo_builder = FedAlgorithmsEnum(cfg.method.name)
    fl_algo = fl_algo_builder.algorithm()()
    fl_algo.init_parties(cfg.protocol.n_clients, data_splitter, cfg.method.hyperparameters)

    log = cfg.exp['logger'].logger(ClassificationEval(fl_algo.server.test_data, 
                                               fl_algo.loss, 
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

    rich.print(Panel(Pretty(cfg), title=f"FL algorithm"))
    # GlobalSettings().set_workers(8)
    fl_algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
    log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


# @app.command()
# def run_boost(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run')):

#     cfg = Configuration(CONFIG_FNAME, alg_cfg)
#     GlobalSettings().set_seed(cfg.exp.seed) 
#     GlobalSettings().set_device(cfg.exp.device)

#     data_splitter = DataSplitter.from_config(cfg.data)

#     log = cfg.logger.logger(ClassificationSklearnEval(test_loader, "macro"), 
#                             name=exp_name,
#                             **cfg.wandb_params)
#     log.init(**cfg)
    
#     clf_args = cfg.method["hyperparameters"]["clf_args"]
#     clf_args["random_state"] = cfg.seed
#     base_model = import_module_from_str(cfg.method["hyperparameters"]["base_classifier"])(**clf_args)
#     cfg.algorithm = FedAdaboostAlgorithmsEnum(cfg.method["name"])

#     fl_algo = cfg.algorithm.algorithm()(cfg.n_clients, cfg.n_rounds, base_model, cfg.eligible_perc)
    
#     rich.print(Panel(Pretty(fl_algo), title=f"FL algorithm"))
    
#     fl_algo.init_parties(data_splitter, callbacks=log)
#     fl_algo.run()

#     log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


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