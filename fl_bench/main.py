import glob

import sys; sys.path.append(".")

from sklearn.tree import DecisionTreeClassifier

from fl_bench import GlobalSettings
from data import DataSplitter, DistributionEnum, FastTensorDataLoader
from data.datasets import DatasetsEnum
from utils import *
from evaluation import ClassificationEval, ClassificationSklearnEval
from algorithms import FedAlgorithmsEnum
from algorithms import FedAdaboostAlgorithmsEnum
from rich.console import Console
from rich.pretty import Pretty
import typer

app = typer.Typer()
console = Console()

DEFAULTS = load_defaults(console)
CONFIG_FNAME = "configs/exp_settings.json"

# cli argument
# file config
# default config

@app.command()
def run(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
        name: str = cli_option(DEFAULTS["name"], help='Name of the experiment'),
        dataset: DatasetsEnum = cli_option(DEFAULTS["dataset"], help='Dataset'),
        n_clients: int = cli_option(DEFAULTS["n_clients"], help='Number of clients'),
        n_rounds: int = cli_option(DEFAULTS["n_rounds"], help='Number of rounds'),
        n_epochs: int = cli_option(DEFAULTS["n_epochs"], help='Number of epochs'),
        batch_size: int = cli_option(DEFAULTS["batch_size"], help='Batch size'),
        eligibility_percentage: float = cli_option(DEFAULTS["eligibility_percentage"], help='Eligibility percentage'),
        distribution: DistributionEnum = cli_option(DEFAULTS["distribution"], help='Data distribution'),
        seed: int = cli_option(DEFAULTS["seed"], help='Seed'),
        logger: LogEnum =  cli_option(DEFAULTS["logger"], help='Log method'),
        device: DeviceEnum = cli_option(DEFAULTS["device"], help="Device to use")):

    cfg = Config(DEFAULTS, CONFIG_FNAME, locals())
    cfg.algorithm = FedAlgorithmsEnum(cfg.method["name"])
    set_seed(cfg.seed) #Reproducibility

    GlobalSettings().set_device(cfg.device.value)
    
    loss = get_loss(cfg.loss)

    data_container = cfg.dataset.klass()()
    if cfg.standardize:
        data_container.standardize()

    model = get_model(
                cfg.model, 
                input_size=data_container.num_features, 
                num_classes=data_container.num_classes
            ).to(GlobalSettings().get_device())

    data_splitter = DataSplitter(*data_container.train,
                                 n_clients=cfg.n_clients, 
                                 distribution=cfg.distribution, 
                                 batch_size=cfg.batch_size,
                                 validation_split=cfg.validation,
                                 sampling_perc=cfg.sampling)

    # Separate test set, i.e., a test set is on the server
    test_loader = FastTensorDataLoader(*data_container.test,
                                       batch_size=100, #this can remain hard-coded
                                       shuffle=False)

    exp_name = f"{cfg.algorithm.value}_{cfg.dataset.value}_{cfg.distribution.value}_C{cfg.n_clients}_R{cfg.n_rounds}_E{cfg.n_epochs}_P{cfg.eligibility_percentage}_S{cfg.seed}" 
    log = cfg.logger.logger(ClassificationEval(test_loader, loss, data_container.num_classes, "macro"), 
                            name=exp_name,
                            **cfg.wandb_params)
    log.init(cfg)
    fl_algo = cfg.algorithm.algorithm()(n_clients=cfg.n_clients,
                                        n_rounds=cfg.n_rounds, 
                                        n_epochs=cfg.n_epochs, 
                                        eligibility_percentage=cfg.eligibility_percentage,
                                        model=model,
                                        loss_fn = loss,
                                        optimizer_cfg=OptimizerConfigurator(
                                            cfg.algorithm.optimizer(), 
                                            lr=cfg.method["optimizer_parameters"]["lr"], 
                                            scheduler_kwargs=cfg.method["optimizer_parameters"]["scheduler_kwargs"]),
                                        **cfg.method["hyperparameters"])

    rich.print(Panel(Pretty(fl_algo), title=f"FL algorithm"))
    
    fl_algo.init_parties(data_splitter, callbacks=log)
    
    if cfg.checkpoint["load"]:
        fl_algo.load_checkpoint(cfg.checkpoint["path"])
    
    if cfg.checkpoint["save"]:
        fl_algo.activate_checkpoint(cfg.checkpoint["path"])

    # GlobalSettings().set_workers(8)
    fl_algo.run()

    log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


@app.command()
def run_boost(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
              dataset: DatasetsEnum = cli_option(DEFAULTS["dataset"], help='Dataset'),
              n_clients: int = cli_option(DEFAULTS["n_clients"], help='Number of clients'),
              n_rounds: int = cli_option(DEFAULTS["n_rounds"], help='Number of rounds'),
              eligibility_percentage: float = cli_option(DEFAULTS["eligibility_percentage"], help='Eligibility percentage'),
              distribution: DistributionEnum = cli_option(DEFAULTS["distribution"], help='Data distribution'),
              seed: int = cli_option(DEFAULTS["seed"], help='Seed'),
              logger: LogEnum =  cli_option(DEFAULTS["logger"], help='Log method')):

    cfg = Config(DEFAULTS, CONFIG_FNAME, locals())
    set_seed(cfg.seed) #Reproducibility

    data_container = cfg.dataset.klass()()
    data_container.standardize()

    data_splitter = DataSplitter(*data_container.train,
                                 n_clients=cfg.n_clients, 
                                 distribution=cfg.distribution, 
                                 batch_size=cfg.batch_size,
                                 validation_split=cfg.validation,
                                 sampling_perc=cfg.sampling)

    # Separate test set, i.e., a test set is on the server
    test_loader = FastTensorDataLoader(*data_container.test,
                                       batch_size=100, #this can remain hard-coded
                                       shuffle=False)

    exp_name = f"{cfg.method['name']}_{cfg.dataset.value}_{cfg.distribution.value}_C{cfg.n_clients}_R{cfg.n_rounds}_P{cfg.eligibility_percentage}_S{cfg.seed}" 
    log = cfg.logger.logger(ClassificationSklearnEval(test_loader, "macro"), 
                            name=exp_name,
                            **cfg.wandb_params)
    log.init(**cfg)
    
    clf_args = cfg.method["hyperparameters"]["clf_args"]
    clf_args["random_state"] = cfg.seed
    base_model = import_module_from_str(cfg.method["hyperparameters"]["base_classifier"])(**clf_args)
    cfg.algorithm = FedAdaboostAlgorithmsEnum(cfg.method["name"])

    fl_algo = cfg.algorithm.algorithm()(cfg.n_clients, cfg.n_rounds, base_model, cfg.eligibility_percentage)
    
    rich.print(Panel(Pretty(fl_algo), title=f"FL algorithm"))
    
    fl_algo.init_parties(data_splitter, callbacks=log)
    fl_algo.run()

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