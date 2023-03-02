import glob

import sys; sys.path.append(".")
from fl_bench import GlobalSettings
from data import DataSplitter, DistributionEnum, FastTensorDataLoader, DatasetsEnum
from utils import *
from evaluation import ClassificationEval
from algorithms import FedAlgorithmsEnum
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
        dataset: DatasetsEnum = cli_option(DEFAULTS["dataset"], help='Dataset'),
        n_clients: int = cli_option(DEFAULTS["n_clients"], help='Number of clients'),
        n_rounds: int = cli_option(DEFAULTS["n_rounds"], help='Number of rounds'),
        n_epochs: int = cli_option(DEFAULTS["n_epochs"], help='Number of epochs'),
        batch_size: int = cli_option(DEFAULTS["batch_size"], help='Batch size'),
        eligibility_percentage: float = cli_option(DEFAULTS["eligibility_percentage"], help='Elegibility percentage'),
        distribution: DistributionEnum = cli_option(DEFAULTS["distribution"], help='Data distribution'),
        seed: int = cli_option(DEFAULTS["seed"], help='Seed'),
        logger: LogEnum =  cli_option(DEFAULTS["logger"], help='Log method'),
        device: DeviceEnum = cli_option(DEFAULTS["device"], help="Device to use")):

    cfg = Config(DEFAULTS, CONFIG_FNAME, locals())
    cfg.algorithm = FedAlgorithmsEnum(cfg.method["name"])
    set_seed(cfg.seed) #Reproducibility

    console.log("Running configuration:", Pretty(cfg), end="\n\n", )

    GlobalSettings().set_device(cfg.device.value)
    model = get_model(cfg.model).to(GlobalSettings().get_device())
    loss = get_loss(cfg.loss)

    data_container = cfg.dataset.klass()()

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
    
    # Test clients, i.e., 10% of the clients are used for testing using the same distribution as the training set
    # test_loader = DataSplitter(*data_container.test, 
    #                            n_clients=n_clients // 10, 
    #                            distribution=distribution, 
    #                            batch_size=batch_size, 
    #                            validation_split=0.0, 
    #                            sampling_perc=1.0).get_loaders()[0]


    exp_name = f"{cfg.algorithm.value}_{cfg.dataset.value}_{cfg.distribution.value}_C{cfg.n_clients}_R{cfg.n_rounds}_E{cfg.n_epochs}_P{cfg.eligibility_percentage}_S{cfg.seed}" 
    log = cfg.logger.logger(ClassificationEval(test_loader, loss, data_container.num_classes, "macro"), 
                            name=exp_name,
                            **cfg.wandb_params)
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

    console.log(f"FL algorithm: {fl_algo}", end="\n\n") 
    
    fl_algo.init_parties(data_splitter, callbacks=log)
    # GlobalSettings().set_workers(8)
    fl_algo.run()
    log.save(f'./log/{fl_algo}_{cfg.dataset.value}_{cfg.distribution.value}.json')


@app.command()
def compare(dataset: str=typer.Option('mnist', help='Dataset'),
            n_clients: int=typer.Option(100, help='Number of clients'),
            n_rounds: int=typer.Option(100, help='Number of rounds'),
            distribution: DistributionEnum=typer.Option(DistributionEnum.IID.value, help='Data distribution'),
            show_loss: bool=typer.Option(True, help='Show loss graph'),
            local: bool=typer.Option(False, help='Compare client-side results')):
    paths = glob.glob(f'./log/*C={n_clients},R={n_rounds},*_{dataset}_{distribution}.json')
    plot_comparison(*paths, local=local, show_loss=show_loss)


@app.callback()
def main(config: str=typer.Option(CONFIG_FNAME, help="Configuration file")): 
    global CONFIG_FNAME
    CONFIG_FNAME = config


if __name__ == '__main__':
    app()