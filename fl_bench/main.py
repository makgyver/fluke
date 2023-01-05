from copy import deepcopy
import torch
import glob
import torch.nn as nn

from algorithms.fedavg import FedAVG
from algorithms.fedsgd import FedSGD
from algorithms.scaffold import SCAFFOLD, ScaffoldOptimizer
from algorithms.fedprox import FedProx
from algorithms.flhalf import FLHalf
from algorithms.fedbn import FedBN
from algorithms.fedopt import FedOpt, FedOptMode

from fl_bench import GlobalSettings
from net import *
from data import DataSplitter, Distribution, FastTensorDataLoader, DatasetsEnum, IIDNESS_MAP
from utils import WandBLog, plot_comparison
from utils import OptimizerConfigurator, Log, set_seed
from evaluation import ClassificationEval

from enum import Enum

from rich.pretty import pprint
import typer
app = typer.Typer()

GlobalSettings().auto_device()
DEVICE = GlobalSettings().get_device()

class FedAlgorithmsEnum(Enum):
    FEDAVG = 'fedavg'
    FEDSGD = 'fedsgd'
    FEDPROX = 'fedprox'
    SCAFFOLD = 'scaffold'
    FLHALF = 'flhalf'
    FEDBN = 'fedbn'
    FEDOPT = 'fedopt'


N_CLIENTS = 5
N_ROUNDS = 100
N_EPOCHS = 5
BATCH_SIZE = 225
ELIGIBILITY_PERCENTAGE = .5
LOSS = nn.CrossEntropyLoss()

@app.command()
def run(algorithm: FedAlgorithmsEnum = typer.Argument(..., help='Algorithm to run'),
        dataset: DatasetsEnum = typer.Argument(..., help='Dataset'),
        n_clients: int = typer.Option(N_CLIENTS, help='Number of clients'),
        n_rounds: int = typer.Option(N_ROUNDS, help='Number of rounds'),
        n_epochs: int = typer.Option(N_EPOCHS, help='Number of epochs'),
        batch_size: int = typer.Option(BATCH_SIZE, help='Batch size'),
        elegibility_percentage: float = typer.Option(ELIGIBILITY_PERCENTAGE, help='Elegibility percentage'),
        distribution: int = typer.Option(Distribution.IID.value, help='Data distribution'),
        seed: int = typer.Option(987654, help='Seed')):
    
    set_seed(seed) #Reproducibility

    print("Running configuration:")
    options = deepcopy(locals())
    options["distribution"] = Distribution(options["distribution"]).name
    pprint(options, expand_all=True)
    print()

    MODEL = MLP().to(DEVICE)

    data_container = dataset.klass()()

    data_splitter = DataSplitter(*data_container.train,
                                 n_clients=n_clients, 
                                 distribution=Distribution(distribution), 
                                 batch_size=batch_size,
                                 validation_split=0,
                                 sampling_perc=.1)

    test_loader = FastTensorDataLoader(*data_container.test,
                                       batch_size=100, 
                                       shuffle=False)

    logger = Log(ClassificationEval(test_loader, LOSS, data_container.num_classes, "macro"))
    # logger = WandBLog(ClassificationEval(test_loader, LOSS), 
    #                   project="fl-bench",
    #                   entity="mlgroup",
    #                   name=f"{algorithm}_{dataset}_{IIDNESS_MAP[Distribution(distribution)]}", 
    #                   config=options)

    if algorithm == FedAlgorithmsEnum.FEDAVG:
        fl_algo = FedAVG(n_clients=n_clients,
                         n_rounds=n_rounds, 
                         n_epochs=n_epochs, 
                         model=MODEL, 
                         optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, scheduler_kwargs={"step_size":10, "gamma":.9}, lr=0.1), 
                         loss_fn=LOSS,
                         elegibility_percentage=elegibility_percentage)
    
    elif algorithm == FedAlgorithmsEnum.FEDSGD:
        fl_algo = FedSGD(n_clients=n_clients,
                         n_rounds=n_rounds, 
                         model=MODEL, 
                         optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.1), 
                         loss_fn=LOSS, 
                         elegibility_percentage=elegibility_percentage)
    
    elif algorithm == FedAlgorithmsEnum.FEDBN:
        fl_algo = FedBN(n_clients=n_clients,
                        n_rounds=n_rounds, 
                        n_epochs=n_epochs, 
                        model=MODEL, 
                        optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
                        loss_fn=LOSS, 
                        elegibility_percentage=elegibility_percentage)

    elif algorithm == FedAlgorithmsEnum.FEDPROX:
        fl_algo = FedProx(n_clients=n_clients,
                          n_rounds=n_rounds, 
                          n_epochs=n_epochs, 
                          client_mu=0.001,
                          model=MODEL, 
                          optimizer_cfg=OptimizerConfigurator(torch.optim.SGD,  scheduler_kwargs={"step_size":10, "gamma":.8}, lr=0.1), 
                          loss_fn=LOSS, 
                          elegibility_percentage=elegibility_percentage)

    elif algorithm == FedAlgorithmsEnum.SCAFFOLD:
        fl_algo = SCAFFOLD(n_clients=n_clients,
                           n_rounds=n_rounds, 
                           n_epochs=n_epochs,
                           global_step=1,
                           model=MODEL, 
                           optimizer_cfg=OptimizerConfigurator(ScaffoldOptimizer, scheduler_kwargs={"step_size":10, "gamma":.8}, lr=0.1), 
                           loss_fn=LOSS, 
                           elegibility_percentage=elegibility_percentage)
    
    elif algorithm == FedAlgorithmsEnum.FEDOPT:
        fl_algo = FedOpt(n_clients=n_clients,
                         n_rounds=n_rounds, 
                         n_epochs=n_epochs, 
                         mode=FedOptMode.FedYogi,
                         server_lr=0.01,
                         beta1=0.9,
                         beta2=0.99,
                         tau=0.0001,                           
                         model=MODEL, 
                         optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
                         loss_fn=LOSS, 
                         elegibility_percentage=elegibility_percentage)

    elif algorithm == FedAlgorithmsEnum.FLHALF:
        fl_algo = FLHalf(n_clients=n_clients,
                         n_rounds=n_rounds, 
                         client_n_epochs=n_epochs, 
                         server_n_epochs=2,
                         server_batch_size=batch_size, 
                         model=MODEL, 
                         server_optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
                         client_optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.05), 
                         loss_fn=LOSS, 
                         private_layers=["fc1", "bn1"],
                         elegibility_percentage=elegibility_percentage)
    
    else:
        raise ValueError(f'Algorithm {algorithm} not supported')
    
    print("FL algorithm: ") 
    pprint(fl_algo)
    print()
    
    fl_algo.init_parties(data_splitter, callback=logger)
    fl_algo.run()
    logger.save(f'./log/{fl_algo}_{dataset.value}_{IIDNESS_MAP[Distribution(distribution)]}.json')


@app.command()
def compare(dataset: str=typer.Option('mnist', help='Dataset'),
            n_clients: int=typer.Option(100, help='Number of clients'),
            n_rounds: int=typer.Option(100, help='Number of rounds'),
            distribution: int=typer.Option(Distribution.IID.value, help='Data distribution'),
            show_loss: bool=typer.Option(True, help='Show loss graph'),
            local: bool=typer.Option(False, help='Compare client-side results')):

    paths = glob.glob(f'./log/*C={n_clients},R={n_rounds},*_{dataset}_{IIDNESS_MAP[Distribution(distribution)]}.json')
    plot_comparison(*paths, local=local, show_loss=show_loss)

# compare('./log/flhalf_noniid_dir.json', './log/fedavg_noniid_dir.json', show_loss=True) 


if __name__ == '__main__':
    app()