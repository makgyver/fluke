from copy import deepcopy
import torch
import torch.nn as nn
from torch.functional import F

from algorithms.fedavg import FedAVG
from algorithms.fedsgd import FedSGD
from algorithms.scaffold import SCAFFOLD, ScaffoldOptimizer
from algorithms.fedprox import FedProx
from algorithms.flhalf import FLHalf

from fl_bench import GlobalSettings
from data import Datasets
from fl_bench.data import DataSplitter, Distribution, FastTensorDataLoader
from fl_bench.utils import plot_comparison
from utils import OptimizerConfigurator, Log
from evaluation import ClassificationEval


from rich.pretty import pprint
import typer
app = typer.Typer()



GlobalSettings().auto_device()
DEVICE = GlobalSettings().get_device()

# train_data, test_data = Datasets.FEMNIST()
# train_data, test_data = Datasets.EMNIST()
train_data, test_data = Datasets.MNIST()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        #self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        #self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        #x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        #x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)
    
    def forward_(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return x
        #x = self.fc1_drop(x)
        #x = F.relu(self.fc2(x))
        #x = self.fc2_drop(x)
        #return F.log_softmax(self.fc3(x), dim=1)

# Define model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output #, x    # return x for visualization

test_loader = FastTensorDataLoader(test_data.data / 255., test_data.targets, batch_size=100, shuffle=False)
logger = Log(ClassificationEval(test_loader, nn.CrossEntropyLoss()))

N_CLIENTS = 100
N_ROUNDS = 100
N_EPOCHS = 5
BATCH_SIZE = 225
ELIGIBILITY_PERCENTAGE = .5
MODEL = MLP()


@app.command()
def run(algorithm: str = typer.Argument(..., help='Algorithm to run'),
        n_clients: int = typer.Option(N_CLIENTS, help='Number of clients'),
        n_rounds: int = typer.Option(N_ROUNDS, help='Number of rounds'),
        n_epochs: int = typer.Option(N_EPOCHS, help='Number of epochs'),
        batch_size: int = typer.Option(BATCH_SIZE, help='Batch size'),
        elegibility_percentage: float = typer.Option(ELIGIBILITY_PERCENTAGE, help='Elegibility percentage'),
        distribution: int = typer.Option(Distribution.IID.value, help='Data distribution'),
        seed: int = typer.Option(42, help='Seed')):
    
    print("Running configuration:")
    options =  deepcopy(locals())
    options["distribution"] = Distribution(options["distribution"]).name
    pprint(options, expand_all=True)
    print()

    data_splitter = DataSplitter(train_data.data / 255., 
                                 train_data.targets, 
                                 n_clients=n_clients, 
                                 distribution=Distribution(distribution), 
                                 batch_size=batch_size)

    if algorithm == 'fedavg':
        fl_algo = FedAVG(n_clients=n_clients,
                           n_rounds=n_rounds, 
                           n_epochs=n_epochs, 
                           batch_size=batch_size, 
                           model=MODEL, 
                           optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
                           loss_fn=nn.CrossEntropyLoss(), 
                           elegibility_percentage=elegibility_percentage,
                           seed=seed)
        fl_algo.init_parties(data_splitter, logger)

    elif algorithm == 'fedprox':
        fl_algo = FedProx(n_clients=n_clients,
                            n_rounds=n_rounds, 
                            n_epochs=n_epochs, 
                            batch_size=batch_size, 
                            client_mu = 0.1,
                            model=MODEL, 
                            optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
                            loss_fn=nn.CrossEntropyLoss(), 
                            elegibility_percentage=elegibility_percentage,
                            seed=seed)
        fl_algo.init_parties(data_splitter, logger)

    elif algorithm == 'scaffold':
        fl_algo = SCAFFOLD(n_clients=n_clients,
                             n_rounds=n_rounds, 
                             n_epochs=n_epochs, 
                             batch_size=batch_size, 
                             model=MODEL, 
                             optimizer_cfg=OptimizerConfigurator(ScaffoldOptimizer, lr=0.01), 
                             loss_fn=nn.CrossEntropyLoss(), 
                             elegibility_percentage=elegibility_percentage,
                             seed=seed)
        fl_algo.init_parties(data_splitter, global_step=1, callback=logger)

    elif algorithm == 'flhalf':
        fl_algo = FLHalf(n_clients=n_clients,
                         n_rounds=n_rounds, 
                         client_n_epochs=n_epochs, 
                         server_n_epochs=2,
                         client_batch_size=batch_size, 
                         server_batch_size=batch_size, 
                         model=MODEL, 
                         server_optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
                         client_optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.05), 
                         loss_fn=nn.CrossEntropyLoss(), 
                         private_layers=["fc1"],
                         elegibility_percentage=elegibility_percentage,
                         seed=seed)
        fl_algo.init_parties(data_splitter, logger)
    
    else:
        raise ValueError(f'Algorithm {algorithm} not supported')
    
    fl_algo.run(10)
    logger.save(f'./log/{algorithm}_{distribution}.json')

# data_splitter = DataSplitter(train_data.data / 255., 
#                              train_data.targets, 
#                              n_clients=N_CLIENTS, 
#                              distribution=Distribution.LABEL_PATHOLOGICAL_SKEWED, 
#                              batch_size=BATCH_SIZE)

# fedavg = FedAVG(n_clients=N_CLIENTS,
#                 n_rounds=N_ROUNDS, 
#                 n_epochs=N_EPOCHS, 
#                 batch_size=BATCH_SIZE, 
#                 model=MODEL, 
#                 optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
#                 loss_fn=nn.CrossEntropyLoss(), 
#                 elegibility_percentage=ELIGIBILITY_PERCENTAGE,
#                 seed=42)

# fedavg.init_parties(data_splitter, logger)
# fedavg.run(10)

# logger.save('./log/fedavg_noniid_path.json')

# scaffold = SCAFFOLD(n_clients=N_CLIENTS,
#                     n_rounds=N_ROUNDS, 
#                     n_epochs=N_EPOCHS, 
#                     batch_size=BATCH_SIZE, 
#                     model=MLP(), 
#                     optimizer_cfg=OptimizerConfigurator(ScaffoldOptimizer, lr=0.01), 
#                     loss_fn=nn.CrossEntropyLoss(), 
#                     elegibility_percentage=ELIGIBILITY_PERCENTAGE,
#                     seed=42)

# scaffold.init_parties(data_splitter, global_step=1, callback=logger)
# scaffold.run(10)

# logger.save('./log/scaffold_noniid_path.json')


# fedprox = FedProx(n_clients=N_CLIENTS,
#        n_rounds=N_ROUNDS, 
#        n_epochs=N_EPOCHS, 
#        batch_size=BATCH_SIZE, 
#        model=MODEL, 
#        client_mu = 0.1,
#        optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
#        loss_fn=nn.CrossEntropyLoss(), 
#        elegibility_percentage=ELIGIBILITY_PERCENTAGE,
#        seed=42)

# fedprox.init_parties(data_splitter, logger)
# fedprox.run(10)

# logger.save('./log/fedprox_noniid_dir.json')

# flhalf = FLHalf(n_clients=N_CLIENTS,
#                 n_rounds=N_ROUNDS, 
#                 client_n_epochs=N_EPOCHS, 
#                 server_n_epochs=2,
#                 client_batch_size=BATCH_SIZE, 
#                 server_batch_size=BATCH_SIZE, 
#                 model=MODEL, 
#                 server_optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
#                 client_optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.05), 
#                 loss_fn=nn.CrossEntropyLoss(), 
#                 private_layers=["fc1"],
#                 elegibility_percentage=ELIGIBILITY_PERCENTAGE,
#                 seed=42)

# flhalf.init_parties(data_splitter, logger)
# flhalf.run(10)

# logger.save('./log/flhalf_noniid_dir.json')

@app.command()
def compare(algorithms: str=typer.Argument(..., help='Algorithms to compare'),
            distribution: int=typer.Option(Distribution.IID.value, help='Data distribution'),
            show_loss: bool=typer.Option(True, help='Show loss graph')):

    algorithms = algorithms.split(',')
    paths = [f'./log/{algorithm}_{distribution}.json' for algorithm in algorithms]
    plot_comparison(*paths, show_loss=show_loss)

# compare('./log/flhalf_noniid_dir.json', './log/fedavg_noniid_dir.json', show_loss=True) 


if __name__ == '__main__':
    app()