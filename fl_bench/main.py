from pickletools import optimize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.functional import F
from torchvision.transforms import ToTensor
from data import MNISTDataset

from algorithms.fedavg import FedAVG
from algorithms.fedsgd import FedSGD
from algorithms.scaffold import SCAFFOLD, ScaffoldOptimizer
from algorithms.fedprox import FedProx
from algorithms.flhalf import FLHalf

from fl_bench import GlobalSettings
from data import Datasets
from utils import OptimizerConfigurator, Log
from evaluation import ClassificationEval

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

test_loader = DataLoader(test_data, batch_size=100, shuffle=False)
logger = Log(ClassificationEval(test_loader, nn.CrossEntropyLoss()))

# fedavg = FedAVG(n_clients=100,
#        n_rounds=100, 
#        n_epochs=5, 
#        batch_size=225, 
#        train_set=train_data, 
#        model=MLP(), 
#        optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
#        loss_fn=nn.CrossEntropyLoss(), 
#        elegibility_percentage=.5,
#        seed=42)

# fedavg.prepare_data(MNISTDataset, transform=ToTensor())
# fedavg.init_parties(logger)
# fedavg.run(10)

# logger.save('../log/fedavg.json')

# scaffold = SCAFFOLD(n_clients=100,
#        n_rounds=100, 
#        n_epochs=5, 
#        batch_size=225, 
#        train_set=train_data, 
#        model=MLP(), 
#        optimizer_cfg=OptimizerConfigurator(ScaffoldOptimizer, lr=0.01), 
#        loss_fn=nn.CrossEntropyLoss(), 
#        elegibility_percentage=1.,
#        seed=42)

# scaffold.prepare_data(MNISTDataset, transform=ToTensor())
# scaffold.init_parties(global_step=1, callback=logger)
# scaffold.run(10)

# logger.save('../log/scaffold.json')


# fedprox = FedProx(n_clients=100,
#        n_rounds=100, 
#        n_epochs=5, 
#        batch_size=225, 
#        train_set=train_data, 
#        model=MLP(), 
#        client_mu = 0.1,
#        optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
#        loss_fn=nn.CrossEntropyLoss(), 
#        elegibility_percentage=.5,
#        seed=42)

# fedprox.prepare_data(MNISTDataset, transform=ToTensor())
# fedprox.init_parties(logger)
# fedprox.run(10)


flhalf = FLHalf(n_clients=100,
       n_rounds=100, 
       n_epochs=5, 
       batch_size=225, 
       train_set=train_data, 
       model=MLP(), 
       optimizer_cfg=OptimizerConfigurator(torch.optim.SGD, lr=0.01), 
       loss_fn=nn.CrossEntropyLoss(), 
       private_layers=["fc1"],
       elegibility_percentage=.5,
       seed=42)

flhalf.prepare_data(MNISTDataset, transform=ToTensor())
flhalf.init_parties(logger)
flhalf.run(10)

logger.save('../log/flhalf.json')