import torch
import torchvision
import torch.nn as nn
from torch.functional import F

# NOTE: The method `forward_` should return the output of the last layer before the global part of the network

class MLP(nn.Module):
    """Three-layer perceptron (MLP) model.
    Each layer is fully connected with 50 hidden units and ReLU activation.
    The size of the input layer is ``input_size`` and the size of the output layer is ``num_classes``.
    The output layer uses a softmax activation.

    Parameters
    ----------
    input_size : int
        Size of the input layer.
    num_classes : int
        Size of the output layer.
    """
    def __init__(self, input_size: int=28*28, num_classes: int=10):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 50)
        #self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        #self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
    
    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        """Partial forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Intermediate representation of the input tensor.
        """
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        return x


class MLP_BN(nn.Module):
    """Three-layer perceptron (MLP) model with batch normalization.
    Each layer is fully connected with 50 hidden units and bith batch normalization and ReLU activation.
    The size of the input layer is ``input_size`` and the size of the output layer is ``num_classes``.
    The output layer uses a softmax activation.

    Parameters
    ----------
    input_size : int
        Size of the input layer.
    num_classes : int
        Size of the output layer.
    """
    def __init__(self, input_size: int=28*28, num_classes: int=10):
        super(MLP_BN, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torchvision.transforms.functional.rgb_to_grayscale(x, num_output_channels=1)
        x = x.reshape(x.shape[0], self.input_size)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return F.log_softmax(self.fc3(x), dim=1)
    
    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        """Partial forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Intermediate representation of the input tensor.
        """
        x = torchvision.transforms.functional.rgb_to_grayscale(x, num_output_channels=1)
        x = x.reshape(x.shape[0], self.input_size)
        x = F.relu(self.bn1(self.fc1(x)))
        return x


class DigitModel(nn.Module):
    """Model for benchmark experiment on Digits. 

    It is a convolutional neural network with 3 convolutional layers and 3 fully connected layers.
    Each layer uses ReLU activation and batch normalization.
    The size of the output layer is ``num_classes`` and it uses a softmax activation.

    Parameters
    ----------
    num_classes : int
        Number of classes, i.e., the size of the output layer.
    """
    def __init__(self, num_classes: int=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(8192, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    


class CNN(nn.Module):
    """Convolutional neural network (CNN) model for single-channel images.

    It is a CNN with 2 convolutional layers and 1 fully connected layer.
    Each convolutional layer uses ReLU activation and max pooling.
    The size of the output layer is 10 and it uses a softmax activation.
    """
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        return F.log_softmax(self.out(x), dim=1) #, x    # return x for visualization