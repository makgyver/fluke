import torch
import torchvision
import torch.nn as nn
from torch.functional import F

# NOTE: (FLHAlf) The method `forward_` should return the output of the last layer before the global part of the network

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
        self.output_size = num_classes
        self.fc1 = nn.Linear(input_size, 200)
        #self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(200, 100)
        #self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, num_classes)

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
        self.output_size = num_classes
        self.fc1 = nn.Linear(input_size, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = torchvision.transforms.functional.rgb_to_grayscale(x, num_output_channels=1)
        #x = x.reshape(x.shape[0], self.input_size)
        x = x.view(-1, self.input_size)
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

# 2 convolutional layers with 64 5 × 5 filters, 2 fully connected hidden layers contains 394 and 192 neurons followed by a softmax layer)

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
        self.output_size = num_classes
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        #self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        #self.bn3 = nn.BatchNorm2d(128)
    
        #self.fc1 = nn.Linear(8192, 2048)
        self.fc1 = nn.Linear(4096, 394)
        self.bn4 = nn.BatchNorm1d(394)
        self.fc2 = nn.Linear(394, 192)
        self.bn5 = nn.BatchNorm1d(192)
        self.fc3 = nn.Linear(192, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        #x = F.relu(self.bn3(self.conv3(x)))

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
        self.output_size = 10
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        return F.log_softmax(self.out(x), dim=1) #, x    # return x for visualization


class Block(nn.Module):
    """Residual block for ResNet18 according to https://arxiv.org/pdf/2003.00295.pdf.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    identity_downsample: nn.Module, optional
        Downsample the identity tensor if the number of channels or spatial dimensions are different.
    stride : int, optional
        Stride of the convolutional layers, by default 1
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 identity_downsample: nn.Module=None, 
                 stride: int=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.GroupNorm(2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.GroupNorm(2, out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    """ResNet18 model with group normalization.

    ResNet18 model as described in the paper https://arxiv.org/pdf/2003.00295.pdf.

    Parameters
    ----------
    image_channels : int
        Number of channels in the input image.
    num_classes : int
        Size of the output layer.
    """
    def __init__(self, image_channels: int=3, num_classes: int=10):
        
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        #self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.GroupNorm(2, 64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            #nn.BatchNorm2d(out_channels)
            nn.GroupNorm(2, out_channels)
        )