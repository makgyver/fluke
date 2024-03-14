from abc import abstractmethod
import torch
import string
import torch.nn as nn
from torch.functional import F
from torchvision.models import resnet50, resnet18, resnet34

# FedAvg: https://arxiv.org/pdf/1602.05629.pdf - hidden_size=[200,200], w/o softmax
# SuPerFed - https://arxiv.org/pdf/2109.07628v3.pdf - hidden_size=[200,200], w/o softmax
# pFedMe: https://arxiv.org/pdf/2006.08848.pdf - hidden_size=[100,100], w/ softmax
# FedDyn: https://openreview.net/pdf?id=B7v4QMR6Z9w - hidden_size=[200,100], w/o softmax
class MNIST_2NN(nn.Module):
    def __init__(self, 
                 hidden_size: tuple[int, int]=(200, 200),
                 softmax: bool=True):
        super(MNIST_2NN, self).__init__()
        self.input_size = 28*28
        self.output_size = 10
        self.use_softmax = softmax

        self.fc1 = nn.Linear(28*28, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_softmax:
            return F.softmax(self.fc3(x), dim=1)
        else:
            return self.fc3(x)


# FedAvg: https://arxiv.org/pdf/1602.05629.pdf
# SuPerFed - https://arxiv.org/pdf/2109.07628v3.pdf
# works with 1 channel input - MNIST4D
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.output_size = 10

        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

# FedBN: https://openreview.net/pdf?id=6YEQUn0QICG
class FedBN_CNN(nn.Module):
    def __init__(self, channels: int=1):
        super(FedBN_CNN, self).__init__()
        self.output_size = 10

        self.conv1 = nn.Conv2d(channels, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
        # this layer is erroneously reported in the paper
        self.conv4 = nn.Conv2d(128, 128, 5, 1, 2)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(6272, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 6272)
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        return self.fc3(x)


# FedProx: https://openreview.net/pdf?id=SkgwE5Ss3N (MNIST and FEMNIST)
# Logistic Regression
class MNIST_LR(nn.Module):
    def __init__(self, num_classes: int=10):
        super(MNIST_LR, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(784, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 784)
        return F.softmax(self.fc(x), dim=1)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


# SuPerFed - https://arxiv.org/pdf/2001.01523.pdf (CIFAR-100)
class ResNet9(nn.Module):
    def __init__(self):
        super(ResNet9, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=1024, out_features=100, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out

# DITTO: https://arxiv.org/pdf/2012.04221.pdf (FEMNIST)
class FEMNIST_CNN(nn.Module):
    def __init__(self):
        super(FEMNIST_CNN, self).__init__()
        self.output_size = 62
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 62)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VGG9_E(nn.Module):

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, seed=0):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, stride=stride, bias=bias)
        torch.manual_seed(seed); torch.nn.init.xavier_normal_(conv.weight)
        return conv
    
    def __init__(self, input_size: int=784, output_size: int=62, seed: int=98765):
        super(VGG9_E, self).__init__()
        self._seed = seed
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = nn.Sequential(
            self._conv_layer(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=False, seed=seed), #FIXME
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_layer(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_layer(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            self._conv_layer(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_layer(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            self._conv_layer(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.encoder(x)


# SuPerFed: https://arxiv.org/pdf/2109.07628v3.pdf (FEMNIST)
class VGG9(nn.Module):
    
    @classmethod
    def _linear_layer(cls, in_features, out_features, bias=False, seed=0):
        fc = nn.Linear(in_features, out_features, bias=bias)
        torch.manual_seed(seed); torch.nn.init.xavier_normal_(fc.weight)
        return fc

    def __init__(self, input_size: int=784, output_size: int=62, seed: int=98765):
        super(VGG9, self).__init__()
        self._seed = seed
        self.input_size = input_size
        self.output_size = output_size

        self.encoder = VGG9_E(input_size, output_size, seed)
        self.downstream = nn.Sequential(
            nn.Flatten(),
            VGG9._linear_layer(in_features=512, out_features=256, bias=False, seed=seed),
            nn.ReLU(True),
            VGG9._linear_layer(in_features=256, out_features=output_size, bias=False, seed=seed)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.downstream(x)
        return x

# FedAvg: https://arxiv.org/pdf/1602.05629.pdf (CIFAR-10)
# FedDyn: https://openreview.net/pdf?id=B7v4QMR6Z9w (CIFAR-10 and CIFAR-100)
class FedavgCNN(nn.Module):
    def __init__(self, input_size=(28,28), output_size=10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.local3 = nn.Linear(4096, 384)
        self.local4 = nn.Linear(384, 192)
        self.softmax_linear = nn.Linear(192, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.local3(x))
        x = F.relu(self.local4(x))
        x = self.softmax_linear(x)

        return x

# FedOpt: https://openreview.net/pdf?id=SkgwE5Ss3N (CIFAR-10)
class ResNet18(nn.Module):
    def __init__(self, output_size=10):
        super(ResNet18, self).__init__()
        self.output_size = output_size
        self.resnet = resnet18(num_classes=output_size)
    
    def forward(self, x):
        return self.resnet(x)

# FedPer: https://arxiv.org/pdf/1912.00818.pdf (CIFAR-100)
class ResNet34(nn.Module):
    def __init__(self, output_size=100):
        super(ResNet34, self).__init__()
        self.output_size = output_size
        self.resnet = resnet34(num_classes=output_size)
    
    def forward(self, x):
        return self.resnet(x)

# MOON: https://arxiv.org/pdf/2103.16257.pdf (CIFAR-100)
class ResNet50(nn.Module):
    def __init__(self, output_size=100):
        super(ResNet50, self).__init__()
        self.output_size = output_size
        self.resnet = resnet50(num_classes=output_size)
    
    def forward(self, x):
        return self.resnet(x)


# FedRep: https://arxiv.org/pdf/2102.07078.pdf (CIFAR-100 and CIFAR-10)
# LG-FedAvg: https://arxiv.org/pdf/2001.01523.pdf
class LeNet5(nn.Module):
    def __init__(self, output_size=100):
        super(LeNet5, self).__init__()
        self.output_size = output_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, output_size)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# SuPerFed: https://arxiv.org/pdf/2109.07628v3.pdf (Shakespeare)
class Shakespeare_LSTM(nn.Module):

    def __init__(self, seed=42):
        super(Shakespeare_LSTM, self).__init__()
        self.output_size = len(string.printable)

        self.encoder = nn.Embedding(self.output_size, 8)
        self.rnn = nn.LSTM(
            input_size=8,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bias=False
        )
        self.classifier = VGG9._linear_layer(256, self.output_size, bias=False, seed=seed)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, torch.prod(x.size()[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# MOONS: https://arxiv.org/pdf/2103.16257.pdf
class MoonsCNN(nn.Module):
    def __init__(self):
        super(MoonsCNN, self).__init__()
        self.output_size = 10

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.projection_head = nn.Linear(84, 256)
        self.out = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # The paper is not clear about the activation of
        # the projection head (PH). We go with ReLU since they
        # cite https://arxiv.org/pdf/2002.05709.pdf where 
        # it is shown that non-linear PHs works better.
        x = F.relu(self.projection_head(x)) 
        x = self.out(x)
        return x


class GlobalLocalNet(nn.Module):
    """Global-Local Network (Abstract Class)

    A network that has two subnetworks, one is meant to be shared (global) and one is meant to be
    personalized (local). The forward method should work as expected, but the forward_local and
    forward_global methods should be used to get the output of the local and global subnetworks, 
    respectively. If this is not possible, they fallback to the forward method (default behavior).
    """

    @abstractmethod
    def get_local(self):
        """Return the local subnetwork"""
        pass

    @abstractmethod
    def get_global(self):
        """Return the global subnetwork"""
        pass

    def forward_local(self, x):
        return self.forward(x)

    def forward_global(self, x):
        return self.forward(x)


# FedPer: https://arxiv.org/pdf/1912.00818.pdf (FEMNIST - meant to be used by FedPer)
class FedPer_VGG9(GlobalLocalNet, VGG9):

    def get_local(self):
        return self.downstream

    def get_global(self):
        return self.encoder

    def forward_local(self, z):
        return self.downstream(z)

    def forward_global(self, x):
        return self.encoder(x)