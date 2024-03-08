import torch
import torch.nn as nn
from torch.functional import F
from torchvision.models import resnet50, resnet18, resnet34


#############################################
# MNIST networks
#############################################

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


# FedProx: https://openreview.net/pdf?id=SkgwE5Ss3N
# Logistic Regression
class MNIST_LR(nn.Module):
    def __init__(self, num_classes: int=10):
        super(MNIST_LR, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(784, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 784)
        return F.softmax(self.fc(x), dim=1)


class MLP_E(nn.Module):
    def __init__(self, input_size: int=28*28, output_size: int=10):
        super(MLP_E, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 100)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class MLP(nn.Module):
    """Three-layer perceptron (MLP) model.
    Each layer is fully connected with 50 hidden units and ReLU activation.
    The size of the input layer is ``input_size`` and the size of the output layer is ``output_size``.
    The output layer uses a softmax activation.

    Parameters
    ----------
    input_size : int
        Size of the input layer.
    output_size : int
        Size of the output layer.
    """
    def __init__(self, input_size: int=28*28, output_size: int=10):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fed_E = MLP_E(input_size, output_size)
        self.fc3 = nn.Linear(100, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fed_E(x)
        return F.log_softmax(self.fc3(x), dim=1)


class FedDiselMLP(nn.Module):
    def __init__(self, input_size: int=28*28, output_size: int=10):
        super(FedDiselMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fed_E = MLP_E(input_size, output_size)
        self.private_E = MLP_E(input_size, output_size)
        self.downstream = nn.Linear(100*2, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_p = self.private_E(x)
        x_f = self.fed_E(x)
        emb = torch.cat((x_p, x_f), 1)
        return F.log_softmax(self.downstream(emb), dim=1)


class MLP_BN(nn.Module):
    """Three-layer perceptron (MLP) model with batch normalization.
    Each layer is fully connected with 50 hidden units and bith batch normalization and ReLU activation.
    The size of the input layer is ``input_size`` and the size of the output layer is ``output_size``.
    The output layer uses a softmax activation.

    Parameters
    ----------
    input_size : int
        Size of the input layer.
    output_size : int
        Size of the output layer.
    """
    def __init__(self, input_size: int=28*28, output_size: int=10):
        super(MLP_BN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = torchvision.transforms.functional.rgb_to_grayscale(x, num_output_channels=1)
        #x = x.reshape(x.shape[0], self.input_size)
        x = x.view(-1, self.input_size)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return F.log_softmax(self.fc3(x), dim=1)


class FLHalf_E(nn.Module):
    def __init__(self, input_size=28*28, output_size=64):
        super(FLHalf_E, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)#x.view(-1, 28*28))


class FLHalf_D(nn.Module):
    def __init__(self, input_size=64, output_size=10):
        super(FLHalf_D, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Decoder
        self.downstream = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.downstream(x)

class EDModule(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(EDModule, self).__init__()
        self.E = encoder
        self.D = decoder

    def forward(self, x):
        x = self.E(x)
        x = self.D(x)
        return x

class FLHalf_F(EDModule):
    def __init__(self, input_size=28*28, output_size=10):
        super(FLHalf_F, self).__init__(FLHalf_E(), FLHalf_D())
        self.input_size = input_size
        self.output_size = output_size
    
    def init(self):
        pass


class FLHalf_Dprime(nn.Module):
    def __init__(self, input_size=100, output_size=10):
        super(FLHalf_Dprime, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Decoder
        self.downstream = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.downstream(x)

class FedDiselNet_E(nn.Module):
    
    def __init__(self, input_size=784, output_size=64):
        super(FedDiselNet_E, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Decoder
        self.fed_E = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fed_E(x)


class FedDiselNet(nn.Module):

    def __init__(self, input_size=784, output_size=10):
        super(FedDiselNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Decoder
        self.private_E = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Decoder
        self.fed_E = FedDiselNet_E()

        # Decoder
        self.downstream = nn.Sequential(
            nn.Linear(64*2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x_p = self.private_E(x)
        x_f = self.fed_E(x)
        pred = self.downstream(torch.cat((x_p, x_f), 1))
        return pred


class VGG9_E(nn.Module):

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, seed=0):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, stride=stride, bias=bias)
        torch.manual_seed(seed); torch.nn.init.xavier_normal_(conv.weight)
        return conv
    
    def _linear_layer(self, in_features, out_features, bias=False, seed=0):
        fc = nn.Linear(in_features, out_features, bias=bias)
        torch.manual_seed(seed); torch.nn.init.xavier_normal_(fc.weight)
        return fc
    
    def __init__(self, input_size: int=784, output_size: int=62, seed: int=98765):
        super(VGG9_E, self).__init__()
        self._seed = seed
        self.input_size = input_size
        self.output_size = output_size
        self.fed_E = nn.Sequential(
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
        return self.fed_E(x)


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

        self.fed_E = VGG9_E(input_size, output_size, seed)
        self.downstream = nn.Sequential(
            nn.Flatten(),
            VGG9._linear_layer(in_features=512, out_features=256, bias=False, seed=seed),
            nn.ReLU(True),
            VGG9._linear_layer(in_features=256, out_features=output_size, bias=False, seed=seed)
        )

    def forward(self, x):
        x = self.fed_E(x)
        x = self.downstream(x)
        return x


class FedDiselVGG9(nn.Module):
    def __init__(self, input_size=784, output_size=62):
        super(FedDiselVGG9, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Encoder private
        self.private_E = VGG9_E()

        # Encoder fed
        self.fed_E = VGG9_E()

        # Decoder
        self.downstream = nn.Sequential(
            nn.Flatten(),
            VGG9._linear_layer(in_features=512*2, out_features=256, bias=False),
            nn.ReLU(True),
            VGG9._linear_layer(in_features=256, out_features=output_size, bias=False)
        )

    def forward(self, x):
        x_p = self.private_E(x)
        x_f = self.fed_E(x)
        emb = torch.cat((x_p, x_f), 1)
        pred = self.downstream(emb)
        return pred

    # NEW NETWORKS

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

