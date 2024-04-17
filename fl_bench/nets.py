"""
This module contains the definition of several neural networks used in state-of-the-art
federated learning papers.
"""
from abc import abstractmethod
import string
import torch
import torch.nn as nn
from torch.functional import F
from torchvision.models import resnet50, resnet18, resnet34


class EncoderHeadNet(nn.Module):
    """Encoder+Head Network (Base Class)

    A network that has two subnetworks, one is meant to be the encoder that learns a latent
    representation of the input and the other is meant to be the head that learns to classify.
    The forward method should work as expected, but the `forward_encoder` and
    `forward_head` methods should be used to get the output of the econer and head subnetworks,
    respectively. If this is not possible, they fallback to the forward method (default behavior).

    Attributes:
        E (nn.Module): Encoder subnetwork.
        D (nn.Module): Head subnetwork.
        output_size (int): Output size of the head subnetwork.

    Args:
        encoder (nn.Module): Encoder subnetwork.
        head (nn.Module): Head subnetwork.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super(EncoderHeadNet, self).__init__()
        self.output_size = head.output_size
        self._encoder = encoder
        self._head = head

    @property
    def encoder(self) -> nn.Module:
        """Return the encoder subnetwork"""
        return self._encoder

    @property
    def head(self) -> nn.Module:
        """Return the global subnetwork"""
        return self._head

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder subnetwork.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the encoder subnetwork.
        """
        return self._encoder(x)

    def forward_head(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the head subnetwork.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the head subnetwork.
        """
        return self._head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(self._encoder(x))


class MNIST_2NN_E(nn.Module):
    def __init__(self,
                 hidden_size: tuple[int, int] = (200, 200)):
        super(MNIST_2NN_E, self).__init__()
        self.input_size = 28*28
        self.output_size = 10

        self.fc1 = nn.Linear(28*28, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MNIST_2NN_D(nn.Module):
    def __init__(self,
                 hidden_size: int = 200,
                 use_softmax: bool = False):
        super(MNIST_2NN_D, self).__init__()
        self.output_size = 10
        self.use_softmax = use_softmax
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_softmax:
            return F.softmax(self.fc3(x), dim=1)
        else:
            return torch.sigmoid(self.fc3(x))


# FedAvg: https://arxiv.org/pdf/1602.05629.pdf - hidden_size=[200,200], w/o softmax
# SuPerFed - https://arxiv.org/pdf/2109.07628v3.pdf - hidden_size=[200,200], w/o softmax
# pFedMe: https://arxiv.org/pdf/2006.08848.pdf - hidden_size=[100,100], w/ softmax
# FedDyn: https://openreview.net/pdf?id=B7v4QMR6Z9w - hidden_size=[200,100], w/o softmax
class MNIST_2NN(EncoderHeadNet):
    """Multi-layer Perceptron for MNIST.

    2-layer neural network for MNIST classification first introduced in the paper
    "Communication-Efficient Learning of Deep Networks from Decentralized Data" by
    H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas, where
    the hidden layers have 200 neurons each and the output layer with sigmoid activation.

    Similar architectures are also used in other papers:
    - SuPerFed: https://arxiv.org/pdf/2109.07628v3.pdf - `hidden_size=(200, 200)`
    - pFedMe: https://arxiv.org/pdf/2006.08848.pdf - `hidden_size=(100, 100)` with softmax layer
    - FedDyn: https://openreview.net/pdf?id=B7v4QMR6Z9w - `hidden_size=(200, 100)`

    Args:
        hidden_size (tuple[int, int], optional): Size of the hidden layers. Defaults to (200, 200).
        softmax (bool, optional): If True, the output is passed through a softmax layer.
          Defaults to True.
    """

    def __init__(self,
                 hidden_size: tuple[int, int] = (200, 200),
                 softmax: bool = False):
        super(MNIST_2NN, self).__init__(
            MNIST_2NN_E(hidden_size),
            MNIST_2NN_D(hidden_size[1], softmax)
        )


class MNIST_CNN_E(nn.Module):
    def __init__(self):
        super(MNIST_CNN_E, self).__init__()
        self.output_size = 1024
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x.view(-1, 1024)


class MNIST_CNN_D(nn.Module):
    def __init__(self):
        super(MNIST_CNN_D, self).__init__()
        self.output_size = 10
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


# FedAvg: https://arxiv.org/pdf/1602.05629.pdf
# SuPerFed - https://arxiv.org/pdf/2109.07628v3.pdf
# works with 1 channel input - MNIST4D
class MNIST_CNN(EncoderHeadNet):
    def __init__(self):
        super(MNIST_CNN, self).__init__(MNIST_CNN_E(), MNIST_CNN_D())


class FedBN_CNN_E(nn.Module):
    def __init__(self, channels: int = 1):
        super(FedBN_CNN_E, self).__init__()
        self.output_size = 6272

        self.conv1 = nn.Conv2d(channels, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
        # this layer is erroneously reported in the paper
        self.conv4 = nn.Conv2d(128, 128, 5, 1, 2)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x.view(-1, 6272)


class FedBN_CNN_D(nn.Module):
    def __init__(self):
        super(FedBN_CNN_D, self).__init__()
        self.output_size = 10
        self.fc1 = nn.Linear(6272, 2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn5(self.fc1(x)))
        x = F.relu(self.bn6(self.fc2(x)))
        return self.fc3(x)


# FedBN: https://openreview.net/pdf?id=6YEQUn0QICG
class FedBN_CNN(EncoderHeadNet):
    def __init__(self, channels: int = 1):
        super(FedBN_CNN, self).__init__(FedBN_CNN_E(channels), FedBN_CNN_D())


# FedProx: https://openreview.net/pdf?id=SkgwE5Ss3N (MNIST and FEMNIST)
# Logistic Regression
class MNIST_LR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(MNIST_LR, self).__init__()
        self.output_size = num_classes
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x) -> torch.Tensor:
        x = x.view(-1, 784)
        return F.softmax(self.fc(x), dim=1)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class ResNet9_E(nn.Module):
    def __init__(self):
        super(ResNet9_E, self).__init__()
        self.output_size = 1024
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x) -> torch.Tensor:
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        return out


class ResNet9_D(nn.Module):
    def __init__(self):
        super(ResNet9_D, self).__init__()
        self.output_size = 100
        self.fc = nn.Linear(in_features=1024, out_features=100, bias=True)

    def forward(self, x) -> torch.Tensor:
        out = self.fc(x)
        return out


# SuPerFed - https://arxiv.org/pdf/2001.01523.pdf (CIFAR-100)
class ResNet9(EncoderHeadNet):
    def __init__(self):
        super(ResNet9, self).__init__(ResNet9_E(), ResNet9_D())


# DITTO: https://arxiv.org/pdf/2012.04221.pdf (FEMNIST)
class FEMNIST_CNN_E(nn.Module):
    def __init__(self):
        super(FEMNIST_CNN_E, self).__init__()
        self.output_size = 3136
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x.view(-1, 7 * 7 * 64)


class FEMNIST_CNN_D(nn.Module):
    def __init__(self):
        super(FEMNIST_CNN_D, self).__init__()
        self.output_size = 62
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 62)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FEMNIST_CNN(EncoderHeadNet):
    def __init__(self):
        super(FEMNIST_CNN, self).__init__(FEMNIST_CNN_E(), FEMNIST_CNN_D())


class VGG9_E(nn.Module):

    def _conv_layer(self,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    groups=1,
                    bias=False,
                    seed=0) -> nn.Conv2d:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         padding=padding, groups=groups, stride=stride, bias=bias)
        torch.manual_seed(seed)
        torch.nn.init.xavier_normal_(conv.weight)
        return conv

    def __init__(self, input_size: int = 784, output_size: int = 62, seed: int = 98765):
        super(VGG9_E, self).__init__()
        self._seed = seed
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = nn.Sequential(
            self._conv_layer(in_channels=1, out_channels=16, kernel_size=3,
                             padding=1, bias=False, seed=seed),  # FIXME
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_layer(in_channels=16, out_channels=32, kernel_size=3,
                             padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_layer(in_channels=32, out_channels=64, kernel_size=3,
                             padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            self._conv_layer(in_channels=64, out_channels=128, kernel_size=3,
                             padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_layer(in_channels=128, out_channels=256,
                             kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            self._conv_layer(in_channels=256, out_channels=512,
                             kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

    def forward(self, x) -> torch.Tensor:
        return self.encoder(x)


class VGG9_D(nn.Module):

    @classmethod
    def _linear_layer(cls, in_features, out_features, bias=False, seed=0):
        fc = nn.Linear(in_features, out_features, bias=bias)
        torch.manual_seed(seed)
        torch.nn.init.xavier_normal_(fc.weight)
        return fc

    def __init__(self, input_size: int = 512, output_size: int = 62, seed: int = 98765):
        super(VGG9_D, self).__init__()
        self.output_size = output_size
        self.downstream = nn.Sequential(
            VGG9_D._linear_layer(in_features=input_size, out_features=256, bias=False, seed=seed),
            nn.ReLU(True),
            VGG9_D._linear_layer(in_features=256, out_features=output_size, bias=False, seed=seed)
        )

    def forward(self, x) -> torch.Tensor:
        return self.downstream(x)


# SuPerFed: https://arxiv.org/pdf/2109.07628v3.pdf (FEMNIST)
class VGG9(EncoderHeadNet):

    def __init__(self, input_size: int = 784, output_size: int = 62, seed: int = 98765):
        super(VGG9, self).__init__(
            VGG9_E(input_size, output_size, seed),
            VGG9_D(input_size=512, output_size=output_size, seed=seed)
        )


# FedAvg: https://arxiv.org/pdf/1602.05629.pdf (CIFAR-10)
# FedDyn: https://openreview.net/pdf?id=B7v4QMR6Z9w (CIFAR-10 and CIFAR-100)
class FedavgCNN_E(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 4096

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.norm1 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        # self.norm2 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # x = self.norm1(x)

        x = F.relu(self.conv2(x))
        # x = self.norm2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        return x


class FedavgCNN_D(nn.Module):
    def __init__(self, output_size: int = 10):
        super().__init__()
        self.output_size = output_size
        self.local3 = nn.Linear(4096, 384)
        self.local4 = nn.Linear(384, 192)
        self.linear = nn.Linear(192, output_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.local3(x))
        x = F.relu(self.local4(x))
        x = self.linear(x)
        return F.softmax(x, dim=1)


class FedavgCNN(EncoderHeadNet):
    def __init__(self, output_size=10):
        super(FedavgCNN, self).__init__(FedavgCNN_E(),
                                        FedavgCNN_D(output_size))


# FedOpt: https://openreview.net/pdf?id=SkgwE5Ss3N (CIFAR-10)
class ResNet18(nn.Module):
    def __init__(self, output_size=10):
        super(ResNet18, self).__init__()
        self.output_size = output_size
        self.resnet = resnet18(num_classes=output_size)

    def forward(self, x) -> torch.Tensor:
        return self.resnet(x)


# FedPer: https://arxiv.org/pdf/1912.00818.pdf (CIFAR-100)
class ResNet34(nn.Module):
    def __init__(self, output_size=100):
        super(ResNet34, self).__init__()
        self.output_size = output_size
        self.resnet = resnet34(num_classes=output_size)

    def forward(self, x) -> torch.Tensor:
        return self.resnet(x)


# MOON: https://arxiv.org/pdf/2103.16257.pdf (CIFAR-100)
class ResNet50(nn.Module):
    def __init__(self, output_size=100):
        super(ResNet50, self).__init__()
        self.output_size = output_size
        self.resnet = resnet50(num_classes=output_size)

    def forward(self, x) -> torch.Tensor:
        return self.resnet(x)


class LeNet5_E(nn.Module):
    # Expected input size: 32x32x3
    def __init__(self):
        super(LeNet5_E, self).__init__()
        self.output_size = 400
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        return out


class LeNet5_D(nn.Module):
    def __init__(self, output_size=100):
        super(LeNet5_D, self).__init__()
        self.output_size = output_size
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, output_size)

    def forward(self, x) -> torch.Tensor:
        out = self.fc(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


# FedRep: https://arxiv.org/pdf/2102.07078.pdf (CIFAR-100 and CIFAR-10)
# LG-FedAvg: https://arxiv.org/pdf/2001.01523.pdf
class LeNet5(EncoderHeadNet):
    def __init__(self, output_size=100):
        super(LeNet5, self).__init__(LeNet5_E(), LeNet5_D(output_size))


# SuPerFed: https://arxiv.org/pdf/2109.07628v3.pdf (Shakespeare)
class Shakespeare_LSTM_E(nn.Module):

    def __init__(self):
        super(Shakespeare_LSTM_E, self).__init__()
        self.output_size = 256
        self.encoder = nn.Embedding(self.output_size, 8)
        self.rnn = nn.LSTM(
            input_size=8,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bias=False
        )

    def forward(self, x) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, torch.prod(x.size()[1:]))
        return x


class Shakespeare_LSTM_D(nn.Module):

    def __init__(self, seed: int):
        super(Shakespeare_LSTM_D, self).__init__()
        self.output_size = len(string.printable)
        self.classifier = VGG9._linear_layer(256, self.output_size, bias=False, seed=seed)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Shakespeare_LSTM(EncoderHeadNet):

    def __init__(self, seed: int = 42):
        super(Shakespeare_LSTM, self).__init__(Shakespeare_LSTM_E(), Shakespeare_LSTM_D(seed))


class MoonCNN_E(nn.Module):
    # Expected input size: 32x32x3
    def __init__(self):
        super(MoonCNN_E, self).__init__()
        self.output_size = 400

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        return x


class MoonCNN_D(nn.Module):
    def __init__(self):
        super(MoonCNN_D, self).__init__()
        self.output_size = 10
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.projection_head = nn.Linear(84, 256)
        self.out = nn.Linear(256, self.output_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # The paper is not clear about the activation of
        # the projection head (PH). We go with ReLU since they
        # cite https://arxiv.org/pdf/2002.05709.pdf where
        # it is shown that non-linear PHs works better.
        x = F.relu(self.projection_head(x))
        x = self.out(x)
        return x


# MOON: https://arxiv.org/pdf/2103.16257.pdf (CIFAR10)
class MoonCNN(EncoderHeadNet):
    def __init__(self):
        super(MoonCNN, self).__init__(MoonCNN_E(), MoonCNN_D())


class SimpleCNN_E(nn.Module):
    # Expected input size: 32x32x3
    def __init__(self, hidden_dims=(100, 100), output_dim=10):
        super(SimpleCNN_E, self).__init__()
        self.output_size = 400
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x


class SimpleCNN_D(nn.Module):
    def __init__(self, hidden_dims=(100, 100), output_dim=10):
        super(SimpleCNN_D, self).__init__()
        self.output_size = output_dim
        self.fc1 = nn.Linear(16*5*5, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(EncoderHeadNet):
    def __init__(self, hidden_dims=(100, 100), output_dim=10):
        super(SimpleCNN, self).__init__(SimpleCNN_E(), SimpleCNN_D(hidden_dims, output_dim))


class GlobalLocalNet(nn.Module):
    """Global-Local Network (Abstract Class)

    A network that has two subnetworks, one is meant to be shared (global) and one is meant to be
    personalized (local). The forward method should work as expected, but the forward_local and
    forward_global methods should be used to get the output of the local and global subnetworks,
    respectively. If this is not possible, they fallback to the forward method (default behavior).
    """

    @abstractmethod
    def get_local(self) -> nn.Module:
        """Return the local subnetwork"""
        pass

    @abstractmethod
    def get_global(self) -> nn.Module:
        """Return the global subnetwork"""
        pass

    def forward_local(self, x) -> torch.Tensor:
        return self.get_local()(x)

    def forward_global(self, x) -> torch.Tensor:
        return self.get_global()(x)


# FedPer: https://arxiv.org/pdf/1912.00818.pdf (FEMNIST - meant to be used by FedPer)
class FedPer_VGG9(GlobalLocalNet, VGG9):

    def get_local(self) -> nn.Module:
        return self._head

    def get_global(self) -> nn.Module:
        return self._encoder


class LG_FedAvg_VGG9(GlobalLocalNet, VGG9):

    def get_local(self) -> nn.Module:
        return self._encoder

    def get_global(self) -> nn.Module:
        return self._head


class MNIST_2NN_GlobalD(GlobalLocalNet, MNIST_2NN):

    def get_local(self) -> nn.Module:
        return self._encoder

    def get_global(self) -> nn.Module:
        return self._head


class MNIST_2NN_GlobalE(GlobalLocalNet, MNIST_2NN):

    def get_local(self) -> nn.Module:
        return self._head

    def get_global(self) -> nn.Module:
        return self._encoder
