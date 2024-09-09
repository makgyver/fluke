"""
This module contains the definition of several neural networks used in state-of-the-art
federated learning papers.
"""
import string
import sys
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.functional import F
from torchvision.models import resnet18, resnet34, resnet50

sys.path.append(".")
sys.path.append("..")

from . import GlobalSettings  # NOQA
from .utils.model import batch_norm_to_group_norm  # NOQA

__all__ = [
    'EncoderHeadNet',
    'GlobalLocalNet',
    'HeadGlobalEncoderLocalNet',
    'EncoderGlobalHeadLocalNet',
    'MNIST_2NN',
    'MNIST_2NN_E',
    'MNIST_2NN_D',
    'MNIST_CNN',
    'MNIST_CNN_E',
    'MNIST_CNN_D',
    'FedBN_CNN',
    'FedBN_CNN_E',
    'FedBN_CNN_D',
    'MNIST_LR',
    'CifarConv2',
    'CifarConv2_E',
    'CifarConv2_D',
    'ResNet9',
    'ResNet9_E',
    'ResNet9_D',
    'FEMNIST_CNN',
    'FEMNIST_CNN_E',
    'FEMNIST_CNN_D',
    'VGG9_E',
    'VGG9_D',
    'VGG9',
    'ResNet18',
    'ResNet34',
    'ResNet50',
    'ResNet18GN',
    'MoonCNN_E',
    'MoonCNN_D',
    'MoonCNN',
    'LeNet5_E',
    'LeNet5_D',
    'LeNet5',
    'Shakespeare_LSTM_E',
    'Shakespeare_LSTM_D',
    'Shakespeare_LSTM'
]


class EncoderHeadNet(nn.Module):
    r"""Encoder (aka backbone) + Head Network [Base Class]
    This type of networks are defined as two subnetworks, where one is meant to be the
    encoder/backbone network that learns a latent representation of the input, and the head network
    that is the classifier part of the model. The forward method should work as usual (i.e.,
    :math:`g(f(\mathbf{x}))` where :math:`\mathbf{x}` is the input, :math:`f` is the encoder and
    :math:`g` is the head), but the ``forward_encoder`` and ``forward_head`` methods should be used
    to get the output of the encoder and head subnetworks, respectively.
    If this is not possible, they fallback to the forward method (default behavior).

    Attributes:
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
        """Return the encoder subnetwork.

        Returns:
            nn.Module: Encoder subnetwork.
        """
        return self._encoder

    @property
    def head(self) -> nn.Module:
        """Return the head subnetwork.

        Returns:
            nn.Module: head subnetwork.
        """
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
        """Forward pass through the head subnetwork. ``z`` is assumed to be the output of the
        encoder subnetwork or an "equivalent" tensor.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the head subnetwork.
        """
        return self._head(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._head(self._encoder(x))


class GlobalLocalNet(nn.Module):
    """Global-Local Network (Abstract Class). This is a network that has two subnetworks, one is
    meant to be shared (global) and one is meant to be personalized (local). The ``forward`` method
    should work as expected, but the ``forward_local`` and ``forward_global`` methods should be used
    to get the output of the local and global subnetworks, respectively. If this is not possible,
    they fallback to the forward method (default behavior).
    """

    @abstractmethod
    def get_local(self) -> nn.Module:
        """Return the local subnetwork.

        Returns:
            nn.Module: The local subnetwork
        """
        raise NotImplementedError

    @abstractmethod
    def get_global(self) -> nn.Module:
        """Return the global subnetwork.

        Returns:
            nn.Module: The global subnetwork
        """
        raise NotImplementedError

    def forward_local(self, x) -> torch.Tensor:
        return self.get_local()(x)

    def forward_global(self, x) -> torch.Tensor:
        return self.get_global()(x)


class EncoderGlobalHeadLocalNet(GlobalLocalNet):
    """This implementation of the Global-Local Network (:class:`GlobalLocalNet`) is meant to be used
    with the Encoder-Head architecture. The global (i.e., that is shared between clients and server)
    subnetwork is the encoder and the local (i.e., not shared) subnetwork is the head.

    Args:
        model (EncoderHeadNet): The federated model to use.

    See Also:
        - :class:`EncoderHeadNet`
        - :class:`GlobalLocalNet`
        - :class:`HeadGlobalEncoderLocalNet`
    """

    def __init__(self, model: EncoderHeadNet):
        assert isinstance(model, EncoderHeadNet), "model must be an EncoderHeadNet."
        super(EncoderGlobalHeadLocalNet, self).__init__()
        self.model = model

    def get_local(self) -> nn.Module:
        return self.model.head

    def get_global(self) -> nn.Module:
        return self.model.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HeadGlobalEncoderLocalNet(GlobalLocalNet):
    """This implementation of the Global-Local Network (:class:`GlobalLocalNet`) is meant to be used
    with the Encoder-Head architecture. The global (i.e., that is shared between clients and server)
    subnetwork is the head and the local (i.e., not shared) subnetwork is the encoder.

    Args:
        model (EncoderHeadNet): The federated model to use.

    See Also:
        - :class:`EncoderHeadNet`
        - :class:`GlobalLocalNet`
        - :class:`EncoderGlobalHeadLocalNet`
    """

    def __init__(self, model: EncoderHeadNet):
        assert isinstance(model, EncoderHeadNet), "model must be an EncoderHeadNet."
        super(HeadGlobalEncoderLocalNet, self).__init__()
        self.model = model

    def get_local(self) -> nn.Module:
        return self.model.encoder

    def get_global(self) -> nn.Module:
        return self.model.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MNIST_2NN_E(nn.Module):
    """Encoder for the :class:`MNIST_2NN` network.

    Args:
        hidden_size (tuple[int, int], optional): Size of the hidden layers. Defaults to (200, 200).

    See Also:
        - :class:`MNIST_2NN`
        - :class:`MNIST_2NN_D`
    """

    def __init__(self,
                 hidden_size: tuple[int, int] = (200, 100)):
        super(MNIST_2NN_E, self).__init__()
        self.input_size = 28*28
        self.output_size = hidden_size[1]

        self.fc1 = nn.Linear(28*28, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MNIST_2NN_D(nn.Module):
    """Head for the :class:`MNIST_2NN` network.

    Args:
        hidden_size (int, optional): Size of the hidden layer. Defaults to 200.
        use_softmax (bool, optional): If True, the output is passed through a softmax layer,
            otherwise, a sigmoid activation is used. Defaults to False.

    See Also:
        - :class:`MNIST_2NN`
        - :class:`MNIST_2NN_E`
    """

    def __init__(self,
                 hidden_size: int = 100,
                 use_softmax: bool = False):
        super(MNIST_2NN_D, self).__init__()
        self.output_size = 10
        self.use_softmax = use_softmax
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_softmax:
            return F.softmax(self.fc3(x), dim=1)
        else:
            # return torch.sigmoid(self.fc3(x))
            return self.fc3(x)


# FedAvg: https://arxiv.org/pdf/1602.05629.pdf - hidden_size=[200,200], w/o softmax
# SuPerFed - https://arxiv.org/pdf/2109.07628v3.pdf - hidden_size=[200,200], w/o softmax
# pFedMe: https://arxiv.org/pdf/2006.08848.pdf - hidden_size=[100,100], w/ softmax
# FedDyn: https://openreview.net/pdf?id=B7v4QMR6Z9w - hidden_size=[200,100], w/o softmax
class MNIST_2NN(EncoderHeadNet):
    """Multi-layer Perceptron for MNIST. This is a
    2-layer neural network for MNIST classification first introduced in the [FedAvg]_ paper,
    where the hidden layers have 200 neurons each and the output layer with sigmoid activation.

    Similar architectures are also used in other papers:

    - [SuPerFed]_: ``hidden_size=(200, 200)``, same as FedAvg;
    - [pFedMe]_: ``hidden_size=(100, 100)`` with softmax on the output layer;
    - [FedDyn]_: ``hidden_size=(200, 100)``.

    Args:
        hidden_size (tuple[int, int], optional): Size of the hidden layers. Defaults to (200, 200).
        softmax (bool, optional): If True, the output is passed through a softmax layer.
          Defaults to True.

    See Also:
        - :class:`MNIST_2NN_E`
        - :class:`MNIST_2NN_D`

    References:
        .. [FedAvg] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y
            Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            In AISTATS (2017).
        .. [SuPerFed] Seok-Ju Hahn, Minwoo Jeong, and Junghye Lee. Connecting Low-Loss Subspace for
            Personalized Federated Learning. In KDD (2022).
        .. [pFedMe] Canh T. Dinh, Nguyen H. Tran, and Tuan Dung Nguyen. Personalized Federated
            Learning with Moreau Envelopes. In NeurIPS (2020).
        .. [FedDyn] S. Wang, T. Liu, and M. Hong. "FedDyn: A Dynamic Federated Learning Framework".
            In ICLR (2021).
    """

    def __init__(self,
                 hidden_size: tuple[int, int] = (200, 100),
                 softmax: bool = False):
        super(MNIST_2NN, self).__init__(
            MNIST_2NN_E(hidden_size),
            MNIST_2NN_D(hidden_size[1], softmax)
        )


class MNIST_CNN_E(nn.Module):
    """Encoder for the :class:`MNIST_CNN` network.

    See Also:
        - :class:`MNIST_CNN`
        - :class:`MNIST_CNN_D`
    """

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
    """Head for the :class:`MNIST_CNN` network.

    See Also:
        - :class:`MNIST_CNN`
        - :class:`MNIST_CNN_E`
    """

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
    """Convolutional Neural Network for MNIST. This is a simple CNN for MNIST classification
    first introduced in the [FedAvg]_ paper, where the architecture consists of two convolutional
    layers with 32 and 64 filters, respectively, followed by two fully connected layers with 512
    and 10 neurons, respectively.

    Very same architecture is also used in the [SuPerFed]_ paper.
    """

    def __init__(self):
        super(MNIST_CNN, self).__init__(MNIST_CNN_E(), MNIST_CNN_D())


class FedBN_CNN_E(nn.Module):
    """Encoder for the :class:`FedBN_CNN` network.

    Args:
        channels (int, optional): Number of input channels. Defaults to 1.

    See Also:
        - :class:`FedBN_CNN`
        - :class:`FedBN_CNN_D`
    """

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
    """Head for the :class:`FedBN_CNN` network.

    See Also:
        - :class:`FedBN_CNN`
        - :class:`FedBN_CNN_E`
    """

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
    """Convolutional Neural Network with Batch Normalization for CIFAR-10. This network
    follows the architecture proposed in the [FedBN]_ paper, where the encoder consists of four
    convolutional layers with 64, 64, 128, and 128 filters, respectively, and the head network
    consists of three fully connected layers with 2048, 512, and 10 neurons, respectively.

    Args:
        channels (int, optional): Number of input channels. Defaults to 1.

    Note:
        In the original paper, the size of the last convolutional layer is erroneously reported.

    See Also:
        - :class:`FedBN_CNN_E`
        - :class:`FedBN_CNN_D`

    References:
        .. [FedBN] Xiaoxiao Li, Meirui JIANG, Xiaofei Zhang, Michael Kamp, and Qi Dou. FedBN:
            Federated Learning on Non-IID Features via Local Batch Normalization. In ICLR (2021).
    """

    def __init__(self, channels: int = 1):
        super(FedBN_CNN, self).__init__(FedBN_CNN_E(channels), FedBN_CNN_D())


# FedNH: https://arxiv.org/abs/2212.02758 (CIFAR-10)
class CifarConv2_E(nn.Module):
    """Encoder for the :class:`CifarConv2` network.

    See Also:
        - :class:`CifarConv2`
        - :class:`CifarConv2_D`
    """

    def __init__(self):  # , output_size: int = 1600):
        super().__init__()
        self.output_size = 1600
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.linear1 = nn.Linear(64 * 5 * 5, 512)
        # self.linear2 = nn.Linear(512, self.output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 5 * 5)
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        return x


class CifarConv2_D(nn.Module):
    """Head for the :class:`CifarConv2` network.

    Args:
        input_size (int, optional): Size of the input. Defaults to 100.
        num_classes (int, optional): Number of classes. Defaults to 10.

    See Also:
        - :class:`CifarConv2`
        - :class:`CifarConv2_E`
    """

    def __init__(self):
        super().__init__()
        self.input_size = 1600
        self.output_size = 10
        # self.linear2 = nn.Linear(self.input_size, self.output_size, bias=False)
        self.linear1 = nn.Linear(self.input_size, 512)
        self.linear2 = nn.Linear(512, self.output_size)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.relu(self.bn3(self.linear1(x)))
        logits = self.linear2(x)
        return logits


class CifarConv2(EncoderHeadNet):
    """Convolutional Neural Network for CIFAR-10. This is a CNN for CIFAR-10 classification
    as described in the [FedNH]_ paper, where the architecture consists of two convolutional
    layers with 64 filters, followed by two fully connected layers with 384 and 100 neurons,
    respectively. The convolutional layers are followed by ReLU activations and max pooling.
    The last classification layer is a linear layer with 10 neurons.

    Args:
        embedding_size (int, optional): Size of the embedding after the second linear layer.
            Defaults to 100.
        num_classes (int, optional): Number of classes. Defaults to 10.

    See Also:
        - :class:`CifarConv2_E`
        - :class:`CifarConv2_D`

    References:
        .. [FedNH] Yutong Dai, Zeyuan Chen, Junnan Li, Shelby Heinecke, Lichao Sun, Ran Xu.
            Tackling Data Heterogeneity in Federated Learning with Class Prototypes.
            In AAAI (2023).
    """

    def __init__(self):
        super().__init__(CifarConv2_E(), CifarConv2_D())


# FedProx: https://openreview.net/pdf?id=SkgwE5Ss3N (MNIST and FEMNIST)
# Logistic Regression
class MNIST_LR(nn.Module):
    """Logistic Regression for MNIST. This is a simple logistic regression model for MNIST
    classification used in the [FedProx]_ paper for both MNIST and FEMNIST datasets.

    Args:
        num_classes (int, optional): Number of classes, i.e., the output size. Defaults to 10.

    References:
        .. [FedProx] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and
            Virginia Smith. Federated Optimization in Heterogeneous Networks. Adaptive & Multitask
            Learning Workshop. In Open Review https://openreview.net/pdf?id=SkgwE5Ss3N (2018).
    """

    def __init__(self, num_classes: int = 10):
        super(MNIST_LR, self).__init__()
        self.output_size = num_classes
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x) -> torch.Tensor:
        x = x.view(-1, 784)
        return F.softmax(self.fc(x), dim=1)


class _ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(_ResidualBlock, self).__init__()
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
    """Encoder for the :class:`ResNet9` network.

    See Also:
        - :class:`ResNet9`
        - :class:`ResNet9_D`
    """

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
            _ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
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
            _ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x) -> torch.Tensor:
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        return out


class ResNet9_D(nn.Module):
    """Head for the :class:`ResNet9` network.

    See Also:
        - :class:`ResNet9`
        - :class:`ResNet9_E`
    """

    def __init__(self):
        super(ResNet9_D, self).__init__()
        self.output_size = 100
        self.fc = nn.Linear(in_features=1024, out_features=100, bias=True)

    def forward(self, x) -> torch.Tensor:
        out = self.fc(x)
        return out


# SuPerFed - https://arxiv.org/pdf/2001.01523.pdf (CIFAR-100)
class ResNet9(EncoderHeadNet):
    """ResNet-9 network for CIFAR-100 classification. This network follows the architecture proposed
    in the [SuPerFed]_ paper, which fllows the standard ResNet-9. The encoder consists of all the
    layers but the last fully connected layer, and thus the head network consists of the last fully
    connected layer.

    See Also:
        - :class:`ResNet9_E`
        - :class:`ResNet9_D`
    """

    def __init__(self):
        super(ResNet9, self).__init__(ResNet9_E(), ResNet9_D())


# DITTO: https://arxiv.org/pdf/2012.04221.pdf (FEMNIST)
class FEMNIST_CNN_E(nn.Module):
    """Encoder for the :class:`FEMNIST_CNN` network.

    See Also:
        - :class:`FEMNIST_CNN`
        - :class:`FEMNIST_CNN_D`
    """

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
    """Head for the :class:`FEMNIST_CNN` network.

    See Also:
        - :class:`FEMNIST_CNN`
        - :class:`FEMNIST_CNN_E`
    """

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
    """Convolutional Neural Network for FEMNIST. This is a simple CNN for FEMNIST classification
    first introduced in the [DITTO]_ paper, where the architecture consists of two convolutional
    layers with 32 and 64 filters, respectively, followed by two fully connected layers with 1024
    and 62 neurons, respectively. Each convolutional layer is followed by a ReLU activation and a
    max pooling layer.

    References:
        .. [DITTO] Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. Ditto: Fair and Robust
            Federated Learning Through Personalization. In ICML (2021).
    """

    def __init__(self):
        super(FEMNIST_CNN, self).__init__(FEMNIST_CNN_E(), FEMNIST_CNN_D())


class VGG9_E(nn.Module):
    """Encoder for the :class:`VGG9` network.

    Args:
        input_size (int, optional): Size of the input tensor. Defaults to 784.
        output_size (int, optional): Number of output classes. Defaults to 62.
        seed (int, optional): Seed used for weight initialization. Defaults to 98765.

    See Also:
        - :class:`VGG9`
        - :class:`VGG9_D`
    """

    @classmethod
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
            VGG9_E._conv_layer(in_channels=1, out_channels=16, kernel_size=3,
                               padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG9_E._conv_layer(in_channels=16, out_channels=32, kernel_size=3,
                               padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG9_E._conv_layer(in_channels=32, out_channels=64, kernel_size=3,
                               padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            VGG9_E._conv_layer(in_channels=64, out_channels=128, kernel_size=3,
                               padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGG9_E._conv_layer(in_channels=128, out_channels=256,
                               kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            VGG9_E._conv_layer(in_channels=256, out_channels=512,
                               kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

    def forward(self, x) -> torch.Tensor:
        return self.encoder(x)


class VGG9_D(nn.Module):
    """Head for the :class:`VGG9` network.

    Args:
        input_size (int, optional): Size of the input tensor. Defaults to 512.
        output_size (int, optional): Number of output classes. Defaults to 62.
        seed (int, optional): Seed used for weight initialization. Defaults to 98765.

    See Also:
        - :class:`VGG9`
        - :class:`VGG9_E`
    """
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
    """VGG-9 network for FEMNIST classification. This network follows the architecture proposed in
    the [SuPerFed]_ paper which follows the standard VGG-9 architecture. In this implementation
    all convolutional layers are considered as the encoder and the fully connected layers are
    considered as the head network.

    Args:
        input_size (int, optional): Size of the input tensor. Defaults to 784.
        output_size (int, optional): Number of output classes. Defaults to 62.
        seed (int, optional): Seed used for weight initialization. Defaults to 98765.

    See Also:
        - :class:`VGG9_E`
        - :class:`VGG9_D`
    """

    def __init__(self, input_size: int = 784, output_size: int = 62, seed: int = 98765):
        super(VGG9, self).__init__(
            VGG9_E(input_size, output_size, seed),
            VGG9_D(input_size=512, output_size=output_size, seed=seed)
        )


# TODO: check if this is the correct architecture
# FedAvg: https://arxiv.org/pdf/1602.05629.pdf (CIFAR-10)
# FedDyn: https://openreview.net/pdf?id=B7v4QMR6Z9w (CIFAR-10 and CIFAR-100)
class FedavgCNN_E(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 4096
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
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
    """ResNet-18 network as defined in the torchvision library.

    Note:
        This class is a wrapper around the ResNet-18 model from torchvision and it does not
        implement the :class:`EncoderHeadNet` interface.

    Args:
        output_size (int, optional): Number of output classes. Defaults to 10.
    """

    def __init__(self, output_size=10):
        super(ResNet18, self).__init__()
        self.output_size = output_size
        self.resnet = resnet18(num_classes=output_size)

    def forward(self, x) -> torch.Tensor:
        return self.resnet(x)


class ResNet18GN(ResNet18):
    """ResNet-18 network as defined in the torchvision library but with Group Normalization layers
    instead of Batch Normalization.

    Note:
        This class is a wrapper around the ResNet-18 model from torchvision and it does not
        implement the :class:`EncoderHeadNet` interface.

    Args:
        output_size (int, optional): Number of output classes. Defaults to 10.
    """

    def __init__(self, output_size=10):
        super(ResNet18GN, self).__init__(output_size)
        batch_norm_to_group_norm(self)


# FedPer: https://arxiv.org/pdf/1912.00818.pdf (CIFAR-100)
class ResNet34(nn.Module):
    """ResNet-34 network as defined in the torchvision library.

    Note:
        This class is a wrapper around the ResNet-18 model from torchvision and it does not
        implement the :class:`EncoderHeadNet` interface.

    Args:
        output_size (int, optional): Number of output classes. Defaults to 100.
    """

    def __init__(self, output_size=100):
        super(ResNet34, self).__init__()
        self.output_size = output_size
        self.resnet = resnet34(num_classes=output_size)

    def forward(self, x) -> torch.Tensor:
        return self.resnet(x)


# MOON: https://arxiv.org/pdf/2103.16257.pdf (CIFAR-100)
class ResNet50(nn.Module):
    """ResNet-50 network as defined in the torchvision library.

    Note:
        This class is a wrapper around the ResNet-18 model from torchvision and it does not
        implement the :class:`EncoderHeadNet` interface.

    Args:
        output_size (int, optional): Number of output classes. Defaults to 100.
    """

    def __init__(self, output_size=100):
        super(ResNet50, self).__init__()
        self.output_size = output_size
        self.resnet = resnet50(num_classes=output_size)

    def forward(self, x) -> torch.Tensor:
        return self.resnet(x)


class LeNet5_E(nn.Module):
    """Encoder for the :class:`LeNet5` network.

    See Also:
        - :class:`LeNet5`
        - :class:`LeNet5_D`
    """
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
    """Head for the :class:`LeNet5` network.

    Args:
        output_size (int, optional): Number of output classes. Defaults to 100.

    See Also:
        - :class:`LeNet5`
        - :class:`LeNet5_E`
    """

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
    """LeNet-5 for CIFAR. This is a LeNet-5 for CIFAR-10/100 classification as described in the
    [FedRep]_ paper, where the architecture consists of two convolutional layers with 6 and 16
    filters, respectively, followed by two fully connected layers with 120 and 84 neurons,
    respectively, and the output layer with 10 neurons. The activation functions are ReLU. The
    architecture is also used in the [LG-FedAvg]_ paper.

    See Also:
        - :class:`LeNet5_E`
        - :class:`LeNet5_D`

    Args:
        output_size (int, optional): Number of output classes. Defaults to 100.

    References:
        .. [FedRep] Liam Collins, Hamed Hassani, Aryan Mokhtari, and Sanjay Shakkottai.
            Exploiting shared representations for personalized federated learning. In ICML (2021).
        .. [LG-FedAvg] Paul Pu Liang, Terrance Liu, Liu Ziyin, Nicholas B. Allen, Randy P. Auerbach,
            David Brent, Ruslan Salakhutdinov, Louis-Philippe Morency. Think Locally, Act Globally:
            Federated Learning with Local and Global Representations.
            In arXiv https://arxiv.org/abs/2001.01523 (2020).
    """

    def __init__(self, output_size=100):
        super(LeNet5, self).__init__(LeNet5_E(), LeNet5_D(output_size))


# SuPerFed: https://arxiv.org/pdf/2109.07628v3.pdf (Shakespeare)
class Shakespeare_LSTM_E(nn.Module):
    """Encoder for the :class:`Shakespeare_LSTM` network.

    See Also:
        - :class:`Shakespeare_LSTM`
        - :class:`Shakespeare_LSTM_D`
    """

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
        x = self.encoder(x)
        x, _ = self.rnn(x)
        return x[:, -1, :]


class Shakespeare_LSTM_D(nn.Module):
    """Head for the :class:`Shakespeare_LSTM` network.

    See Also:
        - :class:`Shakespeare_LSTM`
        - :class:`Shakespeare_LSTM_E`
    """

    def __init__(self):
        super(Shakespeare_LSTM_D, self).__init__()
        self.output_size = len(string.printable)
        self.classifier = VGG9_D._linear_layer(
            256, self.output_size, bias=False, seed=GlobalSettings().get_seed())

    def forward(self, x) -> torch.Tensor:
        return self.classifier(x)


class Shakespeare_LSTM(EncoderHeadNet):
    """LSTM for Shakespeare. This is an LSTM for Shakespeare classification first introduced
    in the [SuPerFed]_ paper, where the architecture consists of an embedding layer with 8
    dimensions, followed by a two-layer LSTM with 256 hidden units, and a linear layer with
    256 neurons.

    See Also:
        - :class:`Shakespeare_LSTM_E`
        - :class:`Shakespeare_LSTM_D`
    """

    def __init__(self):
        super(Shakespeare_LSTM, self).__init__(Shakespeare_LSTM_E(), Shakespeare_LSTM_D())


class MoonCNN_E(nn.Module):
    """Encoder for the :class:`MoonCNN` network.

    See Also:
        - :class:`MoonCNN`
        - :class:`MoonCNN_D`
    """

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
    """Head for the :class:`MoonCNN` network.

    See Also:
        - :class:`MoonCNN`
        - :class:`MoonCNN_E`
    """

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
    """Convolutional Neural Network for CIFAR-10. This is a CNN for CIFAR-10 classification first
    described in the [MOON]_ paper, where the architecture consists of two convolutional layers with
    6 and 16 filters, respectively, followed by two fully connected layers with 120 and 84 neurons,
    respectively, and a projection head with 256 neurons followed by the output layer with 10
    neurons.

    See Also:
        - :class:`MoonCNN_E`
        - :class:`MoonCNN_D`

    References:
        .. [MOON] Qinbin Li, Bingsheng He, and Dawn Song. Model-Contrastive Federated Learning.
            In CVPR (2021).
    """

    def __init__(self):
        super(MoonCNN, self).__init__(MoonCNN_E(), MoonCNN_D())
