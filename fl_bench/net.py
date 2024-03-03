import torch
import torch.nn as nn
from torch.functional import F

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
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            VGG9._linear_layer(in_features=512, out_features=256, bias=False, seed=seed),
            nn.ReLU(True),
            VGG9._linear_layer(in_features=256, out_features=output_size, bias=False, seed=seed)
        )

    def forward(self, x):
        x = self.fed_E(x)
        x = self.classifier(x)
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