from __future__ import annotations

import sys

import torch

sys.path.append(".")
sys.path.append("..")


# 'EncoderHeadNet',
# 'GlobalLocalNet',
# 'HeadGlobalEncoderLocalNet',
# 'EncoderGlobalHeadLocalNet',


from fluke.nets import EncoderGlobalHeadLocalNet  # NOQA
from fluke.nets import (FEMNIST_CNN, MNIST_2NN, MNIST_CNN, MNIST_LR,  # NOQA
                        VGG9, CifarConv2, FedavgCNN, FedBN_CNN,
                        HeadGlobalEncoderLocalNet, LeNet5, MoonCNN, ResNet9,
                        ResNet18, ResNet18GN, ResNet34, ResNet50,
                        Shakespeare_LSTM)


def test_mnist_2nn():

    model = MNIST_2NN()

    x = torch.randn(1, 28, 28)
    y = model(x)

    y1 = model.forward(x)
    z = model.forward_encoder(x)
    y2 = model.forward_head(z)

    assert z.shape == (1, 100)
    assert torch.allclose(y, y1)
    assert torch.allclose(y, y2)

    model = MNIST_2NN(softmax=True)

    y = model(x)
    assert y.shape == (1, 10)
    assert torch.allclose(y.sum(), torch.tensor(1.0))


def test_mnist_lr():
    model = MNIST_LR()
    x = torch.randn(1, 28, 28)
    y = model(x)

    assert y.shape == (1, 10)
    assert torch.allclose(y.sum(), torch.tensor(1.0))


def test_convnets():
    model = MNIST_CNN()
    x = torch.randn(1, 1, 28, 28)
    y1 = model(x)
    z = model.forward_encoder(x)
    y2 = model.forward_head(z)
    assert y1.shape == (1, 10)
    assert torch.allclose(y1.sum(), torch.tensor(1.0))
    assert z.shape == (1, 1024)
    assert torch.allclose(y1, y2)

    model = FEMNIST_CNN()
    x = torch.randn(1, 1, 28, 28)
    y1 = model(x)
    z = model.forward_encoder(x)
    y2 = model.forward_head(z)
    assert y1.shape == (1, 62)
    assert z.shape == (1, 3136)
    assert torch.allclose(y1, y2)

    model = VGG9()
    x = torch.randn(1, 1, 28, 28)
    y1 = model(x)
    z = model.forward_encoder(x)
    y2 = model.forward_head(z)
    assert y1.shape == (1, 62)
    assert z.shape == (1, 512)
    assert torch.allclose(y1, y2)

    model = FedavgCNN()
    x = torch.randn(1, 3, 32, 32)
    y1 = model(x)
    z = model.forward_encoder(x)
    y2 = model.forward_head(z)
    assert y1.shape == (1, 10)
    assert z.shape == (1, 4096)
    assert torch.allclose(y1, y2)

    model = LeNet5(10)
    x = torch.randn(1, 3, 32, 32)
    z = model.forward_encoder(x)
    y1 = model(x)
    y2 = model.forward_head(z)
    assert y1.shape == (1, 10)
    assert z.shape == (1, 400)
    assert torch.allclose(y1, y2)

    model = MoonCNN()
    x = torch.randn(1, 3, 32, 32)
    z = model.forward_encoder(x)
    y1 = model(x)
    y2 = model.forward_head(z)
    assert y1.shape == (1, 10)
    assert z.shape == (1, 400)
    assert torch.allclose(y1, y2)

    model = FedBN_CNN()
    x = torch.randn(2, 1, 28, 28)
    z = model.forward_encoder(x)
    y1 = model(x)
    y2 = model.forward_head(z)
    assert y1.shape == (2, 10)
    assert z.shape == (2, 6272)
    assert torch.allclose(y1, y2)

    model = CifarConv2()
    x = torch.randn(2, 3, 32, 32)
    z = model.forward_encoder(x)
    y2 = model.forward_head(z)
    y1 = model(x)
    assert y1.shape == (2, 10)
    assert z.shape == (2, 1600)
    assert torch.allclose(y1, y2)

    model = ResNet9()
    x = torch.randn(1, 3, 32, 32)
    z = model.forward_encoder(x)
    y1 = model(x)
    y2 = model.forward_head(z)
    assert y1.shape == (1, 100)
    assert z.shape == (1, 1024)
    assert torch.allclose(y1, y2)

    model = ResNet18()
    x = torch.randn(2, 3, 32, 32)
    y1 = model(x)
    assert y1.shape == (2, 10)

    model = ResNet18GN()
    x = torch.randn(2, 3, 32, 32)
    y1 = model(x)
    assert y1.shape == (2, 10)

    model = ResNet34()
    x = torch.randn(2, 3, 32, 32)
    y1 = model(x)
    assert y1.shape == (2, 100)

    model = ResNet50()
    x = torch.randn(2, 3, 32, 32)
    y1 = model(x)
    assert y1.shape == (2, 100)


def test_recurrent():
    model = Shakespeare_LSTM()
    x = torch.randint(0, 100, (1, 10))
    z = model.forward_encoder(x)
    y1 = model(x)
    y2 = model.forward_head(z)
    assert z.shape == (1, 256)
    assert y1.shape == (1, 100)
    assert y2.shape == (1, 100)
    assert torch.allclose(y1, y2)


def test_global_local():
    nn = MNIST_2NN()
    model = HeadGlobalEncoderLocalNet(nn)
    assert nn.encoder is model.get_local()
    assert nn.head is model.get_global()

    model = EncoderGlobalHeadLocalNet(nn)
    assert nn.encoder is model.get_global()
    assert nn.head is model.get_local()


if __name__ == "__main__":
    test_mnist_2nn()
    test_mnist_lr()
    test_convnets()
    test_recurrent()
    test_global_local()
    # 99% coverage on nets.py
