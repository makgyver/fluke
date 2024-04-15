from __future__ import annotations
import torch

import sys
sys.path.append(".")
sys.path.append("..")

from fl_bench.data.datasets import Datasets  # NOQA


# ### MNIST
def test_mnist():
    mnist = Datasets.MNIST("./data")
    assert mnist.train[0].shape == torch.Size([60000, 28, 28])
    assert mnist.train[1].shape == torch.Size([60000])
    assert mnist.test[0].shape == torch.Size([10000, 28, 28])
    assert mnist.test[1].shape == torch.Size([10000])
    assert mnist.num_classes == len(
        set(mnist.train[1].unique().tolist() + mnist.test[1].unique().tolist()))
    assert mnist.num_classes == 10


# ### MNIST 4D
def test_mnist4d():
    mnist4d = Datasets.MNIST("./data", channel_dim=True)
    assert mnist4d.train[0].shape == torch.Size([60000, 1, 28, 28])
    assert mnist4d.train[1].shape == torch.Size([60000])
    assert mnist4d.test[0].shape == torch.Size([10000, 1, 28, 28])
    assert mnist4d.test[1].shape == torch.Size([10000])
    assert mnist4d.num_classes == len(
        set(mnist4d.train[1].unique().tolist() + mnist4d.test[1].unique().tolist()))
    assert mnist4d.num_classes == 10


# ### EMNIST
def test_emnist():
    emnist = Datasets.EMNIST("./data")
    assert emnist.train[0].shape == torch.Size([112800, 28, 28])
    assert emnist.train[1].shape == torch.Size([112800])
    assert emnist.test[0].shape == torch.Size([18800, 28, 28])
    assert emnist.test[1].shape == torch.Size([18800])
    assert emnist.num_classes == len(
        set(emnist.train[1].unique().tolist() + emnist.test[1].unique().tolist()))
    assert emnist.num_classes == 47


# ### SVHN
def test_svhn():
    svhn = Datasets.SVHN("./data")
    assert svhn.train[0].shape == torch.Size([73257, 3, 32, 32])
    assert svhn.train[1].shape == torch.Size([73257])
    assert svhn.test[0].shape == torch.Size([26032, 3, 32, 32])
    assert svhn.test[1].shape == torch.Size([26032])
    assert svhn.num_classes == len(
        set(svhn.train[1].unique().tolist() + svhn.test[1].unique().tolist()))
    assert svhn.num_classes == 10


# ### CIFAR10
def test_cifar10():
    cifar10 = Datasets.CIFAR10("./data")
    assert cifar10.train[0].shape == torch.Size([50000, 3, 32, 32])
    assert cifar10.train[1].shape == torch.Size([50000])
    assert cifar10.test[0].shape == torch.Size([10000, 3, 32, 32])
    assert cifar10.test[1].shape == torch.Size([10000])
    assert cifar10.num_classes == len(
        set(cifar10.train[1].unique().tolist() + cifar10.test[1].unique().tolist()))
    assert cifar10.num_classes == 10


# ### CIFAR100
def test_cifar100():
    cifar100 = Datasets.CIFAR100("./data")
    assert cifar100.train[0].shape == torch.Size([50000, 3, 32, 32])
    assert cifar100.train[1].shape == torch.Size([50000])
    assert cifar100.test[0].shape == torch.Size([10000, 3, 32, 32])
    assert cifar100.test[1].shape == torch.Size([10000])
    assert cifar100.num_classes == len(
        set(cifar100.train[1].unique().tolist() + cifar100.test[1].unique().tolist()))
    assert cifar100.num_classes == 100


# ### MNIST-M
def test_mnistm():
    mnistm = Datasets.MNISTM("./data")
    assert mnistm.train[0].shape == torch.Size([60000, 3, 28, 28])
    assert mnistm.train[1].shape == torch.Size([60000])
    assert mnistm.test[0].shape == torch.Size([10000, 3, 28, 28])
    assert mnistm.test[1].shape == torch.Size([10000])
    assert mnistm.num_classes == len(
        set(mnistm.train[1].unique().tolist() + mnistm.test[1].unique().tolist()))
    assert mnistm.num_classes == 10


# ### Tiny Imagenet
def test_tinyimagenet():
    tiny_imagenet = Datasets.TINY_IMAGENET("./data")
    assert tiny_imagenet.train[0].shape == torch.Size([100000, 3, 64, 64])
    assert tiny_imagenet.train[1].shape == torch.Size([100000])
    assert tiny_imagenet.test[0].shape == torch.Size([10000, 3, 64, 64])
    assert tiny_imagenet.test[1].shape == torch.Size([10000])
    assert tiny_imagenet.num_classes == len(
        set(tiny_imagenet.train[1].unique().tolist() + tiny_imagenet.test[1].unique().tolist()))
    assert tiny_imagenet.num_classes == 200


# ### FEMNIST
def test_femnist():

    femnist = Datasets.FEMNIST("./data")
    assert len(femnist[0]) == 3597  # Total number of clients
    assert len(femnist[1]) == 3597  # Total number of clients
    assert femnist[0][0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist[0][i].tensors[1].shape[0] == femnist[0][i].tensors[0].shape[0]
                for i in range(len(femnist[0]))]) == len(femnist[0])

    assert femnist[1][0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist[1][i].tensors[1].shape[0] == femnist[1][i].tensors[0].shape[0]
                for i in range(len(femnist[1]))]) == len(femnist[1])

    lbl_train = set.union(*[set(femnist[0][i].tensors[1].numpy()) for i in range(len(femnist[0]))])
    lbl_test = set.union(*[set(femnist[1][i].tensors[1].numpy()) for i in range(len(femnist[1]))])
    assert len(lbl_train | lbl_test) == 62  # Total number of classes


# ### Shakespeare
def test_shakespeare():
    shake = Datasets.SHAKESPEARE("./data")
    assert len(shake[0]) == 660
    assert len(shake[1]) == 660
    assert shake[0][0].tensors[0].shape[1:] == torch.Size([80])  # Shakespeare text
    assert shake[1][0].tensors[0].shape[1:] == torch.Size([80])  # Shakespeare text

    assert sum([shake[0][i].tensors[1].shape[0] == shake[0][i].tensors[0].shape[0]
                for i in range(len(shake[0]))]) == len(shake[0])
    assert sum([shake[1][i].tensors[1].shape[0] == shake[1][i].tensors[0].shape[0]
                for i in range(len(shake[1]))]) == len(shake[1])


# ### Fashion MNIST
def test_fashion_mnist():
    fashion = Datasets.FASHION_MNIST("./data")
    assert fashion.train[0].shape == torch.Size([60000, 28, 28])
    assert fashion.train[1].shape == torch.Size([60000])
    assert fashion.test[0].shape == torch.Size([10000, 28, 28])
    assert fashion.test[1].shape == torch.Size([10000])
    assert fashion.num_classes == len(
        set(fashion.train[1].unique().tolist() + fashion.test[1].unique().tolist()))
    assert fashion.num_classes == 10


# ### CINIC10
def test_cinic10():
    cinic = Datasets.CINIC10("./data")
    assert cinic.train[0].shape == torch.Size([90000, 3, 32, 32])
    assert cinic.train[1].shape == torch.Size([90000])
    assert cinic.test[0].shape == torch.Size([90000, 3, 32, 32])
    assert cinic.test[1].shape == torch.Size([90000])
    assert cinic.num_classes == len(
        set(cinic.train[1].unique().tolist() + cinic.test[1].unique().tolist()))
    assert cinic.num_classes == 10


if __name__ == "__main__":
    import os
    print(os.path.abspath("./data"))
    test_mnist()
    test_mnist4d()
    # test_emnist()
    test_svhn()
    test_cifar10()
    test_cifar100()
    test_mnistm()
    test_tinyimagenet()
    test_femnist()
    test_shakespeare()
    test_fashion_mnist()
    test_cinic10()
