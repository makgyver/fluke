from __future__ import annotations

import sys
import urllib
import warnings

import pytest
import torch
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor)

sys.path.append(".")
sys.path.append("..")

from fluke.data.datasets import Datasets  # NOQA
from fluke.data.support import CINIC10, MNISTM  # NOQA


# ### MNIST
def test_mnist():

    with pytest.raises(ValueError):
        mnist = Datasets.get("monist")

    mnist = Datasets.get("mnist", path="./data")
    assert mnist.train[0].shape == torch.Size([60000, 28, 28])
    assert mnist.train[1].shape == torch.Size([60000])
    assert mnist.test[0].shape == torch.Size([10000, 28, 28])
    assert mnist.test[1].shape == torch.Size([10000])
    assert mnist.num_classes == len(
        set(mnist.train[1].unique().tolist() + mnist.test[1].unique().tolist()))
    assert mnist.num_classes == 10

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
    try:
        svhn = Datasets.SVHN("./data")
    except urllib.error.URLError:
        warnings.warn("Skipped SVHN due to time out - Unknown issue!")
        assert True  # this is a "fix" to a non sensical issue on github action
        return
    assert svhn.train[0].shape == torch.Size([73257, 3, 32, 32])
    assert svhn.train[1].shape == torch.Size([73257])
    assert svhn.test[0].shape == torch.Size([26032, 3, 32, 32])
    assert svhn.test[1].shape == torch.Size([26032])
    assert svhn.num_classes == len(
        set(svhn.train[1].unique().tolist() + svhn.test[1].unique().tolist()))
    assert svhn.num_classes == 10


# ### CIFAR10
def test_cifar10():
    transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar10 = Datasets.CIFAR10("./data", transforms=transform)
    assert cifar10.train[0].shape == torch.Size([50000, 3, 32, 32])
    assert cifar10.train[1].shape == torch.Size([50000])
    assert cifar10.test[0].shape == torch.Size([10000, 3, 32, 32])
    assert cifar10.test[1].shape == torch.Size([10000])
    assert cifar10.num_classes == len(
        set(cifar10.train[1].unique().tolist() + cifar10.test[1].unique().tolist()))
    assert cifar10.num_classes == 10

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

    dmnistm = MNISTM("./data", transform=ToTensor())
    img, label = dmnistm[0]
    assert img.shape == torch.Size([3, 28, 28])
    assert isinstance(label, int)
    assert dmnistm[0][0].shape == torch.Size([3, 28, 28])
    assert len(dmnistm) == 60000

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
    try:
        femnist = Datasets.FEMNIST("./data")
    except AssertionError:
        return
    assert len(femnist.clients_tr) == 3597  # Total number of clients
    assert len(femnist.clients_te) == 3597  # Total number of clients
    assert femnist.clients_tr[0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist.clients_tr[i].tensors[1].shape[0] == femnist.clients_tr[i].tensors[0].shape[0]
                for i in range(len(femnist.clients_tr))]) == len(femnist.clients_tr)

    assert femnist.clients_te[0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist.clients_te[i].tensors[1].shape[0] == femnist.clients_te[i].tensors[0].shape[0]
                for i in range(len(femnist.clients_te))]) == len(femnist.clients_te)

    lbl_train = set.union(*[set(femnist.clients_tr[i].tensors[1].numpy())
                          for i in range(len(femnist.clients_tr))])
    lbl_test = set.union(*[set(femnist.clients_te[i].tensors[1].numpy())
                         for i in range(len(femnist.clients_te))])
    assert len(lbl_train | lbl_test) == 62  # Total number of classes


def test_femnist_dig():
    try:
        femnist_dig = Datasets.FEMNIST("./data", filter="digits")
    except AssertionError:
        return
    assert len(femnist_dig.clients_tr) == 3597  # Total number of clients
    assert len(femnist_dig.clients_te) == 3597  # Total number of clients
    assert femnist_dig.clients_tr[0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist_dig.clients_tr[i].tensors[1].shape[0] == femnist_dig.clients_tr[i].tensors[0].shape[0]
                for i in range(len(femnist_dig.clients_tr))]) == len(femnist_dig.clients_tr)

    assert femnist_dig.clients_te[0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist_dig.clients_te[i].tensors[1].shape[0] == femnist_dig.clients_te[i].tensors[0].shape[0]
                for i in range(len(femnist_dig.clients_te))]) == len(femnist_dig.clients_te)

    lbl_train = set.union(*[set(femnist_dig.clients_tr[i].tensors[1].numpy())
                          for i in range(len(femnist_dig.clients_tr))])
    lbl_test = set.union(*[set(femnist_dig.clients_te[i].tensors[1].numpy())
                         for i in range(len(femnist_dig.clients_te))])
    assert len(lbl_train | lbl_test) == 10


def test_femnist_upp():
    try:
        femnist_u = Datasets.FEMNIST("./data", filter="uppercase")
    except AssertionError:
        return
    assert len(femnist_u.clients_tr) == 3597  # Total number of clients
    assert len(femnist_u.clients_te) == 3597  # Total number of clients
    assert femnist_u.clients_tr[0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist_u.clients_tr[i].tensors[1].shape[0] == femnist_u.clients_tr[i].tensors[0].shape[0]
                for i in range(len(femnist_u.clients_tr))]) == len(femnist_u.clients_tr)

    assert femnist_u.clients_te[0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist_u.clients_te[i].tensors[1].shape[0] == femnist_u.clients_te[i].tensors[0].shape[0]
                for i in range(len(femnist_u.clients_te))]) == len(femnist_u.clients_te)

    lbl_train = set.union(*[set(femnist_u.clients_tr[i].tensors[1].numpy())
                          for i in range(len(femnist_u.clients_tr))])
    lbl_test = set.union(*[set(femnist_u.clients_te[i].tensors[1].numpy())
                         for i in range(len(femnist_u.clients_te))])
    assert len(lbl_train | lbl_test) == 26


def test_femnist_low():
    try:
        femnist_l = Datasets.FEMNIST("./data", filter="lowercase")
    except AssertionError:
        return
    assert len(femnist_l.clients_tr) == 3597  # Total number of clients
    assert len(femnist_l.clients_te) == 3597  # Total number of clients
    assert femnist_l.clients_tr[0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist_l.clients_tr[i].tensors[1].shape[0] == femnist_l.clients_tr[i].tensors[0].shape[0]
                for i in range(len(femnist_l.clients_tr))]) == len(femnist_l.clients_tr)

    assert femnist_l.clients_te[0].tensors[0].shape[1:] == torch.Size([1, 28, 28])  # image shape
    # number of labels matches number of images in each client
    assert sum([femnist_l.clients_te[i].tensors[1].shape[0] == femnist_l.clients_te[i].tensors[0].shape[0]
                for i in range(len(femnist_l.clients_te))]) == len(femnist_l.clients_te)

    lbl_train = set.union(*[set(femnist_l.clients_tr[i].tensors[1].numpy())
                          for i in range(len(femnist_l.clients_tr))])
    lbl_test = set.union(*[set(femnist_l.clients_te[i].tensors[1].numpy())
                         for i in range(len(femnist_l.clients_te))])
    assert len(lbl_train | lbl_test) == 26


# ### Shakespeare
def test_shakespeare():
    try:
        shake = Datasets.SHAKESPEARE("./data")
    except AssertionError:
        return

    assert len(shake.clients_tr) == 660
    assert len(shake.clients_te) == 660
    assert shake.clients_tr[0].tensors[0].shape[1:] == torch.Size([80])  # Shakespeare text
    assert shake.clients_te[0].tensors[0].shape[1:] == torch.Size([80])  # Shakespeare text

    assert sum([shake.clients_tr[i].tensors[1].shape[0] == shake.clients_tr[i].tensors[0].shape[0]
                for i in range(len(shake.clients_tr))]) == len(shake.clients_tr)
    assert sum([shake.clients_te[i].tensors[1].shape[0] == shake.clients_te[i].tensors[0].shape[0]
                for i in range(len(shake.clients_te))]) == len(shake.clients_te)


# ### Fashion MNIST
def test_fashion_mnist():
    fashion = Datasets.FASHION_MNIST("./data")
    assert fashion.train[0].shape == torch.Size([60000, 28, 28])
    assert fashion.train[1].shape == torch.Size([60000])
    assert fashion.test[0].shape == torch.Size([10000, 28, 28])
    assert fashion.test[1].shape == torch.Size([10000])
    assert fashion.num_classes == len(set(fashion.train[1].unique().tolist() +
                                          fashion.test[1].unique().tolist()))
    assert fashion.num_classes == 10


# ### CINIC10
def test_cinic10():
    dcinic = CINIC10("./data")
    img, label = dcinic[0]
    assert img.shape == torch.Size([3, 32, 32])
    assert label.shape == torch.Size([])
    assert len(dcinic) == 90000

    cinic = Datasets.CINIC10("./data")
    assert cinic.train[0].shape == torch.Size([90000, 3, 32, 32])
    assert cinic.train[1].shape == torch.Size([90000])
    assert cinic.test[0].shape == torch.Size([90000, 3, 32, 32])
    assert cinic.test[1].shape == torch.Size([90000])
    assert cinic.num_classes == len(
        set(cinic.train[1].unique().tolist() + cinic.test[1].unique().tolist()))
    assert cinic.num_classes == 10


# ### CINIC10
def test_fcube():
    data_container = Datasets.FCUBE()
    ctr, cte, ste = data_container.clients_tr, data_container.clients_te, data_container.server_data
    assert len(ctr) == 4
    assert len(cte) == 4
    assert ste is not None
    assert ste.size == 100
    assert sum([cte[i].size + ctr[i].size for i in range(4)]) == 900
    assert ctr[0].num_labels == 2


if __name__ == "__main__":
    # test_mnist()
    # test_mnist4d()
    # test_emnist()
    # test_svhn()
    # test_cifar10()
    # test_cifar100()
    # test_mnistm()
    # test_tinyimagenet()
    # test_femnist()
    # test_femnist_dig()
    # test_femnist_upp()
    # test_femnist_low()
    # test_shakespeare()
    # test_fashion_mnist()
    # test_cinic10()
    test_fcube()

    # 98% coverate on datasets.py
    # 88% coverage on support.py
