from __future__ import annotations
from . import DataContainer, FastTensorDataLoader, support
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize
from torchvision import datasets
from datasets import load_dataset
from rich.progress import track
from numpy.random import permutation
from enum import Enum
import string
import torch
import json
import os
import sys
sys.path.append(".")
sys.path.append("..")


class Datasets:
    """Static class for loading datasets.

    Each dataset is loaded as a `DataContainer` object.
    """

    @classmethod
    def MNIST(cls,
              path: str = "../data",
              transforms: callable = ToTensor()) -> DataContainer:
        """Load the MNIST dataset.

        The dataset is split into training and testing sets according to the default split of the
        `torchvision.datasets.MNIST` class. The data is normalized to the range [0, 1].
        An example of the dataset is a 28x28 image, i.e., a tensor of shape (28, 28).

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to "../data".
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              `ToTensor`.

        Returns:
            DataContainer: The MNIST dataset.
        """
        train_data = datasets.MNIST(
            root=path,
            train=True,
            transform=transforms,
            download=True,
        )

        test_data = datasets.MNIST(
            root=path,
            train=False,
            transform=transforms,
            download=True
        )

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             10)

    @classmethod
    def MNIST4D(cls,
                path: str = "../data",
                transforms: callable = ToTensor()) -> DataContainer:
        """Load the MNIST dataset.

        The dataset is split into training and testing sets according to the default split of the
        `torchvision.datasets.MNIST` class. The data is normalized to the range [0, 1].
        A 4D example of the dataset is a 1x28x28 image, i.e., a tensor of shape (1, 28, 28).

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to "../data".
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              `ToTensor`.

        Returns:
            DataContainer: The MNIST dataset.
        """
        mnist_dc = Datasets.MNIST(path, transforms)
        return DataContainer(mnist_dc.train[0][:, None, :, :],
                             mnist_dc.train[1],
                             mnist_dc.test[0][:, None, :, :],
                             mnist_dc.test[1],
                             10)

    @classmethod
    def MNISTM(cls,
               path: str = "../data",
               transforms: callable = ToTensor()) -> DataContainer:
        train_data = support.MNISTM(
            root=path,
            train=True,
            transform=transforms,
            download=True,
        )

        test_data = support.MNISTM(
            root=path,
            train=False,
            transform=transforms,
            download=True
        )

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        train_data.data = torch.movedim(train_data.data, 3, 1)
        test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             10)

    @classmethod
    def EMNIST(cls,
               path: str = "../data",
               transforms: callable = ToTensor()) -> DataContainer:

        train_data = datasets.EMNIST(
            root=path,
            split="balanced",
            train=True,
            transform=transforms,
            download=True
        )

        test_data = datasets.EMNIST(
            root=path,
            split="balanced",
            train=False,
            transform=transforms,
            download=True
        )

        return DataContainer(train_data.data / 255.,
                             train_data.targets,
                             test_data.data / 255.,
                             test_data.targets,
                             47)

    @classmethod
    def SVHN(cls,
             path: str = "../data",
             transforms: callable = ToTensor()) -> DataContainer:

        train_data = datasets.SVHN(
            root=path,
            split="train",
            transform=transforms,
            download=True
        )

        test_data = datasets.SVHN(
            root=path,
            split="test",
            transform=transforms,
            download=True
        )

        return DataContainer(train_data.data / 255.,
                             train_data.labels,
                             test_data.data / 255.,
                             test_data.labels,
                             10)

    @classmethod
    def CIFAR10(cls,
                path: str = "../data",
                transforms: callable = ToTensor()) -> DataContainer:

        train_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms
        )

        test_data = datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=transforms
        )

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        train_data.data = torch.movedim(train_data.data, 3, 1)
        test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data,
                             torch.LongTensor(train_data.targets),
                             test_data.data,
                             torch.LongTensor(test_data.targets),
                             10)

    @classmethod
    def CINIC10(cls,
                path: str = "../data",
                transforms: callable = None) -> DataContainer:

        train_data = support.CINIC10(
            root=path,
            partition="train",
            download=True,
            transform=transforms
        )

        test_data = support.CINIC10(
            root=path,
            partition="test",
            download=True,
            transform=transforms
        )

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             10)

    @classmethod
    def CIFAR100(cls,
                 path: str = "../data",
                 transforms: callable = ToTensor()) -> DataContainer:

        train_data = datasets.CIFAR100(
            root=path,
            train=True,
            download=True,
            transform=transforms
        )

        test_data = datasets.CIFAR100(
            root=path,
            train=False,
            download=True,
            transform=transforms
        )

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        train_data.data = torch.movedim(train_data.data, 3, 1)
        test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data,
                             torch.LongTensor(train_data.targets),
                             test_data.data,
                             torch.LongTensor(test_data.targets),
                             100)

    @classmethod
    def FASHION_MNIST(cls,
                      path: str = "../data",
                      transforms: callable = Compose([ToTensor(),
                                                     Normalize([0.5], [0.5])])) -> DataContainer:

        train_data = datasets.FashionMNIST(
            root=path,
            train=True,
            download=True,
            transform=transforms
        )

        test_data = datasets.FashionMNIST(
            root=path,
            train=False,
            download=True,
            transform=transforms
        )

        return DataContainer(train_data.data,
                             torch.LongTensor(train_data.targets),
                             test_data.data,
                             torch.LongTensor(test_data.targets),
                             10)

    @classmethod
    def TINY_IMAGENET(cls,
                      path: str = "../data",
                      transforms: callable = None) -> DataContainer:

        tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet',
                                           split='train',
                                           cache_dir=path)

        tiny_imagenet_test = load_dataset('Maysee/tiny-imagenet',
                                          split='valid',
                                          cache_dir=path)

        X_train, y_train = [], []
        X_test, y_test = [], []

        fix_bw_trans = Lambda(lambda x: x.repeat(1, 3, 1, 1))

        for image in track(tiny_imagenet_train, "Loading Tiny ImageNet train data..."):
            y = image['label']
            image = image['image']
            x = ToTensor()(image).unsqueeze(0)
            if x.shape != torch.Size([1, 3, 64, 64]):
                x = fix_bw_trans(x)
            X_train.append(x)
            y_train.append(y)

        for image in track(tiny_imagenet_test, "Loading Tiny ImageNet test data..."):
            y = image['label']
            image = image['image']
            x = ToTensor()(image).unsqueeze(0)
            if x.shape != torch.Size([1, 3, 64, 64]):
                x = fix_bw_trans(x)
            X_test.append(x)
            y_test.append(y)

        train_data = torch.vstack(X_train)
        test_data = torch.vstack(X_test)

        if transforms is not None:
            train_data = transforms(train_data)
            test_data = transforms(test_data)

        idxperm = torch.randperm(train_data.shape[0])
        train_data = train_data[idxperm]
        y_train = torch.LongTensor(y_train)[idxperm]
        y_test = torch.LongTensor(y_test)

        return DataContainer(train_data,
                             y_train,
                             test_data,
                             y_test,
                             200)

    @classmethod
    def FEMNIST(cls,
                path: str = "./data",
                batch_size: int = 10,
                filter: str = "all"):

        def _filter_femnist(udata, filter):
            # classes: 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
            # labels : 01234567890123456789012345678901234567890123456789012345678901
            if filter == "all":
                return udata
            elif filter == "uppercase":
                udata["x"] = [x for x, y in zip(udata["x"], udata["y"]) if y < 36 and y > 9]
                udata["y"] = [y - 10 for y in udata["y"] if y < 36 and y > 9]
            elif filter == "lowercase":
                udata["x"] = [x for x, y in zip(udata["x"], udata["y"]) if y > 35]
                udata["y"] = [y - 36 for y in udata["y"] if y > 35]
            elif filter == "digits":
                udata["x"] = [x for x, y in zip(udata["x"], udata["y"]) if y < 10]
                udata["y"] = [y for y in udata["y"] if y < 10]
            else:
                raise ValueError(f"Invalid filter: {filter}")
            return udata

        femnist_path = os.path.join(path, "FEMNIST")
        train_dir = os.path.join(femnist_path, 'train')
        test_dir = os.path.join(femnist_path, 'test')

        assert os.path.exists(femnist_path), f"FEMNIST data ({femnist_path}) not found."
        assert os.path.exists(train_dir), f"FEMNIST train data ({train_dir}') not found."
        assert os.path.exists(test_dir), f"FEMNIST test data ({test_dir}') not found."

        # TRAINING
        files = os.listdir(train_dir)
        dict_train = {}
        for file in track(files, "Loading FEMNIST train data..."):
            with open(os.path.join(train_dir, file)) as f:
                data = json.load(f)
            dict_train.update(data["user_data"])

        # TEST
        files = os.listdir(test_dir)
        dict_test = {}
        for file in track(files, "Loading FEMNIST test data..."):
            with open(os.path.join(test_dir, file)) as f:
                data = json.load(f)
            dict_test.update(data["user_data"])

        client_tr_assignments = []
        for k in track(sorted(dict_train), "Creating training data loader..."):
            udata = dict_train[k]
            udata = _filter_femnist(udata, filter)
            Xtr_client = torch.FloatTensor(udata["x"]).reshape(-1, 1, 28, 28)
            ytr_client = torch.LongTensor(udata["y"])
            client_tr_assignments.append(
                FastTensorDataLoader(
                    Xtr_client,
                    ytr_client,
                    num_labels=62,
                    batch_size=batch_size,
                    shuffle=True,
                    percentage=1.0
                )
            )

        client_te_assignments = []
        for k in track(sorted(dict_train), "Creating testing data loader..."):
            udata = dict_test[k]
            udata = _filter_femnist(udata, filter)
            Xte_client = torch.FloatTensor(udata["x"]).reshape(-1, 1, 28, 28)
            yte_client = torch.LongTensor(udata["y"])
            client_te_assignments.append(
                FastTensorDataLoader(
                    Xte_client,
                    yte_client,
                    num_labels=62,
                    batch_size=64,
                    shuffle=True,
                    percentage=1.0
                )
            )

        perm = permutation(len(client_tr_assignments))
        client_tr_assignments = [client_tr_assignments[i] for i in perm]
        client_te_assignments = [client_te_assignments[i] for i in perm]
        return client_tr_assignments, client_te_assignments, None

    @classmethod
    def SHAKESPEARE(cls,
                    path: str = "./data",
                    batch_size: int = 10):

        shake_path = os.path.join(path, "shakespeare")
        train_dir = os.path.join(shake_path, 'train')
        test_dir = os.path.join(shake_path, 'test')

        assert os.path.exists(shake_path), f"shakespeare data ({shake_path}) not found."
        assert os.path.exists(train_dir), f"shakespeare train data ({train_dir}') not found."
        assert os.path.exists(test_dir), f"shakespeare test data ({test_dir}') not found."

        all_chr = string.printable
        chr_map = {c: i for i, c in enumerate(all_chr)}

        # TRAINING
        files = os.listdir(train_dir)
        dict_train = {}
        for file in track(files, "Loading Shakespeare train data..."):
            with open(os.path.join(train_dir, file)) as f:
                data = json.load(f)
            dict_train.update(data["user_data"])

        # TEST
        files = os.listdir(test_dir)
        dict_test = {}
        for file in track(files, "Loading Shakespeare test data..."):
            with open(os.path.join(test_dir, file)) as f:
                data = json.load(f)
            dict_test.update(data["user_data"])

        client_tr_assignments = []
        for k in track(sorted(dict_train), "Creating training data loader..."):
            udata = dict_train[k]
            inputs, targets = udata['x'], udata['y']
            for idx in range(len(inputs)):
                inputs[idx] = [chr_map[char] for char in inputs[idx]]
            for idx in range(len(targets)):
                targets[idx] = chr_map[targets[idx]]

            Xtr_client = torch.LongTensor(inputs)
            ytr_client = torch.LongTensor(targets)
            client_tr_assignments.append(
                FastTensorDataLoader(
                    Xtr_client,
                    ytr_client,
                    num_labels=100,
                    batch_size=batch_size,
                    shuffle=True,
                    percentage=1.0
                )
            )

        client_te_assignments = []
        for k in track(sorted(dict_train), "Creating test data loader..."):
            udata = dict_test[k]
            inputs, targets = udata['x'], udata['y']
            for idx in range(len(inputs)):
                inputs[idx] = [chr_map[char] for char in inputs[idx]]
            for idx in range(len(targets)):
                targets[idx] = chr_map[targets[idx]]

            Xte_client = torch.LongTensor(inputs)
            yte_client = torch.LongTensor(targets)
            client_te_assignments.append(
                FastTensorDataLoader(
                    Xte_client,
                    yte_client,
                    num_labels=100,
                    batch_size=batch_size,
                    shuffle=True,
                    percentage=1.0
                )
            )

        perm = permutation(len(client_tr_assignments))
        client_tr_assignments = [client_tr_assignments[i] for i in perm]
        client_te_assignments = [client_te_assignments[i] for i in perm]
        return client_tr_assignments, client_te_assignments, None


class DatasetsEnum(Enum):
    MNIST = "mnist"
    MNISTM = "mnistm"
    MNIST4D = "mnist4d"
    SVHN = "svhn"
    FEMNIST = "femnist"
    EMNIST = "emnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TINY_IMAGENET = "tiny_imagenet"
    SHAKESPEARE = "shakespeare"
    FASHION_MNIST = "fashion_mnist"

    @classmethod
    def contains(cls, member: object) -> bool:
        if isinstance(member, str):
            return member in cls._value2member_map_.keys()
        elif isinstance(member, DatasetsEnum):
            return member.value in cls._member_names_

    def klass(self):
        DATASET_MAP = {
            "mnist": Datasets.MNIST,
            "mnist4d": Datasets.MNIST4D,
            "svhn": Datasets.SVHN,
            "mnistm": Datasets.MNISTM,
            "femnist": Datasets.FEMNIST,
            "emnist": Datasets.EMNIST,
            "cifar10": Datasets.CIFAR10,
            "cifar100": Datasets.CIFAR100,
            "tiny_imagenet": Datasets.TINY_IMAGENET,
            "shakespeare": Datasets.SHAKESPEARE,
            "fashion_mnist": Datasets.FASHION_MNIST
        }
        return DATASET_MAP[self.value]

    def splitter(self):
        from . import DataSplitter, DummyDataSplitter

        # LEAF datasets are already divided into clients
        if self.value == "femnist" or self.value == "shakespeare":
            return DummyDataSplitter
        else:
            return DataSplitter
