"""
This module contains the :class:`Datasets` for loading the supported datasets.
"""
from __future__ import annotations

import json
import os
import string
import sys
from typing import Optional

import torch
from datasets import load_dataset
from numpy.random import permutation
from rich.progress import track
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torchvision.transforms import Lambda, ToTensor

sys.path.append(".")
sys.path.append("..")

from ..utils import get_class_from_qualified_name  # NOQA
from . import DataContainer, FastDataLoader, support  # NOQA

__all__ = [
    "Datasets"
]


def _apply_transforms(dataset: VisionDataset, transforms: Optional[callable]) -> VisionDataset:
    if transforms is not None:
        new_data = []
        for i in range(len(dataset)):
            new_data.append(transforms(dataset[i][0]))
        dataset.data = torch.stack(new_data)

    dataset.data = torch.Tensor(dataset.data)
    dataset.targets = torch.LongTensor(dataset.targets)
    return dataset


class Datasets:
    """Static class for loading datasets.
    Datasets are downloaded (if needed) into the ``path`` folder. The supported datasets are:
    ``MNIST``, ``MNISTM``, ``SVHN``, ``FEMNIST``, ``EMNIST``, ``CIFAR10``, ``CIFAR100``,
    ``Tiny Imagenet``, ``Shakespear``, ``Fashion MNIST``, and ``CINIC10``.
    Each dataset but ``femnist`` and ``shakespeare`` can be transformed using the ``transforms``
    argument. Each dataset is returned as a :class:`fluke.data.DataContainer` object.

    .. important::
        ``onthefly_transforms`` are transformations that are applied on-the-fly to the data
        through the data loader. This is useful when the transformations are stochastic and
        should be applied at each iteration. These transformations cannot be configured through
        the configuration file.
    """

    @classmethod
    def get(cls, name: str, **kwargs) -> DataContainer | tuple:
        """Get a dataset by name initialized with the provided arguments.
        Supported datasets are: ``mnist``, ``mnistm``, ``svhn``, ``femnist``, ``emnist``,
        ``cifar10``, ``cifar100``, ``tiny_imagenet``, ``shakespeare``, ``fashion_mnist``, and
        ``cinic10``. If `name` is not in the supported datasets, it is assumed to be a fully
        qualified name of a custom dataset function (``callable[..., DataContainer]``).

        Args:
            name (str): The name of the dataset to load or the fully qualified name of a custom
              dataset function.
            **kwargs: Additional arguments to pass to construct the dataset.

        Returns:
            DataContainer: The ``DataContainer`` object containing the dataset.

        Raises:
            ValueError: If the dataset is not supported or the name is wrong.
        """
        if name not in Datasets._DATASET_MAP:
            try:
                data_fun = get_class_from_qualified_name(name)
                return data_fun(**kwargs)
            except (ModuleNotFoundError, TypeError, ValueError) as e:
                if "." in name:
                    raise e
                else:
                    raise ValueError(f"Dataset {name} not found. The supported datasets are: " +
                                     ", ".join(Datasets._DATASET_MAP.keys()) + ".")

        return Datasets._DATASET_MAP[name](**kwargs)

    @classmethod
    def MNIST(cls,
              path: str = "../data",
              transforms: Optional[callable] = None,
              onthefly_transforms: Optional[callable] = None,
              channel_dim: bool = False) -> DataContainer:
        """Load the MNIST dataset.
        The dataset is split into training and testing sets according to the default split of the
        :class:`torchvision.datasets.MNIST` class. If no transformations are provided, the data is
        normalized to the range [0, 1].
        An example of the dataset is a 28x28 image, i.e., a tensor of shape (28, 28).
        The dataset has 10 classes, corresponding to the digits 0-9.

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.
            channel_dim (bool, optional): Whether to add a channel dimension to the data, i.e., the
              shape of the an example becomes (1, 28, 28). Defaults to ``False``.

        Returns:
            DataContainer: The MNIST dataset.
        """
        train_data = datasets.MNIST(
            root=path,
            train=True,
            download=True
        )

        test_data = datasets.MNIST(
            root=path,
            train=False,
            download=True
        )

        train_data = _apply_transforms(train_data, transforms)
        test_data = _apply_transforms(test_data, transforms)

        if transforms is None:
            train_data.data = torch.Tensor(train_data.data / 255.)
            test_data.data = torch.Tensor(test_data.data / 255.)

        return DataContainer(train_data.data if not channel_dim else train_data.data[:, None, :, :],
                             train_data.targets,
                             test_data.data if not channel_dim else test_data.data[:, None, :, :],
                             test_data.targets,
                             10,
                             onthefly_transforms)

    @classmethod
    def MNISTM(cls,
               path: str = "../data",
               transforms: Optional[callable] = None,
               onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """Load the MNIST-M dataset. MNIST-M is a dataset where the MNIST digits are placed on
        random color patches. The dataset is split into training and testing sets according to the
        default split of the data at https://github.com/liyxi/mnist-m/releases/download/data/.
        If no transformations are provided, the data is normalized to the range [0, 1].
        The dataset has 10 classes, corresponding to the digits 0-9.

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.

        Returns:
            DataContainer: The MNIST-M dataset.
        """
        train_data = support.MNISTM(
            root=path,
            train=True,
            download=True,
        )

        test_data = support.MNISTM(
            root=path,
            train=False,
            download=True
        )

        train_data = _apply_transforms(train_data, transforms)
        test_data = _apply_transforms(test_data, transforms)

        if transforms is None:
            train_data.data = train_data.data / 255.
            test_data.data = test_data.data / 255.

            train_data.data = torch.movedim(train_data.data, 3, 1)
            test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             10,
                             onthefly_transforms)

    @classmethod
    def EMNIST(cls,
               path: str = "../data",
               transforms: Optional[callable] = None,
               onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """Load the Extended MNIST (EMNIST) dataset. The dataset is split into training and testing
        sets according to the default split of the data at
        https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist as provided by
        the :class:`torchvision.datasets.EMNIST` class.
        If no transformations are provided, the data is normalized to the range [0, 1]. The dataset
        has 47 classes, corresponding to the digits 0-9 and the uppercase and lowercase letters.

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.

        Returns:
            DataContainer: The EMNIST dataset.
        """

        train_data = datasets.EMNIST(
            root=path,
            split="balanced",
            train=True,
            download=True
        )

        test_data = datasets.EMNIST(
            root=path,
            split="balanced",
            train=False,
            download=True
        )

        train_data = _apply_transforms(train_data, transforms)
        test_data = _apply_transforms(test_data, transforms)

        if transforms is None:
            train_data.data = train_data.data / 255.
            test_data.data = test_data.data / 255.

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             47,
                             onthefly_transforms)

    @classmethod
    def SVHN(cls,
             path: str = "../data",
             transforms: Optional[callable] = None,
             onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """
        Load the Street View House Numbers (SVHN) dataset. The dataset is split into training and
        testing sets according to the default split of the :class:`torchvision.datasets.SVHN` class.
        If no transformations are provided, the data is normalized to the range [0, 1]. The dataset
        contains 10 classes, corresponding to the digits 0-9.

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.

        Returns:
            DataContainer: The SVHN dataset.
        """
        train_data = datasets.SVHN(
            root=path,
            split="train",
            download=True
        )

        test_data = datasets.SVHN(
            root=path,
            split="test",
            download=True
        )

        # Force targets to be named "targets" instead of "labels" for compatibility
        setattr(train_data, "targets", train_data.labels)
        setattr(test_data, "targets", test_data.labels)

        train_data = _apply_transforms(train_data, transforms)
        test_data = _apply_transforms(test_data, transforms)

        if transforms is None:
            train_data.data = torch.Tensor(train_data.data / 255.)
            test_data.data = torch.Tensor(test_data.data / 255.)

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             10,
                             onthefly_transforms)

    @classmethod
    def CIFAR10(cls,
                path: str = "../data",
                transforms: Optional[callable] = None,
                onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """
        Load the CIFAR-10 dataset. The dataset is split into training and testing sets according to
        the default split of the :class:`torchvision.datasets.CIFAR10` class.
        If no transformations are provided, the data is normalized to the range [0, 1]. The dataset
        contains 10 classes, corresponding to the following classes: ``airplane``, ``automobile``,
        ``bird``, ``cat``, ``deer``, ``dog``, ``frog``, ``horse``, ``ship``, and ``truck``.
        The images shape is (3, 32, 32).


        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.

        Returns:
            DataContainer: The CIFAR-10 dataset.
        """
        train_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True
        )

        test_data = datasets.CIFAR10(
            root=path,
            train=False,
            download=True
        )

        train_data = _apply_transforms(train_data, transforms)
        test_data = _apply_transforms(test_data, transforms)

        if transforms is None:
            train_data.data = train_data.data / 255.
            test_data.data = test_data.data / 255.

            train_data.data = torch.movedim(train_data.data, 3, 1)
            test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             10,
                             onthefly_transforms)

    @classmethod
    def CINIC10(cls,
                path: str = "../data",
                transforms: Optional[callable] = None,
                onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """
        Load the CINIC-10 dataset. `CINIC-10 <http://dx.doi.org/10.7488/ds/2448>`_ is an
        augmented extension of CIFAR-10. It contains the images from CIFAR-10
        (60,000 images, 32x32 RGB pixels) and a selection of ImageNet database images
        (210,000 images downsampled to 32x32). It was compiled as a 'bridge' between CIFAR-10 and
        ImageNet, for benchmarking machine learning applications. It is split into three equal
        subsets - train, validation, and test - each of which contain 90,000 images.

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.

        Returns:
            DataContainer: The CINIC-10 dataset.
        """

        train_data = support.CINIC10(
            root=path,
            split="train",
            download=True
        )

        test_data = support.CINIC10(
            root=path,
            split="test",
            download=True
        )

        train_data = _apply_transforms(train_data, transforms)
        test_data = _apply_transforms(test_data, transforms)

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             10,
                             onthefly_transforms)

    @classmethod
    def CIFAR100(cls,
                 path: str = "../data",
                 transforms: Optional[callable] = None,
                 onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """
        Load the CIFAR-100 dataset. The dataset is split into training and testing sets according to
        the default split of the :class:`torchvision.datasets.CIFAR100` class.
        If no transformations are provided, the data is normalized to the range [0, 1]. The dataset
        contains 100 classes, corresponding to different type of objects. The images shape is
        (3, 32, 32).

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.

        Returns:
            DataContainer: The CIFAR-100 dataset.
        """
        train_data = datasets.CIFAR100(
            root=path,
            train=True,
            download=True
        )

        test_data = datasets.CIFAR100(
            root=path,
            train=False,
            download=True
        )

        train_data = _apply_transforms(train_data, transforms)
        test_data = _apply_transforms(test_data, transforms)

        if transforms is None:
            train_data.data = train_data.data / 255.
            test_data.data = test_data.data / 255.

            train_data.data = torch.movedim(train_data.data, 3, 1)
            test_data.data = torch.movedim(test_data.data, 3, 1)

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             100,
                             onthefly_transforms)

    @classmethod
    def FASHION_MNIST(cls,
                      path: str = "../data",
                      transforms: Optional[callable] = None,
                      onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """
        Load the Fashion MNIST dataset. The dataset is split into training and testing sets
        according to the default split of the :class:`torchvision.datasets.FashionMNIST` class.
        The dataset contains 10 classes, corresponding to different types of clothing.
        The images shape is (28, 28).

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.

        Returns:
            DataContainer: The CIFAR-100 dataset.
        """
        train_data = datasets.FashionMNIST(
            root=path,
            train=True,
            download=True
        )

        test_data = datasets.FashionMNIST(
            root=path,
            train=False,
            download=True
        )

        train_data = _apply_transforms(train_data, transforms)
        test_data = _apply_transforms(test_data, transforms)

        return DataContainer(train_data.data,
                             train_data.targets,
                             test_data.data,
                             test_data.targets,
                             10,
                             onthefly_transforms)

    @classmethod
    def TINY_IMAGENET(cls,
                      path: str = "../data",
                      transforms: Optional[callable] = None,
                      onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """
        Load the Tiny-ImageNet dataset.
        This version of the dataset is the one offered by the
        `Hugging Face <https://huggingface.co/datasets/zh-plus/tiny-imagenet>`_. The dataset is
        split into training and testing sets according to the default split of the data.
        The dataset contains 200 classes, corresponding to different types of objects.
        The images shape is (3, 64, 64).

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
            transforms (callable, optional): The transformations to apply to the data. Defaults to
              ``None``.
            onthefly_transforms (callable, optional): The transformations to apply on-the-fly to the
              data through the data loader. Defaults to ``None``.

        Returns:
            DataContainer: The Tiny-ImageNet dataset.
        """
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
            if transforms is not None:
                x = transforms(image)
            if not isinstance(image, torch.Tensor):
                x = ToTensor()(image).unsqueeze(0)
            else:
                x = image.unsqueeze(0)
            if x.shape != torch.Size([1, 3, 64, 64]):
                x = fix_bw_trans(x)
            X_train.append(x)
            y_train.append(y)

        for image in track(tiny_imagenet_test, "Loading Tiny ImageNet test data..."):
            y = image['label']
            image = image['image']
            if transforms is not None:
                image = transforms(image)
            if not isinstance(image, torch.Tensor):
                x = ToTensor()(image).unsqueeze(0)
            else:
                x = image.unsqueeze(0)
            if x.shape != torch.Size([1, 3, 64, 64]):
                x = fix_bw_trans(x)
            X_test.append(x)
            y_test.append(y)

        train_data = torch.vstack(X_train)
        test_data = torch.vstack(X_test)

        # if transforms is not None:
        #     train_data = transforms(train_data)
        #     test_data = transforms(test_data)

        idxperm = torch.randperm(train_data.shape[0])
        train_data = train_data[idxperm]
        y_train = torch.LongTensor(y_train)[idxperm]
        y_test = torch.LongTensor(y_test)

        return DataContainer(train_data,
                             y_train,
                             test_data,
                             y_test,
                             200,
                             onthefly_transforms)

    @classmethod
    def FEMNIST(cls,
                path: str = "./data",
                batch_size: int = 10,
                filter: str = "all",
                onthefly_transforms: Optional[callable] = None) -> DataContainer:
        """
        Load the Federated EMNIST (FEMNIST) dataset.
        This dataset is the one offered by the `Leaf project <https://leaf.cmu.edu/>`_.
        FEMNIST contains images of handwritten digits of size 28 by 28 pixels (with option to make
        them all 128 by 128 pixels) taken from 3500 users.
        The dataset has 62 classes corresponding to different characters.
        The label-class correspondence is as follows:

        .. code-block:: rst

            classes: 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
            labels : 01234567890123456789012345678901234567890123456789012345678901
                               10        20        30        40        50        60

        Important:
            Differently from the other datasets (but :meth:`SHAKESPEARE`), the FEMNIST dataset can
            not be downloaded directly from ``fluke`` but it must be downloaded from the
            `Leaf project <https://leaf.cmu.edu/>`_ and stored in the ``path`` folder.
            The datasets must also be created according to the instructions provided by the Leaf
            project. The expected folder structure is:

            .. code-block:: bash

                path
                ├── FEMNIST
                │   ├── train
                │   │   ├── user_data_0.json
                │   │   ├── user_data_1.json
                │   │   └── ...
                │   └── test
                │       ├── user_data_0.json
                │       ├── user_data_1.json
                │       └── ...

            where in each ``user_data_X.json`` file there is a dictionary with the keys
            ``user_data`` containing the data of the user.

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"./data"``.
            batch_size (int, optional): The batch size. Defaults to ``10``.
            filter (str, optional): The filter for the selection of a specific portion of the
                dataset. The options are: ``all``, ``uppercase``, ``lowercase``, and ``digits``.
                Defaults to ``"all"``.

        Returns:
            tuple: A tuple containing the training and testing data loaders for the clients. The
                server data loader is ``None``.
        """
        def _filter_femnist(udata, filter):
            # classes: 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
            # labels : 01234567890123456789012345678901234567890123456789012345678901
            if filter == "all":
                return udata, 62
            elif filter == "uppercase":
                udata["x"] = [x for x, y in zip(udata["x"], udata["y"]) if y < 36 and y > 9]
                udata["y"] = [y - 10 for y in udata["y"] if y < 36 and y > 9]
                num_classes = 26
            elif filter == "lowercase":
                udata["x"] = [x for x, y in zip(udata["x"], udata["y"]) if y > 35]
                udata["y"] = [y - 36 for y in udata["y"] if y > 35]
                num_classes = 26
            elif filter == "digits":
                udata["x"] = [x for x, y in zip(udata["x"], udata["y"]) if y < 10]
                udata["y"] = [y for y in udata["y"] if y < 10]
                num_classes = 10
            else:
                raise ValueError(f"Invalid filter: {filter}")
            return udata, num_classes

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
            udata, num_classes = _filter_femnist(udata, filter)
            Xtr_client = torch.FloatTensor(udata["x"]).reshape(-1, 1, 28, 28)
            ytr_client = torch.LongTensor(udata["y"])
            client_tr_assignments.append(
                FastDataLoader(
                    Xtr_client,
                    ytr_client,
                    num_labels=num_classes,
                    batch_size=batch_size,
                    shuffle=True,
                    transforms=onthefly_transforms,
                    percentage=1.0
                )
            )

        client_te_assignments = []
        for k in track(sorted(dict_train), "Creating testing data loader..."):
            udata = dict_test[k]
            udata, _ = _filter_femnist(udata, filter)
            Xte_client = torch.FloatTensor(udata["x"]).reshape(-1, 1, 28, 28)
            yte_client = torch.LongTensor(udata["y"])
            client_te_assignments.append(
                FastDataLoader(
                    Xte_client,
                    yte_client,
                    num_labels=num_classes,
                    batch_size=64,
                    shuffle=True,
                    transforms=onthefly_transforms,
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
                    batch_size: int = 10,
                    onthefly_transforms: Optional[callable] = None) -> tuple:
        """Load the Federated Shakespeare dataset.
        This dataset is the one offered by the `Leaf project <https://leaf.cmu.edu/>`_.
        Shakespeare is a text dataset containing dialogues from Shakespeare's plays.
        Dialogues are taken from 660 users and the task is to predict the next character in a
        dialogue (which is solved as a classification problem with 100 classes).

        Important:
            Differently from the other datasets (but :meth:`FEMNIST`), the ``SHAKESPEARE`` dataset
            can not be downloaded directly from ``fluke`` but it must be downloaded from the
            `Leaf project <https://leaf.cmu.edu/>`_ and stored in the ``path`` folder.
            The datasets must also be created according to the instructions provided by the Leaf
            project. The expected folder structure is:

            .. code-block:: bash

                path
                ├── shakespeare
                │   ├── train
                │   │   ├── user_data_0.json
                │   │   ├── user_data_1.json
                │   │   └── ...
                │   └── test
                │       ├── user_data_0.json
                │       ├── user_data_1.json
                │       └── ...

            where in each ``user_data_X.json`` file there is a dictionary with the keys
            ``user_data`` containing the data of the user.

        Args:
            path (str, optional): The path where the dataset is stored. Defaults to ``"./data"``.
            batch_size (int, optional): The batch size. Defaults to ``10``.

        Returns:
            tuple: A tuple containing the training and testing data loaders for the clients. The
                server data loader is ``None``.
        """
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
                FastDataLoader(
                    Xtr_client,
                    ytr_client,
                    num_labels=100,
                    batch_size=batch_size,
                    shuffle=True,
                    transforms=onthefly_transforms,
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
                FastDataLoader(
                    Xte_client,
                    yte_client,
                    num_labels=100,
                    batch_size=batch_size,
                    shuffle=True,
                    transforms=onthefly_transforms,
                    percentage=1.0
                )
            )

        perm = permutation(len(client_tr_assignments))
        client_tr_assignments = [client_tr_assignments[i] for i in perm]
        client_te_assignments = [client_te_assignments[i] for i in perm]
        return client_tr_assignments, client_te_assignments, None


Datasets._DATASET_MAP = {
    "mnist": Datasets.MNIST,
    "svhn": Datasets.SVHN,
    "mnistm": Datasets.MNISTM,
    "femnist": Datasets.FEMNIST,
    "emnist": Datasets.EMNIST,
    "cifar10": Datasets.CIFAR10,
    "cifar100": Datasets.CIFAR100,
    "tiny_imagenet": Datasets.TINY_IMAGENET,
    "shakespeare": Datasets.SHAKESPEARE,
    "fashion_mnist": Datasets.FASHION_MNIST,
    "cinic10": Datasets.CINIC10
}

# class DatasetsEnum(Enum):
#     MNIST = "mnist"
#     MNISTM = "mnistm"
#     SVHN = "svhn"
#     FEMNIST = "femnist"
#     EMNIST = "emnist"
#     CIFAR10 = "cifar10"
#     CIFAR100 = "cifar100"
#     TINY_IMAGENET = "tiny_imagenet"
#     SHAKESPEARE = "shakespeare"
#     FASHION_MNIST = "fashion_mnist"
#     CINIC10 = "cinic10"

#     @classmethod
#     def contains(cls, member: object) -> bool:
#         if isinstance(member, str):
#             return member in cls._value2member_map_.keys()
#         elif isinstance(member, DatasetsEnum):
#             return member.value in cls._member_names_

#     def klass(self):
#         DATASET_MAP = {
#             "mnist": Datasets.MNIST,
#             "svhn": Datasets.SVHN,
#             "mnistm": Datasets.MNISTM,
#             "femnist": Datasets.FEMNIST,
#             "emnist": Datasets.EMNIST,
#             "cifar10": Datasets.CIFAR10,
#             "cifar100": Datasets.CIFAR100,
#             "tiny_imagenet": Datasets.TINY_IMAGENET,
#             "shakespeare": Datasets.SHAKESPEARE,
#             "fashion_mnist": Datasets.FASHION_MNIST,
#             "cinic10": Datasets.CINIC10
#         }
#         return DATASET_MAP[self.value]

#     def splitter(self):
#         from . import DataSplitter, DummyDataSplitter

#         # LEAF datasets are already divided into clients
#         if self.value == "femnist" or self.value == "shakespeare":
#             return DummyDataSplitter
#         else:
#             return DataSplitter
