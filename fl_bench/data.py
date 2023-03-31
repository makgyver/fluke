from __future__ import annotations
from enum import Enum
import warnings

import os
import os.path
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

from typing import List, Tuple
import numpy as np
from PIL import Image
from numpy.random import randint, shuffle, power, choice, dirichlet, permutation
from sklearn.decomposition import PCA
from scipy.stats.mstats import mquantiles


class DataContainer:
    """Container for train and test (classification) data.

    Parameters
    ----------
    X_train : torch.Tensor
        Training data.
    y_train : torch.Tensor
        Training labels.
    X_test : torch.Tensor
        Test data.
    y_test : torch.Tensor
        Test labels.
    """ 
    def __init__(self, 
                 X_train: torch.Tensor, 
                 y_train: torch.Tensor, 
                 X_test: torch.Tensor, 
                 y_test: torch.Tensor,
                 num_classes: int):
        self.train = (X_train, y_train)
        self.test = (X_test, y_test)
        self.num_classes = num_classes


class Datasets:
    """Static class for loading datasets.

    Each dataset is loaded as a DataContainer object.
    """

    @classmethod
    def MNIST(cls) -> DataContainer:
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )

        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor(),
            download = True
        )

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        return DataContainer(train_data.data, 
                             train_data.targets,
                             test_data.data, 
                             test_data.targets, 
                             10)
    
    @classmethod
    def MNIST4D(cls) -> DataContainer:
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )

        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor(),
            download = True
        )

        train_data.data = torch.Tensor(train_data.data / 255.)
        test_data.data = torch.Tensor(test_data.data / 255.)

        train_data.data = train_data.data[:, None, :, :]  # added because probably without the transformation it does not have the correct number of dimension
                                                          # in this way the dimension to the convolutional layer are correct (64,1,28,28)
        test_data.data = test_data.data[:, None, :, :]

        return DataContainer(train_data.data, 
                             train_data.targets,
                             test_data.data, 
                             test_data.targets, 
                             10)
    
    @classmethod
    def MNISTM(cls) -> DataContainer:
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )

        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor(),
            download = True
        )
        return DataContainer(train_data.data / 255., 
                             train_data.targets, 
                             test_data.data / 255., 
                             test_data.targets, 
                             10)
    
    @classmethod
    def EMNIST(cls) -> DataContainer:
        train_data = datasets.EMNIST(
            root="data",
            split="balanced",
            train=True, 
            transform=ToTensor(),
            download = True
        )

        test_data = datasets.EMNIST(
            root="data",
            split="balanced", 
            train=False,
            transform=ToTensor(),
            download = True
        )
        return DataContainer(train_data.data / 255.,
                             train_data.targets, 
                             test_data.data / 255., 
                             test_data.targets, 
                             26)
    
    @classmethod
    def SVHN(cls) -> DataContainer:
        train_data = SVHN(
            root = 'data',
            train = True,
            download = True
        )

        test_data = SVHN(
            root = 'data',
            train = False,
            download = True
        )
        return DataContainer(train_data.data / 255., 
                             train_data.targets, 
                             test_data.data / 255.,
                             test_data.targets, 
                             10)

    @classmethod
    def CIFAR10(cls) -> DataContainer:
        train_data = datasets.CIFAR10(
            root = 'data',
            train = True,
            download = True, 
            transform = None
        )

        test_data = datasets.CIFAR10(
            root = 'data',
            train = False,
            download = True, 
            transform = None
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

    # @classmethod
    # def FEMNIST(cls):
    #     train_data = FEMNIST(
    #         root="data",
    #         train=True, 
    #         transform=ToTensor(),
    #         download = True
    #     )

    #     test_data = FEMNIST(
    #         root="data",
    #         train=False,
    #         transform=ToTensor(),
    #         download = True
    #     )
    #     return train_data, test_data, 62


class SVHN():
    """Load the SVHN dataset.

    Parameters
    ----------
    root : str
        Root directory of dataset where ``processed/training.pt``
        and  ``processed/test.pt`` exist.
    train : bool, optional
        If True, creates dataset from ``training.pt``,
        otherwise from ``test.pt``.
    download : bool, optional
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    """
    def __init__(self, root, train=True, download=True):
        data = datasets.SVHN(
            root = root,
            split = "train" if train else "test",
            download = download,            
        )

        self.data = torch.tensor(data.data)
        self.targets = torch.tensor(data.labels)


# class FEMNIST(MNIST):
#     """
#     This dataset is derived from the Leaf repository
#     (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
#     dataset, grouping examples by writer. Details about Leaf were published in
#     "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
#     """
#     resources = [
#         ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
#          '59c65cec646fc57fe92d27d83afdf0ed')
#     ]

#     def __init__(self, root, train=True, transform=None, target_transform=None,
#                  download=False):
#         super(MNIST, self).__init__(root, transform=transform,
#                                     target_transform=target_transform)
#         self.train = train

#         if download:
#             self.download()

#         if not self._check_exists():
#             raise RuntimeError('Dataset not found.' +
#                                ' You can use download=True to download it')
#         if self.train:
#             data_file = self.training_file
#         else:
#             data_file = self.test_file

#         self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))
#         self.data = self.data * 255.

#     def _check_exists(self) -> bool:
#         return all(
#             utils.check_integrity(os.path.join(self.raw_folder, os.path.basename(url))) for url, _ in self.resources
#         )

#     def download(self):
#         """Download the FEMNIST data if it doesn't exist in processed_folder already."""
#         import shutil

#         if self._check_exists():
#             return

#         os.makedirs(self.raw_folder, exist_ok=True)
#         os.makedirs(self.processed_folder, exist_ok=True)

#         # download files
#         for url, md5 in self.resources:
#             filename = url.rpartition('/')[2]
#             utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

#         # process and save as torch files
#         training_file = os.path.join(self.raw_folder, self.training_file)
#         if not os.path.isfile(training_file):
#             shutil.move(training_file, self.processed_folder)
#         test_file = os.path.join(self.raw_folder, self.test_file)
#         if not os.path.isfile(training_file):
#             shutil.move(test_file, self.processed_folder)


class MNISTM(VisionDataset):
    """MNIST-M Dataset. This dataset is derived from the MNIST dataset by
    applying domain randomization to the original images as described in
    `Unsupervised Domain Adaptation by Backpropagation`_.
    
    .. _`Unsupervised Domain Adaptation by Backpropagation`: https://arxiv.org/abs/1409.7495
    """

    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Parameters
        ----------
        index : int
            Index

        Returns
        -------
        tuple
            (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST-M data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6

    Parameters
    ----------
    *tensors : torch.Tensor
        Tensors to store. Must have the same length @ dim 0.
    batch_size : int
        Batch size to load.
    shuffle : bool
        If True, shuffle the data *in-place* whenever an iterator is created
        out of this object.
    percentage : float
        Percentage of data to use. Useful for debugging.
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False, percentage=1.0):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.size = int(self.tensors[0].shape[0] * percentage)
        self.batch_size = batch_size if batch_size else self.size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.size, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.size)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self) -> tuple:
        if self.i >= self.size:
            raise StopIteration
        batch = tuple(t[self.i: self.i+self.batch_size] for t in self.tensors)
        # Useful in case of batch norm layers
        if batch[0].shape[0] == 1:
            raise StopIteration
        self.i += self.batch_size
        return batch

    def __len__(self) -> int:
        return self.n_batches


class DistributionEnum(Enum):
    """Enum for data distribution across clients."""
    IID = "iid"
    QUANTITY_SKEWED = "qnt"
    CLASSWISE_QUANTITY_SKEWED = "classqnt"
    LABEL_QUANTITY_SKEWED = "lblqnt"
    LABEL_DIRICHLET_SKEWED = "dir"
    LABEL_PATHOLOGICAL_SKEWED = "path"
    COVARIATE_SHIFT = "covshift"

    def __hash__(self) -> int:
        return self.value.__hash__()
    
    def __eq__(self, other) -> bool:
        return self.value == other.value

# # Map distribution to string
# IIDNESS_MAP = {
#     Distribution.IID: "iid",
#     Distribution.QUANTITY_SKEWED: "qnt",
#     Distribution.CLASSWISE_QUANTITY_SKEWED: "classqnt",
#     Distribution.LABEL_QUANTITY_SKEWED: "lblqnt",
#     Distribution.LABEL_DIRICHLET_SKEWED: "dir",
#     Distribution.LABEL_PATHOLOGICAL_SKEWED: "path",
#     Distribution.COVARIATE_SHIFT: "covshift"
# }

# Map string to distribution
# INV_IIDNESS_MAP = {v: k for k, v in IIDNESS_MAP.items()}


class DataSplitter:
    """Class to split data into clients.

    Parameters
    ----------
    X : torch.Tensor
        Data.
    y : torch.Tensor
        Labels.
    n_clients : int
        Number of clients.
    distribution : Distribution
        Distribution (non iidness) of data across clients.
    batch_size : int
        Batch size.
    validation_split : float, optional
        Fraction of data to use for validation/test client-side, by default 0.0.
    sampling_perc : float, optional
        Fraction of data to use, by default 1.0.
    **kwargs
        Additional keyword arguments.
    """
    def __init__(self, 
                 X: torch.Tensor,
                 y: torch.Tensor, 
                 n_clients: int, 
                 distribution: DistributionEnum=DistributionEnum.IID,
                 batch_size: int=32,
                 validation_split: float=0.0,
                 sampling_perc: float=1.0,
                 **kwargs):
        assert X.shape[0] == y.shape[0], "X and y must have the same length."
        assert 0 <= validation_split <= 1, "validation_split must be between 0 and 1."
        assert 0 <= sampling_perc <= 1, "sampling_perc must be between 0 and 1."

        self.X, self.y = X, y
        self.n_clients = n_clients
        self.distribution = distribution
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.sampling_perc = sampling_perc
        self.kwargs = kwargs
        self.assign()

    def assign(self) -> DataSplitter:
        """Assign data to clients."""
        self.assignments = self._iidness_functions[self.distribution](self, self.X, self.y, self.n_clients, **self.kwargs)
        self.client_train_loader = []
        self.client_test_loader = []
        # self.total_training_size = 0
        for c in range(self.n_clients):
            client_X = self.X[self.assignments[c]]
            client_y = self.y[self.assignments[c]]
            if self.validation_split > 0.0:
                X_train, X_test, y_train, y_test = train_test_split(client_X, client_y, test_size=self.validation_split, stratify=client_y)
                self.client_train_loader.append(FastTensorDataLoader(X_train, y_train, batch_size=self.batch_size, shuffle=True, percentage=self.sampling_perc))
                self.client_test_loader.append(FastTensorDataLoader(X_test, y_test, batch_size=self.batch_size, shuffle=True, percentage=self.sampling_perc))
            else:
                self.client_train_loader.append(FastTensorDataLoader(client_X, client_y, batch_size=self.batch_size, shuffle=True, percentage=self.sampling_perc))
                self.client_test_loader.append(None)
            # self.total_training_size += self.client_train_loader[c].size
        return self

# possiamo ottenere la training size dalla size di y(1-val.size)

    def get_loaders(self) -> Tuple[List[FastTensorDataLoader], List[FastTensorDataLoader]]:
        """Get the client-side data loaders.

        Returns
        -------
        Tuple[List[FastTensorDataLoader], List[FastTensorDataLoader]]
            The client-side train and test data loaders.
        """
        return self.client_train_loader, self.client_test_loader

    def uniform(self,
                X: torch.Tensor,
                y: torch.Tensor,
                n: int) -> List[torch.Tensor]:
        """Distribute the examples uniformly across the users.

        Parameters
        ----------
        X: torch.Tensor
            The examples.
        y: torch.Tensor
            The labels. Not used.
        n: int
            The number of clients upon which the examples are distributed.

        Returns
        -------
        List[torch.Tensor]
            The examples' ids assignment.
        """
        ex_client = X.shape[0] // n
        idx = np.random.permutation(X.shape[0])
        return [idx[range(ex_client*i, ex_client*(i+1))] for i in range(n)]


    def quantity_skew(self,
                      X: torch.Tensor,
                      y: torch.Tensor, #not used
                      n: int,
                      min_quantity: int=2,
                      alpha: float=4.) -> List[torch.Tensor]:
        """
        Distribute the examples across the users according to the following probability density function:
        $P(x; a) = a x^{a-1}$
        where x is the id of a client (x in [0, n-1]), and a = `alpha` > 0 with
        - alpha = 1  => examples are equidistributed across clients;
        - alpha = 2  => the examples are "linearly" distributed across users;
        - alpha >= 3 => the examples are power law distributed;
        - alpha -> \infty => all users but one have `min_quantity` examples, and the remaining user all the rest.
        Each client is guaranteed to have at least `min_quantity` examples.

        Parameters
        ----------
        X: torch.Tensor
            The examples.
        y: torch.Tensor
            The labels. Not used.
        n: int
            The number of clients upon which the examples are distributed.
        min_quantity: int, default 2
            The minimum quantity of examples to assign to each user.
        alpha: float=4.
            Hyper-parameter of the power law density function  $P(x; a) = a x^{a-1}$.

        Returns
        -------
        List[torch.Tensor]
            The examples' ids assignment.
        """
        assert min_quantity*n <= X.shape[0], "# of instances must be > than min_quantity*n"
        assert min_quantity > 0, "min_quantity must be >= 1"
        s = np.array(power(alpha, X.shape[0] - min_quantity*n) * n, dtype=int)
        m = np.array([[i] * min_quantity for i in range(n)]).flatten()
        assignment = np.concatenate([s, m])
        shuffle(assignment)
        return [np.where(assignment == i)[0] for i in range(n)]


    def classwise_quantity_skew(self,
                                 X: torch.Tensor,
                                 y: torch.Tensor,
                                 n: int,
                                 min_quantity: int=2,
                                 alpha: float=4.) -> List[torch.Tensor]:
        assert min_quantity*n <= X.shape[0], "# of instances must be > than min_quantity*n"
        assert min_quantity > 0, "min_quantity must be >= 1"
        labels = list(range(len(torch.unique(y).numpy())))
        lens = [np.where(y == l)[0].shape[0] for l in labels]
        min_lbl = min(lens)
        assert min_lbl >= n, "Under represented class!"

        s = [np.array(power(alpha, lens[c] - n) * n, dtype=int) for c in labels]
        assignment = []
        for c in labels:
            ass = np.concatenate([s[c], list(range(n))])
            shuffle(ass)
            assignment.append(ass)

        res = [[] for _ in range(n)]
        for c in labels:
            idc = np.where(y == c)[0]
            for i in range(n):
                res[i] += list(idc[np.where(assignment[c] == i)[0]])

        return [np.array(r, dtype=int) for r in res]


    def label_quantity_skew(self,
                            X: torch.Tensor, #not used
                            y: torch.Tensor,
                            n: int,
                            class_per_client: int=2) -> List[torch.Tensor]:
        """
        Suppose each party only has data samples of `class_per_client` (i.e., k) different labels.
        We first randomly assign k different label IDs to each party. Then, for the samples of each
        label, we randomly and equally divide them into the parties which own the label.
        In this way, the number of labels in each party is fixed, and there is no overlap between
        the samples of different parties.
        See: https://arxiv.org/pdf/2102.02079.pdf

        Parameters
        ----------
        X: torch.Tensor
            The examples. Not used.
        y: torch.Tensor
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        class_per_client: int, default 2
            The number of different labels in each client.

        Returns
        -------
        List[torch.Tensor]
            The examples' ids assignment.
        """
        labels = set(torch.unique(y).numpy())
        assert 0 < class_per_client <= len(labels), "class_per_client must be > 0 and <= #classes"
        assert class_per_client * n >= len(labels), "class_per_client * n must be >= #classes"
        nlbl = [choice(len(labels), class_per_client, replace=False)  for u in range(n)]
        check = set().union(*[set(a) for a in nlbl])
        while len(check) < len(labels):
            missing = labels - check
            for m in missing:
                nlbl[randint(0, n)][randint(0, class_per_client)] = m
            check = set().union(*[set(a) for a in nlbl])
        class_map = {c:[u for u, lbl in enumerate(nlbl) if c in lbl] for c in labels}
        assignment = np.zeros(y.shape[0])
        for lbl, users in class_map.items():
            ids = np.where(y == lbl)[0]
            assignment[ids] = choice(users, len(ids))
        return [np.where(assignment == i)[0] for i in range(n)]


    def label_dirichlet_skew(self,
                             X: torch.Tensor,
                             y: torch.Tensor,
                             n: int,
                             beta: float=.1) -> List[torch.Tensor]:
        """
        The function samples p_k ~ Dir_n (beta) and allocate a p_{k,j} proportion of the instances of
        class k to party j. Here Dir(_) denotes the Dirichlet distribution and beta is a
        concentration parameter (beta > 0).
        See: https://arxiv.org/pdf/2102.02079.pdf

        Parameters
        ----------
        X: torch.Tensor
            The examples. Not used.
        y: torch.Tensor
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        beta: float, default .5
            The beta parameter of the Dirichlet distribution, i.e., Dir(beta).

        Returns
        -------
        List[torch.Tensor]
            The examples' ids assignment.
        """
        assert beta > 0, "beta must be > 0"
        labels = set(torch.unique(y).numpy())
        pk = {c: dirichlet([beta]*n, size=1)[0] for c in labels}
        assignment = np.zeros(y.shape[0])
        for c in labels:
            ids = np.where(y == c)[0]
            shuffle(ids)
            shuffle(pk[c])
            assignment[ids[n:]] = choice(n, size=len(ids)-n, p=pk[c])
            assignment[ids[:n]] = list(range(n))

        return [np.where(assignment == i)[0] for i in range(n)]


    def label_pathological_skew(self,
                                X: torch.Tensor, #not used
                                y: torch.Tensor,
                                n: int,
                                shards_per_client: int=2) -> List[torch.Tensor]:
        """
        The function first sort the data by label, divide it into `n * shards_per_client` shards, and
        assign each of n clients `shards_per_client` shards. This is a pathological non-IID partition
        of the data, as most clients will only have examples of a limited number of classes.
        See: http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf

        Parameters
        ----------
        X: torch.Tensor
            The examples. Not used.
        y: torch.Tensor
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        shards_per_client: int, default 2
            Number of shards per client.

        Returns
        -------
        List[torch.Tensor]
            The examples' ids assignment.
        """
        sorted_ids = np.argsort(y)
        n_shards = int(shards_per_client * n)
        shard_size = int(np.ceil(len(y) / n_shards))
        assignments = np.zeros(y.shape[0])
        perm = permutation(n_shards)
        j = 0
        for i in range(n):
            for _ in range(shards_per_client):
                left = perm[j] * shard_size
                right = min((perm[j]+1) * shard_size, len(y))
                assignments[sorted_ids[left:right]] = i
                j += 1
        return [np.where(assignments == i)[0] for i in range(n)]


    def covariate_shift(self,
                        X: torch.Tensor,
                        y: torch.Tensor,
                        n: int,
                        modes: int=2) -> List[torch.Tensor]:
        """
        The function first extracts the first principal component (through PCA) and then divides it in
        `modes` percentiles. To each user, only examples from a single mode are selected (uniformly).
        
        Parameters
        ----------
        X: torch.Tensor
            The examples.
        y: torch.Tensor
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        modes: int, default 2
            Number of different modes to consider in the input data first principal component.
        
        Returns
        -------
        List[torch.Tensor]
            The examples' ids assignment.
        """
        assert 2 <= modes <= n, "modes must be >= 2 and <= n"

        ids_mode = [[] for _ in range(modes)]
        for lbl in set(torch.unique(y).numpy()):
            ids = np.where(y == lbl)[0]
            X_pca = PCA(n_components=2).fit_transform(X.view(X.size()[0], -1)[ids])
            quantiles = mquantiles(X_pca[:, 0], prob=np.linspace(0, 1, num=modes+1)[1:-1])

            y_ = np.zeros(y[ids].shape)
            for i, q in enumerate(quantiles):
                if i == 0: continue
                id_pos = np.where((quantiles[i-1] < X_pca[:, 0]) & (X_pca[:, 0] <= quantiles[i]))[0]
                y_[id_pos] = i
            y_[np.where(X_pca[:, 0] > quantiles[-1])[0]] = modes-1

            for m in range(modes):
                ids_mode[m].extend(ids[np.where(y_ == m)[0]])

        ass_mode = (list(range(modes)) * int(np.ceil(n/modes)))[:n]
        shuffle(ass_mode)
        mode_map = {m:[u for u, mu in enumerate(ass_mode) if mu == m] for m in range(modes)}
        assignment = np.zeros(y.shape[0])
        for mode, users in mode_map.items():
            ids = ids_mode[mode]
            assignment[ids] = choice(users, len(ids))
        return [np.where(assignment == i)[0] for i in range(n)]
    

    _iidness_functions = {
        DistributionEnum.IID: uniform,
        DistributionEnum.QUANTITY_SKEWED: quantity_skew,
        DistributionEnum.CLASSWISE_QUANTITY_SKEWED: classwise_quantity_skew,
        DistributionEnum.LABEL_QUANTITY_SKEWED: label_quantity_skew,
        DistributionEnum.LABEL_DIRICHLET_SKEWED: label_dirichlet_skew,
        DistributionEnum.LABEL_PATHOLOGICAL_SKEWED: label_pathological_skew,
        DistributionEnum.COVARIATE_SHIFT: covariate_shift
    }


# Mapping between dataset names and their corresponding enum value

class DatasetsEnum(Enum):
    MNIST = "mnist"
    MNIST4D = "mnist4d"
    MNISTM = "mnistm"
    SVHN = "svhn"
    # FEMNIST = "femnist"
    EMNIST = "emnist"
    CIFAR10 = "cifar10"

    def klass(self):
        DATASET_MAP = {
            "mnist": Datasets.MNIST,
            "mnistm": Datasets.MNISTM,
            "mnist4d": Datasets.MNIST4D,
            "svhn": Datasets.SVHN,
            # "femnist": Datasets.FEMNIST,
            "emnist": Datasets.EMNIST,
            "cifar10": Datasets.CIFAR10
        } 
        return DATASET_MAP[self.value]



