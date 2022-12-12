from enum import Enum
from torch.utils.data import Dataset
from PIL import Image

from torchvision.datasets import MNIST, utils
import os
import os.path
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from typing import List, Tuple
import numpy as np
from numpy.random import randint, shuffle, power, choice, dirichlet, normal, permutation
from sklearn.decomposition import PCA
from scipy.stats.mstats import mquantiles

class Datasets:

    @classmethod
    def MNIST(cls):
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
        return train_data, test_data
    
    @classmethod
    def EMNIST(cls):
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
        return train_data, test_data
    
    @classmethod
    def FEMNIST(cls):
        train_data = FEMNIST(
            root="data",
            train=True, 
            transform=ToTensor(),
            download = True
        )

        test_data = FEMNIST(
            root="data",
            train=False,
            transform=ToTensor(),
            download = True
        )
        return train_data, test_data


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))
        self.data = self.data * 255.

    def _check_exists(self) -> bool:
        return all(
            utils.check_integrity(os.path.join(self.raw_folder, os.path.basename(url))) for url, _ in self.resources
        )

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        training_file = os.path.join(self.raw_folder, self.training_file)
        if not os.path.isfile(training_file):
            shutil.move(training_file, self.processed_folder)
        test_file = os.path.join(self.raw_folder, self.test_file)
        if not os.path.isfile(training_file):
            shutil.move(test_file, self.processed_folder)


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.size = self.tensors[0].shape[0]
        self.batch_size = batch_size
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

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        batch = tuple(t[self.i: self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class Distribution(Enum):
    IID = 1
    QUANTITY_SKEWED = 2
    CLASSWISE_QUANTITY_SKEWED = 3
    LABEL_QUANTITY_SKEWED = 4
    LABEL_DIRICHLET_SKEWED = 5
    LABEL_PATHOLOGICAL_SKEWED = 6
    COVARIATE_SHIFT = 7

class DataSplitter():

    def __init__(self, 
                 X: torch.Tensor,
                 y: torch.Tensor, 
                 n_clients: int, 
                 distribution: Distribution=Distribution.IID,
                 batch_size: int=32,
                 **kwargs):

        self.X, self.y = X, y
        self.n_clients = n_clients
        self.distribution = distribution
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.assign()

    def assign(self):
        assignments = self._iidness_functions[self.distribution](self, self.X, self.y, self.n_clients, **self.kwargs)
        self.client_loader = [FastTensorDataLoader(self.X[assignments[c]], 
                                                   self.y[assignments[c]], 
                                                   batch_size=self.batch_size, 
                                                   shuffle=True) for c in range(self.n_clients)]

    def uniform(self,
                X: torch.Tensor,
                y: torch.Tensor,
                n: int) -> List[np.ndarray]:
        ex_client = X.shape[0] // n
        return [np.array(range(ex_client * n, ex_client*(n+1))) for n in range(n)]


    def quantity_skew(self,
                      X: torch.Tensor,
                      y: torch.Tensor, #not used
                      n: int,
                      min_quantity: int=2,
                      alpha: float=4.) -> List[np.ndarray]:
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
            The labels.
        n: int
            The number of clients upon which the examples are distributed.
        min_quantity: int, default 2
            The minimum quantity of examples to assign to each user.
        alpha: float=4.
            Hyper-parameter of the power law density function  $P(x; a) = a x^{a-1}$.

        Returns
        -------
        n-dimensional list of arrays. The examples' ids assignment.
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
                                 alpha: float=4.) -> List[np.ndarray]:
        assert min_quantity*n <= X.shape[0], "# of instances must be > than min_quantity*n"
        assert min_quantity > 0, "min_quantity must be >= 1"
        labels = list(range(len(set(y))))
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
                            class_per_client: int=2) -> List[np.ndarray]:
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
            The examples.
        y: torch.Tensor
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        class_per_client: int, default 2
            The number of different labels in each client.

        Returns
        -------
        n-dimensional list of arrays. The examples' ids assignment.
        """
        labels = set(y)
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
                             beta: float=.5) -> List[np.ndarray]:
        """
        The function samples p_k ~ Dir_n (beta) and allocate a p_{k,j} proportion of the instances of
        class k to party j. Here Dir(_) denotes the Dirichlet distribution and beta is a
        concentration parameter (beta > 0).
        See: https://arxiv.org/pdf/2102.02079.pdf

        Parameters
        ----------
        X: torch.Tensor
            The examples.
        y: torch.Tensor
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        beta: float, default .5
            The beta parameter of the Dirichlet distribution, i.e., Dir(beta).

        Returns
        -------
        n-dimensional list of arrays. The examples' ids assignment.
        """
        assert beta > 0, "beta must be > 0"
        labels = set(y)
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
                                shards_per_client: int=2) -> List[np.ndarray]:
        """
        The function first sort the data by label, divide it into `n * shards_per_client` shards, and
        assign each of n clients `shards_per_client` shards. This is a pathological non-IID partition
        of the data, as most clients will only have examples of a limited number of classes.
        See: http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf

        Parameters
        ----------
        X: torch.Tensor
            The examples.
        y: torch.Tensor
            The lables.
        n: int
            The number of clients upon which the examples are distributed.
        shards_per_client: int, default 2
            Number of shards per client.

        Returns
        -------
        n-dimensional list of arrays. The examples' ids assignment.
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
                        modes: int=2) -> List[np.ndarray]:
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
        n-dimensional list of arrays. The examples' ids assignment.
        """
        assert 2 <= modes <= n, "modes must be >= 2 and <= n"

        ids_mode = [[] for _ in range(modes)]
        for lbl in set(y):
            ids = np.where(y == lbl)[0]
            X_pca = PCA(n_components=2).fit_transform(X[ids])
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
        Distribution.IID: uniform,
        Distribution.QUANTITY_SKEWED: quantity_skew,
        Distribution.CLASSWISE_QUANTITY_SKEWED: classwise_quantity_skew,
        Distribution.LABEL_QUANTITY_SKEWED: label_quantity_skew,
        Distribution.LABEL_DIRICHLET_SKEWED: label_dirichlet_skew,
        Distribution.LABEL_PATHOLOGICAL_SKEWED: label_pathological_skew,
        Distribution.COVARIATE_SHIFT: covariate_shift
    }
