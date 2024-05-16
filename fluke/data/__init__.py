"""This module contains the data utilities for ``fluke``."""
from __future__ import annotations
from scipy.stats.mstats import mquantiles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.random import randint, shuffle, power, choice, dirichlet, permutation
import numpy as np
from typing import Sequence, Optional
from rich.progress import track
import rich
import torch
from sklearn.model_selection import train_test_split
import sys

sys.path.append(".")
sys.path.append("..")


from .. import DDict  # NOQA
# from .datasets import Datasets  # NOQA

__all__ = [
    'datasets',
    'support',
    'DataContainer',
    'FastDataLoader',
    'DataSplitter',
    'DummyDataSplitter'
]


class DataContainer:
    """Container for train and test (classification) data.

    Args:
        X_train (torch.Tensor): The training data.
        y_train (torch.Tensor): The training labels.
        X_test (torch.Tensor): The test data.
        y_test (torch.Tensor): The test labels.
        num_classes (int): The number of classes.
    """

    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 X_test: torch.Tensor,
                 y_test: torch.Tensor,
                 num_classes: int):
        self.train = (X_train, y_train)
        self.test = (X_test, y_test)
        self.num_features = np.prod([i for i in X_train.shape[1:]]).item()
        self.num_classes = num_classes

    def standardize(self):
        """Standardize the data.
        The data is standardized using the ``StandardScaler`` from ``sklearn``. The method modifies
        the :attr:`train` and :attr:`test` attributes.
        """
        data_train, data_test = self.train[0], self.test[0]
        scaler = StandardScaler()
        scaler.fit(data_train)
        self.train = (torch.FloatTensor(scaler.transform(data_train)), self.train[1])
        self.test = (torch.FloatTensor(scaler.transform(data_test)), self.test[1])


class FastDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    Important:
        This type of data loader does not support the application of different transformations
        to the data at each iteration. If you need to apply different transformations to the data
        at each iteration, you should use the standard PyTorch ``DataLoader``.


    Note:
        This implementation is based on the following discussion:
        https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6

    Args:
        *tensors (Sequence[torch.Tensor]): tensors to be loaded.
        batch_size (int): batch size.
        shuffle (bool): whether the data should be shuffled.
        percentage (float): the percentage of the data to be used.
        skip_singleton (bool): whether to skip batches with a single element. If you have batchnorm
            layers, you might want to set this to ``True``.
        single_batch (bool): whether to return a single batch at each generator iteration.

    Caution:
        When sampling a percentage of the data (i.e., ``percentage < 1``), the data is sampled at
        each epoch. This means that the data varies at each epoch. If you want to keep the data
        constant across epochs, you should sample the data once and pass the sampled
        data to the :class:`FastDataLoader` and set the ``percentage`` parameter to ``1.0``.

    Attributes:
        tensors (Sequence[torch.Tensor]): Tensors of the dataset. Ideally, the first tensor should
            be the input data, and the second tensor should be the labels. However, this is not
            enforced and the user is responsible for ensuring that the tensors are used correctly.
        batch_size (int): batch size.
        shuffle (bool): whether the data should be shuffled at each epoch. If ``True``, the data is
            shuffled at each iteration.
        percentage (float): the percentage of the data to be used. If `1.0`, all the data is used.
            Otherwise, the data is sampled according to the given percentage. **Note that the
            sampled data varies at each epoch.**
        skip_singleton (bool): whether to skip batches with a single element. If you have batchnorm
            layers, you might want to set this to ``True``.
        single_batch (bool): whether to return a single batch at each generator iteration.
        size (int): the size of the dataset according to the percentage of the data to be used.
        max_size (int): the total size of the dataset.

    Raises:
        AssertionError: if the tensors do not have the same size along the first dimension.
    """

    def __init__(self,
                 *tensors: torch.Tensor,
                 num_labels: int,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 percentage: float = 1.0,
                 skip_singleton: bool = True,
                 single_batch: bool = False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors), \
            "All tensors must have the same size along the first dimension."
        self.tensors: Sequence[torch.Tensor] = tensors
        self.num_labels: int = num_labels
        self.max_size = self.tensors[0].shape[0]
        self.set_sample_size(percentage)
        self.shuffle: bool = shuffle
        self.skip_singleton: bool = skip_singleton
        self.batch_size: int = batch_size if batch_size > 0 else self.size
        self.single_batch: bool = single_batch

    def __getitem__(self, index: int) -> tuple:
        """Get the entry at the given index for each tensor.

        Args:
            index (int): the index.

        Raises:
            IndexError: if the index is out of bounds.

        Returns:
            tuple: the entry at the given index for each tensor.
        """
        if index >= self.max_size:
            raise IndexError("Index out of bounds.")
        return tuple(t[index] for t in self.tensors)

    def set_sample_size(self, percentage: float) -> int:
        """Set the sample size.

        Args:
            percentage (float): the percentage of the data to be used.

        Returns:
            int: the sample size.
        """
        if percentage > 1.0 or percentage <= 0.0:
            raise ValueError("percentage must be in (0, 1]")
        self.size = max(int(self.tensors[0].shape[0] * percentage), 1)
        return self.size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if value <= 0:
            raise ValueError("batch_size must be > 0")
        self._batch_size = value
        n_batches, remainder = divmod(self.size, self._batch_size)
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
        if self.single_batch and self.i > 0:
            raise StopIteration
        if self.i >= self.size:
            raise StopIteration
        batch = tuple(t[self.i: self.i+self._batch_size] for t in self.tensors)
        # Useful in case of batch norm layers
        if self.skip_singleton and batch[0].shape[0] == 1:
            raise StopIteration
        self.i += self._batch_size
        return batch

    def __len__(self) -> int:
        return self.n_batches


# class DistributionEnum(Enum):
#     """Enum for data distribution across clients."""
#     IID = "iid"  # : Independent and Identically Distributed data.
#     QUANTITY_SKEWED = "qnt"  # : Quantity skewed data.
#     CLASSWISE_QUANTITY_SKEWED = "classqnt"  # : Class-wise quantity skewed data.
#     LABEL_QUANTITY_SKEWED = "lblqnt"  # : Label quantity skewed data.
#     LABEL_DIRICHLET_SKEWED = "dir"  # : Label skewed data according to the Dirichlet distribution.
#     # : Pathological skewed data (i.e., each client has data from a small subset of the classes).
#     LABEL_PATHOLOGICAL_SKEWED = "path"
#     COVARIATE_SHIFT = "covshift"  # : Covariate shift skewed data.

#     def __hash__(self) -> int:
#         return self.value.__hash__()

#     def __eq__(self, other) -> bool:
#         return self.value == other.value


class DataSplitter:
    """Utility class for splitting the data across clients."""

    def _safe_train_test_split(self,
                               X: torch.Tensor,
                               y: torch.Tensor,
                               test_size: float,
                               client_id: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            if test_size == 0.0:
                return X, None, y, None
            else:
                return train_test_split(X, y, test_size=test_size, stratify=y)
        except ValueError:
            client_str = f"[Client {client_id}]" if client_id is not None else ""
            rich.print(
                f"[bold red]Warning{client_str}: [/bold red] Stratified split failed. " +
                "Falling back to random split."
            )
            return train_test_split(X, y, test_size=test_size)

    def __init__(self,
                 dataset: DataContainer,
                 distribution: str = "iid",
                 client_split: float = 0.0,
                 sampling_perc: float = 1.0,
                 server_test: bool = True,
                 keep_test: bool = True,
                 server_split: float = 0.0,
                 uniform_test: bool = False,
                 dist_args: DDict = None):
        """Initialize the ``DataSplitter``.

        Args:
            dataset (DataContainer or str): The dataset.
            distribution (str, optional): The data distribution function. Defaults to
                ``"iid"``.
            client_split (float, optional): The size of the client's test set. Defaults to 0.0.
            sampling_perc (float, optional): The percentage of the data to be used. Defaults to 1.0.
            server_test (bool, optional): Whether to keep a server test set. Defaults to True.
            keep_test (bool, optional): Whether to keep the test set provided by the dataset.
                Defaults to True.
            server_split (float, optional): The size of the server's test set. Defaults to 0.0. This
                parameter is used only if ``server_test`` is ``True`` and ``keep_test`` is
                ``False``.
            uniform_test (bool, optional): Whether to distribute the test set in a IID across the
                clients. If ``False``, the test set is distributed according to the distribution
                function. Defaults to False.
            builder_args (DDict, optional): The arguments for the dataset class. Defaults to None.
            dist_args (DDict, optional): The arguments for the distribution function. Defaults to
                None.

        Raises:
            AssertionError: If the parameters are not in the correct range or configuration.
        """
        assert 0 <= client_split <= 1, "client_split must be between 0 and 1."
        assert 0 <= sampling_perc <= 1, "sampling_perc must be between 0 and 1."
        assert 0 <= server_split <= 1, "server_split must be between 0 and 1."
        if not keep_test and server_test and server_split == 0.0:
            raise AssertionError(
                "server_split must be > 0.0 if server_test is True and keep_test is False.")
        if not server_test and client_split == 0.0:
            raise AssertionError("Either client_split > 0 or server_test = True must be true.")

        self.data_container = dataset
        self.distribution = distribution
        self.client_split = client_split
        self.sampling_perc = sampling_perc
        self.keep_test = keep_test
        self.server_test = server_test
        self.server_split = server_split
        self.uniform_test = uniform_test
        self.dist_args = dist_args if dist_args is not None else DDict()

    # def num_features(self) -> int:
    #     return self.data_container.num_features

    @ property
    def num_classes(self) -> int:
        """Return the number of classes of the dataset.

        Returns:
            int: The number of classes.
        """
        return self.data_container.num_classes

    def assign(self,
               n_clients: int,
               batch_size: int = 32) -> tuple[tuple[FastDataLoader,
                                                    Optional[FastDataLoader]],
                                              FastDataLoader]:
        """Assign the data to the clients and the server according to the configuration.
        Specifically, we can have the following scenarios:

        1. ``server_test = True`` and ``keep_test = True``: The server has a test set that
           corresponds to the test set of the dataset. The clients have a training set and,
           if ``client_split > 0``, a test set.
        2. ``server_test = True`` and ``keep_test = False``: The server has a test set that
           is sampled from the test set of whole dataset (training set and test set are merged). The
           sampling is done according to the ``server_split`` parameter. The clients have a training
           set and, if ``client_split > 0``, a test set.
        3. ``server_test = False`` and ``keep_test = True``: The server does not have a test set.
           The clients have a training set and a test set that corresponds to the test set of the
           dataset distributed uniformly across the clients. In this case the ``client_split`` is
           ignored.
        4. ``server_test = False`` and ``keep_test = False``: The server does not have a test set.
           The clients have a training set and, if ``client_split > 0``, a test set.

        If ``uniform_test = False``, the training and test set are distributed across the clients
        according to the provided distribution. The only exception is done for the test set in
        scenario 3. The test set is IID distributed across clients if ``uniform_test = True``.

        Args:
            n_clients (int): The number of clients.
            batch_size (Optional[int], optional): The batch size. Defaults to 32.

        Returns:
            tuple[tuple[FastDataLoader, Optional[FastDataLoader]],
                  FastDataLoader]: The clients' training and testing assignments and the
                  server's testing assignment.
        """
        if self.server_test and self.keep_test:
            server_X, server_Y = self.data_container.test
            client_X, client_Y = self.data_container.train
            client_Xtr, client_Xte, client_Ytr, client_Yte = self._safe_train_test_split(
                client_X, client_Y, test_size=self.client_split)
        elif not self.keep_test:
            Xtr, ytr = self.data_container.train
            Xte, yte = self.data_container.test
            # Merge and shuffle the data
            X, Y = torch.cat((Xtr, Xte), dim=0), torch.cat((ytr, yte), dim=0)
            idx = torch.randperm(X.size(0))
            X, Y = X[idx], Y[idx]
            # Split the data
            if self.server_test:
                client_X, server_X, client_Y, server_Y = train_test_split(
                    X, Y, test_size=self.server_split)
            else:
                server_X, server_Y = None, None
                client_X, client_Y = X, Y
            client_Xtr, client_Xte, client_Ytr, client_Yte = self._safe_train_test_split(
                client_X, client_Y, test_size=self.client_split)

        else:  # keep_test and not server_test
            server_X, server_Y = None, None
            client_Xtr, client_Ytr = self.data_container.train
            client_Xte, client_Yte = self.data_container.test

        # Clients have test set
        if client_Xte is not None:
            if self.uniform_test:
                assignments_te = self.iid(client_Xte, client_Yte, n_clients)
            else:
                assignments_te = self._iidness_functions[self.distribution](
                    self, client_Xte, client_Yte, n_clients, **self.dist_args)
        else:  # otherwise
            assignments_te = None

        assignments_tr = self._iidness_functions[self.distribution](
            self, client_Xtr, client_Ytr, n_clients, **self.dist_args)

        client_tr_assignments = []
        client_te_assignments = []
        for c in range(n_clients):
            Xtr_client, Ytr_client = client_Xtr[assignments_tr[c]], client_Ytr[assignments_tr[c]]
            client_tr_assignments.append(FastDataLoader(Xtr_client,
                                                        Ytr_client,
                                                        num_labels=self.num_classes,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        percentage=self.sampling_perc))
            if assignments_te is not None:
                Xte_client = client_Xte[assignments_te[c]]
                Yte_client = client_Yte[assignments_te[c]]
                client_te_assignments.append(FastDataLoader(Xte_client,
                                                            Yte_client,
                                                            num_labels=self.num_classes,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            percentage=self.sampling_perc))
            else:
                client_te_assignments.append(None)

        server_te = FastDataLoader(server_X, server_Y,
                                   num_labels=self.num_classes,
                                   batch_size=128,
                                   shuffle=True,
                                   percentage=self.sampling_perc) if self.server_test \
            else None
        return (client_tr_assignments, client_te_assignments), server_te

    def iid(self,
            X: torch.Tensor,
            y: torch.Tensor,
            n: int) -> list[torch.Tensor]:
        """Distribute the examples uniformly across the users.

        Args:
            X (torch.Tensor): The examples.
            y (torch.Tensor): The labels.
            n (int): The number of clients upon which the examples are distributed.

        Returns:
            list[torch.Tensor]: The examples' ids assignment.
        """
        assert X.shape[0] >= n, "# of instances must be > than #clients"

        ex_client = X.shape[0] // n
        idx = np.random.permutation(X.shape[0])
        assignment = [idx[range(ex_client*i, ex_client*(i+1))] for i in range(n)]
        # Assign the remaining examples to the first clients
        if X.shape[0] % n > 0:
            assignment[0] = np.concatenate((assignment[0], idx[ex_client*n:]))
        return assignment

    def quantity_skew(self,
                      X: torch.Tensor,
                      y: torch.Tensor,  # not used
                      n: int,
                      min_quantity: int = 2,
                      alpha: float = 4.) -> list[torch.Tensor]:
        r"""
        Distribute the examples across the users according to the following probability density
        function: :math:`P(x; a) = a x^{a-1}`
        where :math:`x` is the id of a client (:math:`x \in [0, n-1]`), and ``a = alpha > 0`` with

        - ``alpha = 1``: examples are equidistributed across clients;
        - ``alpha = 2``: the examples are "linearly" distributed across users;
        - ``alpha >= 3``: the examples are power law distributed;
        - ``alpha`` :math:`\rightarrow \infty`: all users but one have ``min_quantity`` examples,
          and the remaining user all the rest.

        Each client is guaranteed to have at least ``min_quantity`` examples.

        Args:
            X (torch.Tensor): The examples.
            y (torch.Tensor): The labels. Not used.
            n (int): The number of clients upon which the examples are distributed.
            min_quantity (int, optional): The minimum number of examples per client. Defaults to 2.
            alpha (float, optional): The skewness parameter. Defaults to 4.

        Returns:
            list[torch.Tensor]: The examples' ids assignment.
        """  # noqa: W605
        # The abow comment is to avoid flake8 error W605 (invalid escape sequence)
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
                                min_quantity: int = 2,
                                alpha: float = 4.) -> list[torch.Tensor]:
        """Class-wise quantity skewed data distribution. This type of skewness is similar to the
        quantity skewness, but it is applied to each class separately.
        This method distribute the examples of each class across the users according to the
        following probability density function:
        :math:`P(x; a) = a x^{a-1}` where :math:`x` is the id of a client :math:`(x \in [0, n-1])`,
        and ``a = alpha > 0`` with

        - ``alpha = 1``: examples are equidistributed across clients;
        - ``alpha = 2``: the examples are "linearly" distributed across users;
        - ``alpha >= 3``: the examples are power law distributed;

        Args:
            X (torch.Tensor): The examples.
            y (torch.Tensor): The labels.
            n (int): The number of clients upon which the examples are distributed.
            min_quantity (int, optional): The minimum number of examples per client. Defaults to 2.
            alpha (float, optional): The skewness parameter. Defaults to 4.

        Returns:
            list[torch.Tensor]: The examples' ids assignment.
        """  # noqa: W605
        # The abow comment is to avoid flake8 error W605 (invalid escape sequence)
        assert min_quantity*n <= X.shape[0], "# of instances must be > than min_quantity*n"
        assert min_quantity > 0, "min_quantity must be >= 1"

        labels = list(range(len(torch.unique(torch.LongTensor(y)).numpy())))
        lens = [np.where(y == lbl)[0].shape[0] for lbl in labels]
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
                            X: torch.Tensor,  # not used
                            y: torch.Tensor,
                            n: int,
                            class_per_client: int = 2) -> list[torch.Tensor]:
        """
        This method distribute the data across client according to a specific type of skewness of
        the lables. Specifically:
        suppose each party only has data samples of ``class_per_client`` different labels.
        We first randomly assign k different label IDs to each party. Then, for the samples of each
        label, we randomly and equally divide them into the parties which own the label.
        In this way, the number of labels in each party is fixed, and there is no overlap between
        the samples of different parties.
        See: https://arxiv.org/pdf/2102.02079.pdf

        Args:
            X (torch.Tensor): The examples. Not used.
            y (torch.Tensor): The lables.
            n (int): The number of clients upon which the examples are distributed.
            class_per_client (int, optional): The number of classes per client. Defaults to 2.

        Returns:
            list[torch.Tensor]: The examples' ids assignment.
        """
        labels = set(torch.unique(torch.LongTensor(y)).numpy())
        assert 0 < class_per_client <= len(labels), "class_per_client must be > 0 and <= #classes"
        assert class_per_client * n >= len(labels), "class_per_client * n must be >= #classes"
        nlbl = [choice(len(labels), class_per_client, replace=False) for u in range(n)]
        check = set().union(*[set(a) for a in nlbl])
        while len(check) < len(labels):
            missing = labels - check
            for m in missing:
                nlbl[randint(0, n)][randint(0, class_per_client)] = m
            check = set().union(*[set(a) for a in nlbl])
        class_map = {c: [u for u, lbl in enumerate(nlbl) if c in lbl] for c in labels}
        assignment = np.zeros(y.shape[0])
        for lbl, users in class_map.items():
            ids = np.where(y == lbl)[0]
            assignment[ids] = choice(users, len(ids))
        return [np.where(assignment == i)[0] for i in range(n)]

    def label_dirichlet_skew(self,
                             X: torch.Tensor,
                             y: torch.Tensor,
                             n: int,
                             beta: float = .1,
                             min_ex_class: int = 2) -> list[torch.Tensor]:
        r"""
        The method samples :math:`p_k \sim \text{Dir}_n(\beta)` and allocates a :math:`p_{k,j}`
        proportion of the instances of class :math:`k` to party :math:`j`. Here
        :math:`\text{Dir}(\cdot)` denotes the Dirichlet distribution and beta is a concentration
        parameter :math:`(\beta > 0)`.
        See: https://arxiv.org/pdf/2102.02079.pdf

        Args:
            X (torch.Tensor): The examples. Not used - Allows for a common interface with other
              methods.
            y (torch.Tensor): The lables.
            n (int): The number of clients upon which the examples are distributed.
            beta (float, optional): The concentration parameter. Defaults to 0.1.
            min_ex_class (int, optional): The minimum number of examples per class. Defaults to 2.

        Returns:
            list[torch.Tensor]: The examples' ids assignment.
        """
        assert beta > 0, "beta must be > 0"
        labels = set(torch.unique(torch.LongTensor(y)).numpy())
        pk = {c: dirichlet([beta]*n) for c in labels}
        assignment = np.zeros(y.shape[0])
        for c in labels:
            ids = np.where(y == c)[0]
            shuffle(ids)
            shuffle(pk[c])
            fixed = n * min_ex_class
            assignment[ids[fixed:]] = choice(n, size=len(ids)-fixed, p=pk[c])
            assignment[ids[:fixed]] = list(range(n)) * min_ex_class

        return [np.where(assignment == i)[0] for i in range(n)]

    def label_pathological_skew(self,
                                X: torch.Tensor,  # not used
                                y: torch.Tensor,
                                n: int,
                                shards_per_client: int = 2) -> list[torch.Tensor]:
        """
        The method first sort the data by label, divide it into ``n * shards_per_client`` shards,
        and assign each of ``n`` clients ``shards_per_client`` shards. This is a pathological
        non-IID partition of the data, as most clients will only have examples of a limited number
        of classes.
        See: http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf

        Args:
            X (torch.Tensor): The examples. Not used.
            y (torch.Tensor): The lables.
            n (int): The number of clients upon which the examples are distributed.
            shards_per_client (int, optional): The number of shards per client. Defaults to 2.

        Returns:
            list[torch.Tensor]: The examples' ids assignment.
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
                        modes: int = 2) -> list[torch.Tensor]:
        """
        This method first extracts the first principal component (through PCA) and then divides it
        in ``modes`` percentiles. To each user, only examples from a single mode are selected
        (uniformly).

        Attention:
            This type of skewness is not present in the literature and this method may also
            be not very efficient.

        Args:
            X (torch.Tensor): The examples.
            y (torch.Tensor): The lables.
            n (int): The number of clients upon which the examples are distributed.
            modes (int, optional): The number of modes. Defaults to 2.

        Returns:
            list[torch.Tensor]: The examples' ids assignment.
        """
        assert 2 <= modes <= n, "modes must be >= 2 and <= n"

        ids_mode = [[] for _ in range(modes)]
        for lbl in track(set(torch.unique(torch.LongTensor(y)).numpy()),
                         "Simulating Covariate Shift..."):
            ids = np.where(y == lbl)[0]
            X_pca = PCA(n_components=2).fit_transform(X.view(X.size()[0], -1)[ids])
            quantiles = mquantiles(X_pca[:, 0], prob=np.linspace(0, 1, num=modes+1)[1:-1])

            y_ = np.zeros(y[ids].shape)
            for i, q in enumerate(quantiles):
                if i == 0:
                    continue
                id_pos = np.where((quantiles[i-1] < X_pca[:, 0]) & (X_pca[:, 0] <= quantiles[i]))[0]
                y_[id_pos] = i
            y_[np.where(X_pca[:, 0] > quantiles[-1])[0]] = modes-1

            for m in range(modes):
                ids_mode[m].extend(ids[np.where(y_ == m)[0]])

        ass_mode = (list(range(modes)) * int(np.ceil(n/modes)))[:n]
        shuffle(ass_mode)
        mode_map = {m: [u for u, mu in enumerate(ass_mode) if mu == m] for m in range(modes)}
        assignment = np.zeros(y.shape[0])
        for mode, users in mode_map.items():
            ids = ids_mode[mode]
            assignment[ids] = choice(users, len(ids))
        return [np.where(assignment == i)[0] for i in range(n)]

    _iidness_functions = {
        "iid": iid,
        "qnt": quantity_skew,
        "classwise_qnt": classwise_quantity_skew,
        "lbl_qnt": label_quantity_skew,
        "dir": label_dirichlet_skew,
        "pathological": label_pathological_skew,
        "covariate": covariate_shift
    }


class DummyDataSplitter(DataSplitter):
    """
    This data splitter assumes that the data is already pre-assigned to the clients.
    This must be used in the case you start with a pre-divided datasets that you want to use as
    is (e.g., FEMNIST and Shakespeare).
    """

    def __init__(self,
                 dataset: tuple[FastDataLoader, Optional[FastDataLoader], Optional[FastDataLoader]],
                 #  num_features: int,
                 #  num_classes: int,
                 builder_args: DDict = None,
                 **kwargs):
        self.data_container = None
        # self.standardize = False
        self.distribution = "iid"
        self.client_split = None
        self.sampling_perc = 1.0
        (self.client_tr_assignments,
         self.client_te_assignments,
         self.server_te) = dataset
        # self._num_features = num_features
        self._num_classes = self._compute_num_classes()

    def _compute_num_classes(self) -> int:

        labels = set()
        for ftdl in self.client_tr_assignments:
            y = ftdl.tensors[1]
            labels.update(set(list(y.numpy().flatten())))

        for ftdl in self.client_te_assignments:
            if ftdl:
                y = ftdl.tensors[1]
                labels.update(set(list(y.numpy().flatten())))

        if self.server_te:
            y = self.server_te.tensors[1]
            labels.update(set(list(y.numpy().flatten())))

        return len(labels)

    # def num_features(self) -> int:
    #     return self._num_features

    def num_classes(self) -> int:
        return self._num_classes

    def assign(self, n_clients: int, batch_size: Optional[int] = None):
        """This override of the :meth:`DataSplitter.assign` method returns the pre-assigned data.
        No further processing and computation is done.

        Important:
           The arguments of this method are not used.

        Args:
            n_clients (int): The number of clients.
            batch_size (Optional[int], optional): The batch size. Defaults to None.

        Returns:
            tuple[tuple[FastDataLoader, Optional[FastDataLoader]], FastDataLoader]: The clients'
            training and testing assignments and the server's testing assignment.
        """
        return (self.client_tr_assignments, self.client_te_assignments), self.server_te
