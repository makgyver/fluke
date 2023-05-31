from __future__ import annotations
from enum import Enum

from pyparsing import Optional
from sklearn.model_selection import train_test_split
import torch

from typing import List, Union
import numpy as np
from numpy.random import randint, shuffle, power, choice, dirichlet, permutation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import mquantiles


class DataContainer:
    """Container for train and test (classification) data.

    Parameters
    ----------
    X_train : torch.Tensor or np.array
        Training data.
    y_train : torch.Tensor or np.array
        Training labels.
    X_test : torch.Tensor or np.array
        Test data.
    y_test : torch.Tensor or np.array
        Test labels.
    """ 
    def __init__(self, 
                 X_train: Union[torch.Tensor, np.array], 
                 y_train: Union[torch.Tensor, np.array], 
                 X_test: Union[torch.Tensor, np.array], 
                 y_test: Union[torch.Tensor, np.array],
                 num_classes: int):
        self.train = (torch.FloatTensor(X_train), torch.LongTensor(y_train))
        self.test = (torch.FloatTensor(X_test), torch.LongTensor(y_test))
        self.num_features = np.prod([i for i in X_train.shape[1:]]).item()
        self.num_classes = num_classes
    
    def standardize(self):
        data_train, data_test = self.train[0], self.test[0]
        if isinstance(data_train, torch.Tensor):
            data_train = data_train.numpy()
        scaler = StandardScaler()
        scaler.fit(data_train)
        self.train = (torch.FloatTensor(scaler.transform(data_train)), self.train[1])
        self.test = (torch.FloatTensor(scaler.transform(data_test)), self.test[1])

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

    from fl_bench.data.datasets import DatasetsEnum
    
    @classmethod
    def from_config(cls, config: dict) -> DataSplitter:
        return DataSplitter(config['dataset'],
                            config['standardize'],
                            config['distribution'],
                            config['validation_split'],
                            config['sampling_perc'])


    def __init__(self, 
                 dataset: DatasetsEnum,
                 standardize: bool=False,
                 distribution: DistributionEnum=DistributionEnum.IID,
                 validation_split: float=0.0,
                 sampling_perc: float=1.0,
                 **kwargs):
        assert 0 <= validation_split <= 1, "validation_split must be between 0 and 1."
        assert 0 <= sampling_perc <= 1, "sampling_perc must be between 0 and 1."

        self.data_container = dataset.klass()() #FIXME for handling different test set size
        self.standardize = standardize
        if standardize:
            self.data_container.standardize()

        self.distribution = distribution
        self.validation_split = validation_split
        self.sampling_perc = sampling_perc
        self.kwargs = kwargs
    
    def num_features(self) -> int:
        return self.data_container.num_features

    def num_classes(self) -> int:
        return self.data_container.num_classes

    def assign(self, n_clients: int, batch_size: Optional[int]=None, ):
        Xtr, Ytr = self.data_container.train
        self.assignments = self._iidness_functions[self.distribution](self, Xtr, Ytr, n_clients, **self.kwargs)
        client_tr_assignments = []
        client_te_assignments = []
        for c in range(n_clients):
            client_X = Xtr[self.assignments[c]]
            client_y = Ytr[self.assignments[c]]
            if self.validation_split > 0.0:
                Xtr_client, Xte_client, Ytr_client, Yte_client = train_test_split(client_X, 
                                                                                  client_y,
                                                                                  test_size=self.validation_split, 
                                                                                  stratify=client_y)
                client_tr_assignments.append(FastTensorDataLoader(Xtr_client, 
                                                                     Ytr_client, 
                                                                     batch_size=batch_size, 
                                                                     shuffle=True, 
                                                                     percentage=self.sampling_perc))
                client_te_assignments.append(FastTensorDataLoader(Xte_client, 
                                                                    Yte_client, 
                                                                    batch_size=batch_size, 
                                                                    shuffle=True, 
                                                                    percentage=self.sampling_perc))
            else:
                client_tr_assignments.append(FastTensorDataLoader(client_X, 
                                                                     client_y, 
                                                                     batch_size=batch_size, 
                                                                     shuffle=True, 
                                                                     percentage=self.sampling_perc))
                client_te_assignments.append(None)

        server_te = FastTensorDataLoader(*self.data_container.test,
                                         batch_size=100,
                                         shuffle=False)
        
        return (client_tr_assignments, client_te_assignments), server_te

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
        labels = list(range(len(torch.unique(torch.LongTensor(y)).numpy())))
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
        labels = set(torch.unique(torch.LongTensor(y)).numpy())
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
        labels = set(torch.unique(torch.LongTensor(y)).numpy())
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
        for lbl in set(torch.unique(torch.LongTensor(y)).numpy()):
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
