from __future__ import annotations

import pytest
import torch
import sys
sys.path.append(".")
sys.path.append("..")

from fluke.data import DataContainer, FastTensorDataLoader, DataSplitter, DistributionEnum  # NOQA
from fluke.data.datasets import DatasetsEnum  # NOQA
from fluke import DDict  # NOQA


def test_container():
    data = DataContainer(
        X_train=torch.rand(100, 20),
        y_train=torch.randint(0, 10, (100,)),
        X_test=torch.rand(100, 20),
        y_test=torch.randint(0, 10, (100,)),
        num_classes=10
    )

    assert data.train[0].shape == torch.Size([100, 20])
    assert data.train[1].shape == torch.Size([100])
    assert data.test[0].shape == torch.Size([100, 20])
    assert data.test[1].shape == torch.Size([100])
    assert data.num_classes == 10
    assert data.num_features == 20

    data.standardize()

    assert torch.allclose(data.train[0].mean(dim=0), torch.zeros(20), atol=1e-7)
    assert torch.allclose(data.train[0].std(dim=0), torch.ones(20), atol=1e-2)


def test_ftdl():
    loader = FastTensorDataLoader(torch.rand(100, 20),
                                  torch.randint(0, 10, (100,)),
                                  num_labels=10,
                                  batch_size=10,
                                  shuffle=True,
                                  percentage=1,
                                  skip_singleton=True)

    assert len(loader) == 10

    for X, y in loader:
        assert X.shape == torch.Size([10, 20])
        assert y.shape == torch.Size([10])
        break

    loader.batch_size = 99

    cnt = 0
    for X, y in loader:
        cnt += 1
    assert cnt == 1

    loader.skip_singleton = False

    cnt = 0
    for X, y in loader:
        cnt += 1
    assert cnt == 2

    assert loader.size == 100
    loader.set_sample_size(0.2)
    assert loader.size == 20

    with pytest.raises(ValueError):
        loader.batch_size = -10

    with pytest.raises(ValueError):
        loader.set_sample_size(-10)

    with pytest.raises(ValueError):
        loader.set_sample_size(1.1)

    loader = FastTensorDataLoader(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
                                  torch.LongTensor([0, 1]),
                                  num_labels=10,
                                  batch_size=1,
                                  shuffle=False,
                                  percentage=1,
                                  skip_singleton=True)

    for X, y in loader:
        assert X == torch.FloatTensor([[1, 2, 3]])
        assert y == torch.tensor([0])

    loader.shuffle = True

    for X, y in loader:
        assert X == torch.FloatTensor([[4, 5, 6]]) or X == torch.FloatTensor([[1, 2, 3]])
        assert y == torch.tensor([1]) or y == torch.tensor([0])

    loader = FastTensorDataLoader(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
                                  torch.LongTensor([0, 1]),
                                  num_labels=10,
                                  batch_size=1,
                                  shuffle=False,
                                  percentage=1,
                                  skip_singleton=False,
                                  single_batch=True)

    cnt = 0
    for X, y in loader:
        cnt += 1
    assert cnt == 1

    with pytest.raises(AssertionError):
        loader = FastTensorDataLoader(torch.rand(100, 20),
                                      torch.randint(0, 10, (101,)),
                                      num_labels=10,
                                      batch_size=10,
                                      shuffle=True,
                                      percentage=1,
                                      skip_singleton=True)

    loader = FastTensorDataLoader(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
                                  torch.LongTensor([0, 1]),
                                  num_labels=10,
                                  batch_size=0,
                                  shuffle=False,
                                  percentage=1,
                                  skip_singleton=False,
                                  single_batch=False)

    assert loader.batch_size == 2


def test_splitter():
    cfg = DDict(
        client_split=0.1,
        dataset={
            "name": DatasetsEnum.MNIST,
        },
        distribution={
            "name": DistributionEnum.IID,
        },
        sampling_perc=0.1,
        server_test=True,
        server_split=0.2,
        keep_test=False
    )

    splitter = DataSplitter.from_config(cfg)
    assert splitter.client_split == 0.1
    assert splitter.sampling_perc == 0.1
    assert splitter.server_split == 0.2
    assert not splitter.keep_test
    assert splitter.distribution == DistributionEnum.IID

    (ctr, cte), ste = splitter.assign(10, batch_size=10)

    assert len(ctr) == 10
    assert len(cte) == 10
    assert isinstance(ctr[0], FastTensorDataLoader)
    x, y = next(iter(ctr[0]))
    assert x.shape == torch.Size([10, 28, 28])
    assert y.shape == torch.Size([10])
    x, y = next(iter(cte[0]))
    assert x.shape == torch.Size([10, 28, 28])
    assert y.shape == torch.Size([10])
    assert isinstance(ste, FastTensorDataLoader)
    x, y = next(iter(ste))
    assert x.shape == torch.Size([128, 28, 28])
    assert y.shape == torch.Size([128])

    splitter.distribution = DistributionEnum.LABEL_DIRICHLET_SKEWED
    (ctr, cte), ste = splitter.assign(10, batch_size=10)

    splitter.distribution = DistributionEnum.COVARIATE_SHIFT
    (ctr, cte), ste = splitter.assign(10, batch_size=10)

    splitter.distribution = DistributionEnum.CLASSWISE_QUANTITY_SKEWED
    (ctr, cte), ste = splitter.assign(10, batch_size=10)

    splitter.distribution = DistributionEnum.LABEL_PATHOLOGICAL_SKEWED
    (ctr, cte), ste = splitter.assign(10, batch_size=10)

    splitter.distribution = DistributionEnum.LABEL_QUANTITY_SKEWED
    (ctr, cte), ste = splitter.assign(10, batch_size=10)

    splitter.distribution = DistributionEnum.QUANTITY_SKEWED
    (ctr, cte), ste = splitter.assign(10, batch_size=10)


if __name__ == "__main__":
    test_container()
    test_ftdl()
    test_splitter()
