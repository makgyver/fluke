from __future__ import annotations

import pytest
import torch
import sys
sys.path.append(".")
sys.path.append("..")

from fl_bench.data import DataContainer, FastTensorDataLoader  # NOQA


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

    with pytest.raises(AssertionError):
        loader = FastTensorDataLoader(torch.rand(100, 20),
                                      torch.randint(0, 10, (101,)),
                                      num_labels=10,
                                      batch_size=10,
                                      shuffle=True,
                                      percentage=1,
                                      skip_singleton=True)


if __name__ == "__main__":
    test_container()
    test_ftdl()
