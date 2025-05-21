from __future__ import annotations

import sys

import pytest
import torch
from torchvision.transforms import v2

sys.path.append(".")
sys.path.append("..")

from fluke import DDict  # NOQA
from fluke.data import (DataContainer, DataSplitter,  # NOQA
                        DummyDataContainer, FastDataLoader)
from fluke.data.datasets import Datasets  # NOQA


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

    # data.standardize()

    # assert torch.allclose(data.train[0].mean(dim=0), torch.zeros(20), atol=1e-7)
    # assert torch.allclose(data.train[0].std(dim=0), torch.ones(20), atol=1e-2)


def test_ftdl():
    X, y = torch.rand(100, 20), torch.randint(0, 10, (100,))
    loader = FastDataLoader(X,
                            y,
                            num_labels=10,
                            batch_size=10,
                            shuffle=True,
                            percentage=1,
                            skip_singleton=True)

    assert len(loader) == 10
    assert torch.allclose(loader[0][0], X[0])
    assert torch.allclose(loader[0][1], y[0])

    with pytest.raises(IndexError):
        loader[300]

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

    loader = FastDataLoader(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
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

    loader = FastDataLoader(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
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
        loader = FastDataLoader(torch.rand(100, 20),
                                torch.randint(0, 10, (101,)),
                                num_labels=10,
                                batch_size=10,
                                shuffle=True,
                                percentage=1,
                                skip_singleton=True)

    loader = FastDataLoader(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]),
                            torch.LongTensor([0, 1]),
                            num_labels=10,
                            batch_size=0,
                            shuffle=False,
                            percentage=1,
                            skip_singleton=False,
                            single_batch=False)

    assert loader.batch_size == 2

    imgs = torch.randint(0, 256, size=(5, 3, 32, 32), dtype=torch.uint8)
    lbls = torch.randint(0, 10, size=(5,), dtype=torch.long)

    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=False),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loader = FastDataLoader(imgs,
                            lbls,
                            num_labels=10,
                            batch_size=1,
                            shuffle=False,
                            transforms=transforms,
                            percentage=1,
                            skip_singleton=False,
                            single_batch=False)

    for X, y in loader:
        assert X.shape == torch.Size([1, 3, 224, 224])
        assert y.shape == torch.Size([1])
        break

    x0_tr = loader[0]
    assert x0_tr[0].shape == torch.Size([3, 224, 224])
    assert torch.any(x0_tr[0] != X[0])

    dataloader = loader.as_dataloader()
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == 1
    assert len(dataloader.dataset) == 5



def test_splitter():
    cfg = DDict(
        client_split=0.1,
        dataset={
            "name": "mnist",
        },
        distribution={
            "name": "iid",
        },
        sampling_perc=0.1,
        server_test=True,
        server_split=0.2,
        keep_test=False
    )

    data_container = Datasets.get(**cfg.dataset)
    splitter = DataSplitter(dataset=data_container,
                            distribution=cfg.distribution.name,
                            dist_args=cfg.distribution.exclude("name"),
                            **cfg.exclude('dataset', 'distribution'))
    assert splitter.client_split == 0.1
    assert splitter.sampling_perc == 0.1
    assert splitter.server_split == 0.2
    assert not splitter.keep_test
    assert splitter.distribution == "iid"

    # OK uniform
    (ctr, cte), ste = splitter.assign(11, batch_size=10)

    assert len(ctr) == 11
    assert len(cte) == 11
    assert isinstance(ctr[0], FastDataLoader)
    x, y = next(iter(ctr[0]))
    assert x.shape == torch.Size([10, 28, 28])
    assert y.shape == torch.Size([10])
    x, y = next(iter(cte[0]))
    assert x.shape == torch.Size([10, 28, 28])
    assert y.shape == torch.Size([10])
    assert isinstance(ste, FastDataLoader)
    x, y = next(iter(ste))
    assert x.shape == torch.Size([128, 28, 28])
    assert y.shape == torch.Size([128])

    n_clients = 100

    # OK
    splitter.distribution = "dir"
    splitter.dist_args.balanced = True
    (ctr, cte), ste = splitter.assign(n_clients, batch_size=10)

    # OK?
    # splitter.distribution = "covariate"
    # (ctr, cte), ste = splitter.assign(n_clients, batch_size=10)

    # OK
    # splitter.distribution = "classwise_qnt"
    # (ctr, cte), ste = splitter.assign(n_clients, batch_size=10)

    del splitter.dist_args["balanced"]
    splitter.distribution = "pathological"
    (ctr, cte), ste = splitter.assign(n_clients, batch_size=10)

    splitter.distribution = "lbl_qnt"
    (ctr, cte), ste = splitter.assign(n_clients, batch_size=10)

    splitter.distribution = "qnt"
    (ctr, cte), ste = splitter.assign(n_clients, batch_size=10)

    splitter.distribution = "dir"
    splitter.dist_args.balanced = False
    (ctr, cte), ste = splitter.assign(n_clients, batch_size=10)

    cfg = DDict(
        dataset={
            "name": "mnist",
        },
        distribution={
            "name": "iid",
        },
        sampling_perc=0.1,
        keep_test=True,
        server_test=True,
        server_split=0.2,
        client_split=0.1,
        uniform_test=True
    )
    data_container = Datasets.get(**cfg.dataset)
    splitter = DataSplitter(dataset=data_container,
                            distribution=cfg.distribution.name,
                            dist_args=cfg.distribution.exclude("name"),
                            **cfg.exclude('dataset', 'distribution'))
    (ctr, cte), ste = splitter.assign(n_clients=10, batch_size=10)

    n_examples = splitter.data_container.train[0].shape[0]
    assert ctr[0].size == int(n_examples / 10 * 0.1 * 0.9)
    assert cte[0].size == int(n_examples / 10 * 0.1 * 0.1)
    assert ste.size == 1000

    # cfg.client_split = 0.2
    # splitter = DataSplitter(dataset=data_container,
    #                         distribution="dir",
    #                         dist_args={"beta": 0.3},
    #                         **cfg.exclude('dataset', 'distribution'))

    # (ctr, cte), ste = splitter.assign(n_clients=500, batch_size=10)

    cfg.client_split = 0
    splitter = DataSplitter(dataset=data_container,
                            distribution=cfg.distribution.name,
                            dist_args=cfg.distribution.exclude("name"),
                            **cfg.exclude('dataset', 'distribution'))
    (ctr, cte), ste = splitter.assign(n_clients=10, batch_size=10)

    assert ctr[0].size == int(n_examples / 10 * 0.1)
    assert cte[0] is None
    assert ste.size == 1000

    cfg.server_test = False
    cfg.client_split = 0.1
    splitter = DataSplitter(dataset=data_container,
                            distribution=cfg.distribution.name,
                            dist_args=cfg.distribution.exclude("name"),
                            **cfg.exclude('dataset', 'distribution'))
    (ctr, cte), ste = splitter.assign(n_clients=10, batch_size=10)

    n_examples_te = splitter.data_container.test[0].shape[0]
    assert ctr[0].size == int(n_examples / 10 * 0.1)
    assert cte[0].size == int(n_examples_te / 10 * 0.1)
    assert ste is None

    cfg.keep_test = False
    splitter = DataSplitter(dataset=data_container,
                            distribution=cfg.distribution.name,
                            dist_args=cfg.distribution.exclude("name"),
                            **cfg.exclude('dataset', 'distribution'))
    (ctr, cte), ste = splitter.assign(n_clients=10, batch_size=10)

    assert ctr[0].size == int((n_examples + n_examples_te) / 10 * 0.1 * 0.9)
    assert cte[0].size == int((n_examples + n_examples_te) / 10 * 0.1 * 0.1)
    assert ste is None

    cfg.client_split = 0
    cfg.server_test = True
    cfg.server_split = 0.2
    splitter = DataSplitter(dataset=data_container,
                            distribution=cfg.distribution.name,
                            dist_args=cfg.distribution.exclude("name"),
                            **cfg.exclude('dataset', 'distribution'))
    (ctr, cte), ste = splitter.assign(n_clients=10, batch_size=10)

    assert ctr[0].size == int((n_examples + n_examples_te) * 0.8 / 10 * 0.1)
    assert cte[0] is None
    assert ste.size == int((n_examples + n_examples_te) * 0.1 * 0.2)

    cfg.client_split = 0
    cfg.keep_test = False
    cfg.server_test = False
    with pytest.raises(AssertionError):
        DataSplitter(data_container, **cfg.exclude("dataset"))

    cfg.server_test = True
    cfg.server_split = 0
    with pytest.raises(AssertionError):
        DataSplitter(data_container, **cfg.exclude("dataset"))

    dummy = DummyDataContainer(ctr, cte, ste, 10)
    assert dummy.num_classes == 10
    assert dummy.clients_tr == ctr
    assert dummy.clients_te == cte
    assert dummy.server_data == ste

    splitter = DataSplitter(dataset=dummy)
    (ctr_, cte_), ste_ = splitter.assign(10, batch_size=10)
    assert torch.all(ctr_[0][0][0] == ctr[0][0][0])
    assert cte_[0] is None
    assert torch.all(ste_[0][0][0] == ste[0][0][0])


if __name__ == "__main__":
    test_container()
    test_ftdl()
    test_splitter()

    # 95% coverage on fluke/data
