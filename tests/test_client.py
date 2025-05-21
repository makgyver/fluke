import sys

import pytest
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD

sys.path.append(".")
sys.path.append("..")

from fluke import DDict, FlukeENV  # NOQA
from fluke.client import Client, PFLClient  # NOQA
from fluke.config import OptimizerConfigurator  # NOQA
from fluke.data import FastDataLoader  # NOQA
from fluke.evaluation import ClassificationEval  # NOQA
from fluke.server import Server  # NOQA


def test_client_cache():
    _test_client(False)


def test_client():
    _test_client(True)


class Model(Linear):
    def __init__(self):
        super().__init__(10, 2)

        # initialize weights to 0
        self.weight.data.fill_(0)
        self.bias.data.fill_(0)


def _test_client(inmemory):

    FlukeENV().set_inmemory(inmemory)
    if not inmemory:
        FlukeENV().open_cache("test_client")
    FlukeENV().set_eval_cfg(pre_fit=True, post_fit=True)

    # function that taken a 10-dimensional input returns a 0 if the
    # sum of the first 7 elements is less than 2.5
    def target_function(x):
        return 0 if x[:7].sum() < 2.5 else 1

    Xtr = torch.rand((100, 10))
    ytr = torch.tensor([target_function(x) for x in Xtr])
    Xte = torch.rand((100, 10))
    yte = torch.tensor([target_function(x) for x in Xte])
    train_set = FastDataLoader(
        Xtr, ytr,
        num_labels=2,
        batch_size=10,
        shuffle=True
    )

    test_set = FastDataLoader(
        Xte, yte,
        num_labels=2,
        batch_size=10,
        shuffle=True
    )

    client = Client(
        index=0,
        train_set=train_set,
        test_set=test_set,
        clipping=10,
        optimizer_cfg=OptimizerConfigurator(
            optimizer_cfg=DDict(name=SGD, lr=0.1),
            scheduler_cfg=DDict(step_size=1, gamma=0.1),
        ),
        loss_fn=CrossEntropyLoss(),
        local_epochs=10
    )

    assert client.index == 0
    assert client.train_set == train_set
    assert client.test_set == test_set
    assert client.n_examples == 100

    server = Server(
        model=Model(),
        test_set=None,
        clients=[client]
    )

    client.set_channel(server.channel)
    assert client.channel == server.channel
    server.broadcast_model([client])

    evaluator = ClassificationEval(1, 2)
    FlukeENV().set_evaluator(evaluator)
    ev0 = client.evaluate(evaluator, client.test_set)
    client.local_update(1)
    ev1 = client.evaluate(evaluator, client.test_set)
    assert server.channel._buffer["server"]
    assert not ev0
    assert ev1
    # if inmemory:
    assert isinstance(client.optimizer, SGD)
    assert client.optimizer.defaults["lr"] == 0.1
    assert isinstance(client.scheduler, torch.optim.lr_scheduler.StepLR)
    assert client.scheduler.step_size == 1

    # assert str(client) == "Client[0](optim=OptCfg(SGD, lr=0.1, StepLR(step_size=1, gamma=0.1)), " + \
    #     "batch_size=10, loss_fn=CrossEntropyLoss(), local_epochs=10, fine_tuning_epochs=0, clipping=10)"

    client.test_set = None  # THIS IS NOT ALLOWED
    assert client.evaluate(evaluator, client.test_set) == {}

    server.broadcast_model([client])
    client.receive_model()
    # assert client.model.weight.data.sum() == 0
    # assert client.model.bias.data.sum() == 0

    client.send_model()

    m = server.channel.receive("server", client.index, "model")
    assert id(m) != id(client.model)
    assert m is not client.model
    assert torch.all(client.local_model.weight.data == client.model.weight.data)
    assert torch.all(client.local_model.bias.data == client.model.bias.data)

    server.broadcast_model([client])
    client.local_update(1)
    server.broadcast_model([client])
    client.finalize()
    assert client.model.weight.data.sum() == 0
    assert client.model.bias.data.sum() == 0

    # assert str(client) == "Client[0](optim=OptCfg(SGD, lr=0.1, StepLR(step_size=1, gamma=0.1)), " + \
    #     "batch_size=10, loss_fn=CrossEntropyLoss(), local_epochs=10, fine_tuning_epochs=0, clipping=10)"
    assert str(client) == repr(client)

    client.save("tmp/client")
    client2 = Client(
        index=0,
        train_set=train_set,
        test_set=test_set,
        clipping=10,
        optimizer_cfg=OptimizerConfigurator(
            optimizer_cfg=DDict(name=SGD, lr=0.1),
            scheduler_cfg=DDict(step_size=1, gamma=0.1),
        ),
        loss_fn=CrossEntropyLoss(),
        local_epochs=10
    )
    client2.load("tmp/client", Model())
    assert torch.all(client2.model.weight.data == client.model.weight.data)

    if not inmemory:
        FlukeENV().close_cache()


def test_pflclient():

    class Model(Linear):
        def __init__(self):
            super().__init__(10, 2)

    # function that taken a 10-dimensional input returns a 0 if the
    # sum of the first 7 elements is less than 2.5
    def target_function(x):
        return 0 if x[:7].sum() < 2.5 else 1

    Xtr = torch.rand((100, 10))
    ytr = torch.tensor([target_function(x) for x in Xtr])
    train_set = FastDataLoader(
        Xtr, ytr,
        num_labels=2,
        batch_size=10,
        shuffle=True
    )

    client = PFLClient(
        index=0,
        model=Model(),
        train_set=train_set,
        test_set=train_set,
        optimizer_cfg=OptimizerConfigurator(
            optimizer_cfg=DDict(name=SGD, lr=0.1),
            scheduler_cfg=DDict(step_size=1, gamma=0.1)
        ),
        loss_fn=CrossEntropyLoss(),
        local_epochs=10
    )

    assert client.model is None
    assert client.personalized_model is not None, isinstance(client.personalized_model, Model)

    evaluator = ClassificationEval(1, 2)
    try:
        client.evaluate(evaluator, client.test_set)
    except Exception:
        pytest.fail("Unexpected exception!")
    client.test_set = None  # THIS IS NOT ALLOWED
    assert client.evaluate(evaluator, client.test_set) == {}

    server = Server(
        model=Model(),
        test_set=None,
        clients=[client]
    )

    client.set_channel(server.channel)
    client.pers_optimizer = SGD(client.personalized_model.parameters(), lr=0.1)
    client.pers_scheduler = torch.optim.lr_scheduler.StepLR(
        client.pers_optimizer, step_size=1, gamma=0.1)
    server.broadcast_model([client])
    client.local_update(1)
    client.save("tmp/pflclient")
    client.fit(1)

    client2 = PFLClient(
        index=0,
        model=Model(),
        train_set=train_set,
        test_set=train_set,
        optimizer_cfg=OptimizerConfigurator(
            optimizer_cfg=DDict(name=SGD, lr=0.1),
            scheduler_cfg=DDict(step_size=1, gamma=0.1)
        ),
        loss_fn=CrossEntropyLoss(),
        local_epochs=10
    )
    client2.load("tmp/pflclient", Model())
    assert torch.all(client2.personalized_model.weight.data ==
                     client.personalized_model.weight.data)
    assert torch.all(client2.model.weight.data ==
                     client.model.weight.data)


if __name__ == "__main__":
    test_client()
    # 100% coverage for client.py
