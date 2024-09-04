import sys

import pytest
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD

sys.path.append(".")
sys.path.append("..")

from fluke import DDict, GlobalSettings  # NOQA
from fluke.client import Client, PFLClient  # NOQA
from fluke.data import FastDataLoader  # NOQA
from fluke.evaluation import ClassificationEval  # NOQA
from fluke.server import Server  # NOQA
from fluke.utils import OptimizerConfigurator  # NOQA


def test_client():

    class Model(Linear):
        def __init__(self):
            super().__init__(10, 2)
            self.output_size = 2

            # initialize weights to 0
            self.weight.data.fill_(0)
            self.bias.data.fill_(0)

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

    client.set_server(server)
    assert client.server == server
    assert client.channel == server.channel
    server.broadcast_model([client])

    evaluator = ClassificationEval(1, 2)
    GlobalSettings().set_evaluator(evaluator)
    ev0 = client.evaluate(evaluator, client.test_set)
    client.local_update(1)
    ev1 = client.evaluate(evaluator, client.test_set)
    assert server.channel._buffer[server]
    assert not ev0
    assert ev1

    assert str(client) == "Client[0](optim=OptCfg(SGD, lr=0.1, StepLR(step_size=1, gamma=0.1)), " + \
        "batch_size=10, loss_fn=CrossEntropyLoss(), local_epochs=10)"

    client.test_set = None  # THIS IS NOT ALLOWED
    assert client.evaluate(evaluator, client.test_set) == {}

    server.broadcast_model([client])
    client.receive_model()
    assert client.model.weight.data.sum() == 0
    assert client.model.bias.data.sum() == 0

    client.send_model()

    m = server.channel.receive(server, client, "model")
    assert id(m) != id(client.model)
    assert m is not client.model

    server.broadcast_model([client])
    client.local_update(1)
    server.broadcast_model([client])
    client.finalize()
    assert client.model.weight.data.sum() == 0
    assert client.model.bias.data.sum() == 0

    assert str(client) == "Client[0](optim=OptCfg(SGD, lr=0.1, StepLR(step_size=1, gamma=0.1)), " + \
        "batch_size=10, loss_fn=CrossEntropyLoss(), local_epochs=10)"
    assert str(client) == repr(client)


def test_pflclient():

    class Model(Linear):
        def __init__(self):
            super().__init__(10, 2)
            self.output_size = 2

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


if __name__ == "__main__":
    test_client()
    # 100% coverage for client.py
