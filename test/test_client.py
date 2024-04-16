import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
import pytest
import sys
sys.path.append(".")
sys.path.append("..")

from fl_bench.client import Client, PFLClient  # NOQA
from fl_bench.server import Server  # NOQA
from fl_bench.utils import OptimizerConfigurator  # NOQA
from fl_bench.data import FastTensorDataLoader  # NOQA
from fl_bench.comm import Message  # NOQA


def test_client():

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
    Xte = torch.rand((100, 10))
    yte = torch.tensor([target_function(x) for x in Xte])
    train_set = FastTensorDataLoader(
        Xtr, ytr,
        num_labels=2,
        batch_size=10,
        shuffle=True
    )

    test_set = FastTensorDataLoader(
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
            optimizer_class=SGD,
            scheduler_kwargs={"step_size": 1, "gamma": 0.1},
            lr=0.1
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
        test_data=None,
        clients=[client]
    )

    client.set_server(server)
    assert client.server == server
    assert client.channel == server.channel
    server._broadcast_model([client])

    ev0 = client.evaluate()
    client.fit()
    ev1 = client.evaluate()
    assert server.channel._buffer[server]
    assert ev1["loss"] <= ev0["loss"]

    assert str(client) == "Client[0](optim=OptCfg(SGD,lr=0.1,StepLR(step_size=1,gamma=0.1))," + \
        "batch_size=10,loss_fn=CrossEntropyLoss(),local_epochs=10)"

    client.test_set = None  # THIS IS NOT ALLOWED
    assert client.evaluate() == {}


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
    train_set = FastTensorDataLoader(
        Xtr, ytr,
        num_labels=2,
        batch_size=10,
        shuffle=True
    )

    client = PFLClient(
        index=0,
        model=Model(),
        train_set=train_set,
        test_set=None,
        optimizer_cfg=OptimizerConfigurator(
            optimizer_class=SGD,
            scheduler_kwargs={"step_size": 1, "gamma": 0.1},
            lr=0.1
        ),
        loss_fn=CrossEntropyLoss(),
        local_epochs=10
    )

    assert client.model is None
    assert client.personalized_model is not None, isinstance(client.personalized_model, Model)

    try:
        client.evaluate()
    except Exception:
        pytest.fail("Unexpected exception!")
    client.test_set = None  # THIS IS NOT ALLOWED
    assert client.evaluate() == {}


if __name__ == "__main__":
    test_client()
