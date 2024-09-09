from __future__ import annotations

import sys

import pytest
import torch
from torch.nn import Linear

sys.path.append(".")
sys.path.append("..")

from fluke import DDict  # NOQA
from fluke import GlobalSettings  # NOQA
from fluke.client import Client  # NOQA
from fluke.data import FastDataLoader  # NOQA
from fluke.evaluation import ClassificationEval  # NOQA
from fluke.server import Server  # NOQA
from fluke.utils import OptimizerConfigurator, ServerObserver  # NOQA


def test_server():

    class Observer(ServerObserver):

        def start_round(self, round, global_model):
            assert round == 1
            assert global_model is not None

        def end_round(self,
                      round):
            assert round == 1

        def selected_clients(self, round, clients):
            assert len(clients) == 2
            assert round == 1

        def server_evaluation(self, round, type, evals, **kwargs) -> None:
            assert round == 1
            assert type == "global"
            assert "accuracy" in evals

        def finished(self, round):
            assert round == 1 or round == 2

    class Model(Linear):
        def __init__(self):
            super().__init__(10, 2)
            self.output_size = 2

    def target_function(x):
        return 0 if x[:7].sum() < 2.5 else 1

    Xtr = [torch.rand((100, 10)), torch.rand((100, 10))]
    ytr = [torch.tensor([target_function(x) for x in Xtr[0]]),
           torch.tensor([target_function(x) for x in Xtr[1]])]
    Xte = torch.rand((100, 10))
    yte = torch.tensor([target_function(x) for x in Xte])

    ftdl_client = [FastDataLoader(Xtr[i], ytr[i], num_labels=2, batch_size=10, shuffle=True)
                   for i in range(2)]
    ftdl_server = FastDataLoader(Xte, yte, num_labels=2, batch_size=10, shuffle=False)

    cfg = OptimizerConfigurator(optimizer_cfg=DDict(name=torch.optim.SGD, lr=0.1, momentum=0.9))
    clients = [Client(index=i,
                      train_set=ftdl_client[i],
                      test_set=ftdl_client[i] if i == 0 else None,
                      optimizer_cfg=cfg,
                      loss_fn=torch.nn.CrossEntropyLoss(),
                      local_epochs=3)
               for i in range(2)]

    server = Server(clients=clients,
                    model=Model(),
                    test_set=ftdl_server,
                    weighted=True)

    assert server.clients == clients
    assert isinstance(server.model, Model)
    assert server.test_set == ftdl_server
    assert server.hyper_params.weighted
    assert server.channel == clients[0].channel
    assert server.rounds == 0
    assert server.has_test
    assert server.has_model

    evaluator = ClassificationEval(1, 2)
    GlobalSettings().set_evaluator(evaluator)
    obs = Observer()
    server.attach(obs)
    ev0 = server.evaluate(evaluator, server.test_set)
    server.fit(1, 1)
    ev1 = server.evaluate(evaluator, server.test_set)

    # assert ev0["loss"] >= ev1["loss"]
    assert server.rounds == 1
    assert str(server) == "Server(weighted=True)"

    server.detach(obs)
    server.hyper_params.weighted = False
    try:
        server.fit(2, 0.5)
    except Exception:
        pytest.fail("Unexpected exception!")

    for c in clients:
        c.send_model()
    cmodels = server.get_client_models(clients, state_dict=False)
    assert len(cmodels) == 2
    assert isinstance(cmodels[0], Model)

    server.test_set = None
    assert not server.evaluate(evaluator, server.test_set)

    server.broadcast_model(clients)

    for c in clients:
        m = c.channel.receive(c, server, "model")
        assert id(m) != id(server.model)
        assert m is not server.model

    assert str(server) == "Server(weighted=False)"
    assert str(server) == repr(server)


if __name__ == "__main__":
    test_server()
    # 98% coverage for server.py
