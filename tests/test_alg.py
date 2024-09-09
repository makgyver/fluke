import sys
import tempfile
from typing import Any

from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD

sys.path.append(".")
sys.path.append("..")

from fluke import DDict, GlobalSettings  # NOQA
from fluke.algorithms import CentralizedFL, PersonalizedFL  # NOQA
from fluke.client import Client, PFLClient  # NOQA
from fluke.comm import ChannelObserver, Message  # NOQA
from fluke.data import DataSplitter  # NOQA
from fluke.data.datasets import Datasets  # NOQA
from fluke.evaluation import ClassificationEval  # NOQA
from fluke.nets import MNIST_2NN  # NOQA
from fluke.server import Server  # NOQA
from fluke.utils import (ClientObserver, Configuration, ServerObserver,  # NOQA
                         get_class_from_qualified_name)
from fluke.utils.log import Log  # NOQA


def test_centralized_fl():
    hparams = DDict(
        # model="fluke.nets.MNIST_2NN",
        model=MNIST_2NN(),
        client=DDict(batch_size=32,
                     local_epochs=1,
                     loss=CrossEntropyLoss,
                     optimizer=DDict(
                         lr=0.1,
                         momentum=0.9),
                     scheduler=DDict(
                         step_size=1,
                         gamma=0.1)
                     ),
        server=DDict(weighted=True)
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist, client_split=0.1)
    fl = CentralizedFL(2, splitter, hparams)

    assert fl.n_clients == 2
    assert fl.hyper_params == hparams
    assert len(fl.clients) == 2
    assert isinstance(fl.clients[0], Client)
    assert isinstance(fl.server, Server)
    assert isinstance(fl.server.model, MNIST_2NN)
    assert fl.clients[0].test_set is not None
    assert fl.clients[1].test_set is not None
    assert isinstance(fl.clients[0].hyper_params.loss_fn, CrossEntropyLoss)

    hparams = DDict(
        model="fluke.nets.MNIST_2NN",
        client=DDict(batch_size=32,
                     local_epochs=1,
                     loss="CrossEntropyLoss",
                     optimizer=DDict(
                         lr=0.1,
                         momentum=0.9),
                     scheduler=DDict(
                         step_size=1,
                         gamma=0.1)
                     ),
        server=DDict(weighted=True)
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist, client_split=0.1)
    fl = CentralizedFL(2, splitter, hparams)

    GlobalSettings().set_evaluator(ClassificationEval(1, 10))
    GlobalSettings().set_eval_cfg(DDict(post_fit=True, pre_fit=True))

    assert isinstance(fl.server.model, MNIST_2NN)
    assert isinstance(fl.clients[0].hyper_params.loss_fn, CrossEntropyLoss)

    assert fl.get_optimizer_class() == SGD
    assert fl.get_client_class() == Client
    assert fl.get_server_class() == Server

    class Observer(ClientObserver, ServerObserver, ChannelObserver):
        def __init__(self):
            super().__init__()
            self.called_start = False
            self.called_end = False
            self.called_selected = False
            self.called_error = False
            self.called_finished = False
            self.called_client_eval = False
            self.called_server_eval = False
            self.called_start_fit = False
            self.called_end_fit = False

        def start_round(self, round, global_model):
            assert round == 1
            assert global_model is not None
            self.called_start = True

        def end_round(self, round):
            assert round == 1
            self.called_end = True

        def server_evaluation(self, round, type, evals) -> None:
            assert round == 1
            assert type == "global"
            assert "accuracy" in evals
            self.called_server_eval = True

        def client_evaluation(self, round, client_id, phase, evals, **kwargs: dict[str, Any]):
            assert round == 1 or (round == -1 and phase == "pre-fit")
            assert phase == "post-fit" or phase == "pre-fit"
            assert client_id == 0 or client_id == 1
            self.called_client_eval = True

        def selected_clients(self, round, clients):
            assert len(clients) == 1
            assert round == 1
            self.called_selected = True

        def finished(self, round):
            assert round == 2
            self.called_finished = True

        def start_fit(self, round: int, client_id: int, model: Module, **kwargs: dict[str, Any]):
            assert round == 1
            assert client_id == 0 or client_id == 1
            self.called_start_fit = True

        def end_fit(self, round: int, client_id: int, model: Module, loss: float, **kwargs: dict[str, Any]):
            assert round == 1
            assert client_id == 0 or client_id == 1
            assert loss >= 0.0
            self.called_end_fit = True

        def message_received(self, message: Message):
            pass

    obs = Observer()
    fl.set_callbacks(obs)

    assert fl.server._observers == [obs]

    strfl = "CentralizedFL(model=fluke.nets.MNIST_2NN,Client[0-1](optim=OptCfg(SGD,lr=0.1," + \
        "momentum=0.9,StepLR(step_size=1,gamma=0.1)),batch_size=32,loss_fn=CrossEntropyLoss()," + \
        "local_epochs=1),Server(weighted=True))"

    assert str(fl).replace(" ", "").replace(
        "\n", "").replace("\t", "") == strfl
    assert fl.__repr__().replace(" ", "").replace(
        "\n", "").replace("\t", "") == strfl

    fl.run(1, 0.5)
    assert obs.called_start
    assert obs.called_end
    assert obs.called_selected
    assert obs.called_finished
    assert obs.called_start_fit
    assert obs.called_end_fit
    assert obs.called_client_eval
    assert obs.called_server_eval

    with tempfile.TemporaryDirectory() as tmpdirname:
        fl.save(tmpdirname)
        fl2 = CentralizedFL(2, splitter, hparams)
        fl2.load(tmpdirname)

        assert fl2.server.rounds == fl.server.rounds

    hparams = DDict(
        # model="fluke.nets.MNIST_2NN",
        model=MNIST_2NN(),
        client=DDict(batch_size=32,
                     local_epochs=1,
                     model=MNIST_2NN(),
                     loss=CrossEntropyLoss,
                     optimizer=DDict(
                         lr=0.1,
                         momentum=0.9),
                     scheduler=DDict(
                         step_size=1,
                         gamma=0.1)
                     ),
        server=DDict(weighted=True)
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist)
    fl = PersonalizedFL(2, splitter, hparams)

    assert fl.n_clients == 2
    assert fl.hyper_params == hparams
    assert len(fl.clients) == 2
    assert isinstance(fl.clients[0], PFLClient)
    assert isinstance(fl.server, Server)
    assert isinstance(fl.server.model, MNIST_2NN)
    assert isinstance(fl.clients[0].personalized_model, MNIST_2NN)
    assert fl.clients[0].test_set is None
    assert isinstance(fl.clients[0].hyper_params.loss_fn, CrossEntropyLoss)

    fl.run(1, 0.5)
    with tempfile.TemporaryDirectory() as tmpdirname:
        fl.save(tmpdirname)
        fl2 = PersonalizedFL(2, splitter, hparams)
        fl2.load(tmpdirname)

        assert fl2.server.rounds == fl.server.rounds


def _test_algo(exp_config, alg_config, rounds=1, oncpu=True):
    cfg = Configuration(exp_config, alg_config)
    GlobalSettings().set_seed(cfg.exp.seed)
    if oncpu:
        cfg.exp.device = "cpu"
    else:
        # if torch.cuda.is_available():
        #     cfg.exp.device = "cuda"
        # elif torch.backends.mps.is_available():
        #     cfg.exp.device = "mps"
        # else:
        return None, None
    GlobalSettings().set_device(cfg.exp.device)
    dataset = Datasets.get(**cfg.data.dataset)
    splitter = DataSplitter(dataset=dataset,
                            distribution=cfg.data.distribution.name,
                            dist_args=cfg.data.distribution.exclude("name"),
                            **cfg.data.exclude('dataset', 'distribution'))

    fl_algo_class = get_class_from_qualified_name(cfg.method.name)
    algo = fl_algo_class(cfg.protocol.n_clients,
                         splitter,
                         cfg.method.hyperparameters)

    log = Log()
    log.init(**cfg)
    algo.set_callbacks(log)
    # algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
    algo.run(rounds, 1)
    return algo, log


def test_apfl():
    apfl, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/apfl.yaml")
    # apfl, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/apfl.yaml", oncpu=False)


def test_ccvr():
    ccvr, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/ccvr.yaml")
    # ccvr, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/ccvr.yaml", oncpu=False)


def test_ditto():
    ditto, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/ditto.yaml")
    # ditto, log = _test_algo("./tests/configs/exp.yaml",
    #                         "./tests/configs/alg/ditto.yaml", oncpu=False)


def test_fedala():
    fedala, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedala.yaml")
    # fedala, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedala.yaml", oncpu=False)


def test_fedamp():
    fedamp, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedamp.yaml")
    # fedamp, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedamp.yaml", oncpu=False)


def test_fedavg():
    fedavg, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedavg.yaml")
    # fedavg, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedavg.yaml", oncpu=False)


def test_fedavgm():
    fedavgm, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedavgm.yaml")
    # fedavgm, log = _test_algo("./tests/configs/exp.yaml",
    #                           "./tests/configs/alg/fedavgm.yaml", oncpu=False)


def test_fedaws():
    fedaws, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedaws.yaml")
    # fedaws, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedaws.yaml", oncpu=False)


def test_fedbabu():
    fedbabu, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedbabu.yaml")
    fedbabu, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedbabu_head.yaml")
    fedbabu, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedbabu_body.yaml")
    # fedbabu, log = _test_algo("./tests/configs/exp.yaml",
    #                           "./tests/configs/alg/fedbabu.yaml", oncpu=False)
    # fedbabu, log = _test_algo("./tests/configs/exp.yaml",
    #                           "./tests/configs/alg/fedbabu_head.yaml", oncpu=False)
    # fedbabu, log = _test_algo("./tests/configs/exp.yaml",
    #                           "./tests/configs/alg/fedbabu_body.yaml", oncpu=False)


def test_fedbn():
    fedbn, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedbn.yaml")
    # fedbn, log = _test_algo("./tests/configs/exp.yaml",
    #                         "./tests/configs/alg/fedbn.yaml", oncpu=False)


def test_feddyn():
    feddyn, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/feddyn.yaml")
    # feddyn, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/feddyn.yaml", oncpu=False)


def test_fedexp():
    fedexp, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedexp.yaml")
    # fedexp, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedexp.yaml", oncpu=False)


def test_fedhp():
    fedhp, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedhp.yaml")
    # fedhp, log = _test_algo("./tests/configs/exp.yaml",
    #                         "./tests/configs/alg/fedhp.yaml", oncpu=False)


def test_fedlc():
    fedlc, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedlc.yaml")
    # fedlc, log = _test_algo("./tests/configs/exp.yaml",
    #                         "./tests/configs/alg/fedlc.yaml", oncpu=False)


def test_fednh():
    fednh, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fednh.yaml")
    # fednh, log = _test_algo("./tests/configs/exp.yaml",
    #                         "./tests/configs/alg/fednh.yaml", oncpu=False)


def test_fednova():
    fednova, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fednova.yaml")
    # fednova, log = _test_algo("./tests/configs/exp.yaml",
    #                           "./tests/configs/alg/fednova.yaml", oncpu=False)


def test_fedopt():
    fedopt, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedadam.yaml")
    fedopt, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedadagrad.yaml")
    fedopt, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedyogi.yaml")
    # fedopt, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedadam.yaml", oncpu=False)
    # fedopt, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedadagrad.yaml", oncpu=False)
    # fedopt, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedyogi.yaml", oncpu=False)


def test_fedproto():
    fedproto, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedproto.yaml")
    # fedproto, log = _test_algo("./tests/configs/exp.yaml",
    #                            "./tests/configs/alg/fedproto.yaml", oncpu=False)


def test_fedprox():
    fedprox, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedprox.yaml")
    # fedprox, log = _test_algo("./tests/configs/exp.yaml",
    #                           "./tests/configs/alg/fedprox.yaml", oncpu=False)


def test_fedper():
    fedper, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedper.yaml")
    # fedper, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedper.yaml", oncpu=False)


def test_fedrep():
    fedrep, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedrep.yaml")
    # fedrep, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedrep.yaml", oncpu=False)


def test_fedrod():
    fedrod, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedrod.yaml")
    # fedrod, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedrod.yaml", oncpu=False)


def test_fedrs():
    fedrs, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedrs.yaml")
    # fedrs, log = _test_algo("./tests/configs/exp.yaml",
    #                         "./tests/configs/alg/fedrs.yaml", oncpu=False)


def test_fedsam():
    fedsam, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedsam.yaml")
    # fedsam, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedsam.yaml", oncpu=False)


def test_fedsgd():
    fedsgd, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedsgd.yaml")
    # fedsgd, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedsgd.yaml", oncpu=False)


def test_lgfedavg():
    lgfedavg, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/lg_fedavg.yaml")
    # lgfedavg, log = _test_algo("./tests/configs/exp.yaml",
    #                            "./tests/configs/alg/lg_fedavg.yaml", oncpu=False)


def test_moon():
    moon, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/moon.yaml")
    # moon, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/moon.yaml", oncpu=False)


def test_per_fedavg():
    per_fedavg, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/per_fedavg.yaml")
    # per_fedavg, log = _test_algo("./tests/configs/exp.yaml",
    #                              "./tests/configs/alg/per_fedavg.yaml", oncpu=False)
    per_fedavg, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/per_fedavg2.yaml")
    # per_fedavg, log = _test_algo("./tests/configs/exp.yaml",
    #                              "./tests/configs/alg/per_fedavg2.yaml", oncpu=False)


def test_pfedme():
    pfedme, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/pfedme.yaml")
    # pfedme, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/pfedme.yaml", oncpu=False)


def test_scaffold():
    scaffold, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/scaffold.yaml")
    # scaffold, log = _test_algo("./tests/configs/exp.yaml",
    #                            "./tests/configs/alg/scaffold.yaml", oncpu=False)


def test_superfed():
    superfed, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/superfed.yaml", 3)
    # superfed, log = _test_algo("./tests/configs/exp.yaml",
    #                            "./tests/configs/alg/superfed.yaml", 3, oncpu=False)


if __name__ == "__main__":
    # test_centralized_fl()
    # test_apfl()
    # test_ccvr()
    # test_ditto()
    # test_fedamp()
    # test_fedbabu()
    # test_feddyn()
    # test_fedlc()
    # test_fednova()
    # test_fedper()
    # test_fedrep()
    # test_lgfedavg()
    # test_moon()
    test_fedbn()
    # test_pfedme()  # TO BE CHECKED
    # test_scaffold()
    # test_superfed()
    # test_per_fedavg()
    # test_fedavg()
    # test_fedprox()
    # test_fedsgd()
    # test_fedexp()
    # test_fedproto()
    # test_fedopt()
    # test_fedavgm()
    # test_fedhp()
    # test_fednh()
