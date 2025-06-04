import gc
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD

sys.path.append(".")
sys.path.append("..")

from fluke import DDict, FlukeENV  # NOQA
from fluke.algorithms import CentralizedFL, PersonalizedFL  # NOQA
from fluke.client import Client  # NOQA
from fluke.comm import ChannelObserver, Message  # NOQA
from fluke.config import Configuration  # NOQA
from fluke.data import DataSplitter  # NOQA
from fluke.data.datasets import Datasets  # NOQA
from fluke.evaluation import ClassificationEval  # NOQA
from fluke.nets import MNIST_2NN  # NOQA
from fluke.server import Server  # NOQA
from fluke.utils import ClientObserver, ServerObserver, get_class_from_qualified_name  # NOQA
from fluke.utils.log import Log  # NOQA

FlukeENV().set_evaluator(ClassificationEval(1, 10))
FlukeENV().set_eval_cfg(post_fit=True, pre_fit=True)
FlukeENV().set_save_options(path="tests/tmp/tmp", save_every=1, global_only=True)


def test_centralized_fl():
    hparams = DDict(
        # model="fluke.nets.MNIST_2NN",
        model=MNIST_2NN(),
        client=DDict(
            batch_size=32,
            local_epochs=1,
            loss="CrossEntropyLoss",
            optimizer=DDict(name="SGD", lr=0.1, momentum=0.9),
            scheduler=DDict(step_size=1, gamma=0.1),
        ),
        server=DDict(weighted=True),
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist, client_split=0.1)
    fl = CentralizedFL(2, splitter, hparams)

    assert fl.n_clients == 2
    assert fl.hyper_params.match(hparams)
    assert len(fl.clients) == 2
    assert isinstance(fl.clients[0], Client)
    assert isinstance(fl.server, Server)
    assert isinstance(fl.server.model, MNIST_2NN)
    assert fl.clients[0].test_set is not None
    assert fl.clients[1].test_set is not None
    assert isinstance(fl.clients[0].hyper_params.loss_fn, CrossEntropyLoss)

    hparams = DDict(
        model="fluke.nets.MNIST_2NN",
        client=DDict(
            batch_size=32,
            local_epochs=1,
            loss="CrossEntropyLoss",
            optimizer=DDict(lr=0.1, momentum=0.9),
            scheduler=DDict(step_size=1, gamma=0.1),
        ),
        server=DDict(weighted=True),
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist, client_split=0.1)
    fl = CentralizedFL(2, splitter, hparams)

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

        def server_evaluation(self, round, eval_type, evals) -> None:
            assert round == 1
            assert eval_type == "global"
            assert "accuracy" in evals
            self.called_server_eval = True

        def client_evaluation(self, round, client_id, phase, evals, **kwargs):
            # assert round == 1 or (round == -1 and phase == "pre-fit")
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

        def start_fit(self, round: int, client_id: int, model: Module, **kwargs):
            assert round == 1
            assert client_id == 0 or client_id == 1
            self.called_start_fit = True

        def end_fit(self, round: int, client_id: int, model: Module, loss: float, **kwargs):
            assert round == 1
            assert client_id == 0 or client_id == 1
            assert loss >= 0.0
            self.called_end_fit = True

        def message_received(self, by: Any, message: Message):
            pass

    obs = Observer()
    fl.set_callbacks(obs)

    assert fl.server._observers == [fl, obs]

    strfl = (
        f"CentralizedFL[{fl.id}](model=fluke.nets.MNIST_2NN(),Client[0-1](optim=OptCfg(SGD,lr=0.1,"
        + "momentum=0.9,StepLR(step_size=1,gamma=0.1)),batch_size=32,loss_fn=CrossEntropyLoss(),"
        + "local_epochs=1,fine_tuning_epochs=0,clipping=0),Server(weighted=True,lr=1.0))"
    )

    assert str(fl).replace(" ", "").replace("\n", "").replace("\t", "") == strfl
    assert fl.__repr__().replace(" ", "").replace("\n", "").replace("\t", "") == strfl

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
        temppath = fl.save(tmpdirname)
        fl2 = CentralizedFL(2, splitter, hparams)
        fl2.load(temppath)

        assert fl2.server.rounds == fl.server.rounds

    shutil.rmtree(f"tests/tmp/tmp_{fl.id}")

    FlukeENV().set_inmemory(True)
    hparams = DDict(
        # model="fluke.nets.MNIST_2NN",
        model=MNIST_2NN(),
        client=DDict(
            batch_size=32,
            local_epochs=1,
            model=MNIST_2NN(),
            loss=CrossEntropyLoss,
            optimizer=DDict(name=SGD, lr=0.1, momentum=0.9),
            scheduler=DDict(step_size=1, gamma=0.1),
        ),
        server=DDict(weighted=True),
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist)
    fl = PersonalizedFL(2, splitter, hparams)

    assert fl.n_clients == 2
    assert fl.hyper_params == hparams
    assert len(fl.clients) == 2
    assert isinstance(fl.clients[0], Client)
    assert isinstance(fl.server, Server)
    assert isinstance(fl.server.model, MNIST_2NN)
    # assert isinstance(fl.clients[0].model, MNIST_2NN)
    assert fl.clients[0].test_set is None
    assert isinstance(fl.clients[0].hyper_params.loss_fn, CrossEntropyLoss)

    fl.run(1, 0.5)
    with tempfile.TemporaryDirectory() as tmpdirname:
        temppath = fl.save(tmpdirname)
        fl2 = PersonalizedFL(2, splitter, hparams)
        fl2.load(temppath)

        assert fl2.server.rounds == fl.server.rounds

    shutil.rmtree(f"tests/tmp/tmp_{fl.id}")


def get_splitter(cfg):
    dataset = Datasets.get(**cfg.data.dataset)
    splitter = DataSplitter(
        dataset=dataset,
        distribution=cfg.data.distribution.name,
        dist_args=cfg.data.distribution.exclude("name"),
        **cfg.data.exclude("dataset", "distribution"),
    )
    return splitter


SPLITTER = None


def _test_algo(exp_config, alg_config, oncpu=True, tol=1e-5):
    accs = []
    cfg = Configuration(exp_config, alg_config)
    if oncpu:
        cfg.exp.device = "cpu"
    else:
        # if torch.cuda.is_available():
        #     cfg.exp.device = "cuda"
        # elif torch.backends.mps.is_available():
        #     cfg.exp.device = "mps"
        # else:
        return None, None
    FlukeENV().configure(cfg)
    gc.collect()
    global SPLITTER
    if SPLITTER is None:
        dataset = Datasets.get(**cfg.data.dataset)
        SPLITTER = DataSplitter(
            dataset=dataset,
            distribution=cfg.data.distribution.name,
            dist_args=cfg.data.distribution.exclude("name"),
            **cfg.data.exclude("dataset", "distribution"),
        )
    fl_algo_class = get_class_from_qualified_name(cfg.method.name)

    for mem in [False, True]:
        FlukeENV().configure(cfg)
        cfg.exp.inmemory = mem
        FlukeENV().set_inmemory(mem)
        algo = fl_algo_class(cfg.protocol.n_clients, SPLITTER, cfg.method.hyperparameters)

        log = Log()
        log.init(**cfg)
        algo.set_callbacks(log)
        algo.run(cfg.protocol.n_rounds, cfg.protocol.eligible_perc)
        FlukeENV().close_cache()
        shutil.rmtree(f"tests/tmp/tmp_{algo.id}")
        del algo
        if log.tracker["global"] and log.tracker.get("global", cfg.protocol.n_rounds):
            accs.append(log.tracker.get("global", cfg.protocol.n_rounds)["accuracy"])
        else:
            accs.append(log.tracker.summary("post-fit", cfg.protocol.n_rounds)["accuracy"])
    assert np.allclose(accs[0], accs[1], atol=tol)
    return None, log


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


def test_fat():
    fat, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fat.yaml")


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


def test_fedld():
    fedlc, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedld.yaml")
    # fedlc, log = _test_algo("./tests/configs/exp.yaml",
    #


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
    fedsam, log = _test_algo(
        "./tests/configs/exp.yaml", "./tests/configs/alg/fedsam.yaml", tol=1e-3
    )
    # fedsam, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedsam.yaml", oncpu=False)


def test_fedsgd():
    fedsgd, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/fedsgd.yaml")
    # fedsgd, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/fedsgd.yaml", oncpu=False)


def test_gear():
    gear, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/gear.yaml")
    # gear, log = _test_algo("./tests/configs/exp.yaml",
    #                          "./tests/configs/alg/gear.yaml", oncpu=False)


def test_kafe():
    kafe, log = _test_algo("./tests/configs/exp.yaml", "./tests/configs/alg/kafe.yaml")


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
    # test_fat()
    # test_fedamp()
    # test_fedavg()
    # test_fedavgm()
    # test_fedaws()
    # test_fedbabu()
    # test_fedbn()
    # test_feddyn()
    # test_fedexp()
    # test_fedhp()
    # test_fedlc()
    # test_fedld()
    # test_fednh()
    # test_fednova()
    # test_fedopt()
    # test_fedper()
    # test_fedprox()
    # test_fedproto()
    # test_fedrep()
    # test_fedrod()
    # test_fedrs()
    test_fedsam()
    # test_fedsgd()
    # test_gear()
    # test_kafe()
    # test_lgfedavg()
    # test_moon()
    # test_per_fedavg()
    # test_pfedme()
    # test_scaffold()
    # test_superfed()
