from __future__ import annotations

import gc
import shutil
import sys

import numpy as np

from fluke import FlukeENV
from fluke.config import Configuration
from fluke.data import DataSplitter
from fluke.data.datasets import Datasets
from fluke.utils import get_class_from_qualified_name
from fluke.utils.log import Log
from fluke.evaluation import ClassificationEval

sys.path.append(".")
sys.path.append("..")

SPLITTER = None

FlukeENV().set_evaluator(ClassificationEval(1, 10))
FlukeENV().set_eval_cfg(post_fit=True, pre_fit=True)
FlukeENV().set_save_options(path="tests/tmp/tmp", save_every=1, global_only=True)

def _test_algo(exp_config, alg_config, oncpu=True, tol=1e-5, policy="random"):
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
    cfg.client.policy = policy

    for mem in [False, True]:
        FlukeENV().configure(cfg)
        cfg.exp.inmemory = mem
        FlukeENV().set_inmemory(mem)
        algo = fl_algo_class(cfg.protocol.n_clients, SPLITTER, cfg.method.hyperparameters)
        
        strfl = (
            f"GossipDFL[{algo.id}](model=MNIST_2NN()" + "server={'weighted':True},GossipClient[0-99](" + \
            f"optim=OptCfg(SGD,lr=0.1,momentum=0.9,StepLR(gamma=1,step_size=1))," + \
            f"batch_size=128,loss_fn=CrossEntropyLoss(),local_epochs=1,fine_tuning_epochs=0,clipping=0," + \
            f"activation_rate=1,policy={policy},eta=1.0))"
        )


        assert str(algo).replace(" ", "").replace("\n", "").replace("\t", "") == strfl
        assert algo.__repr__().replace(" ", "").replace("\n", "").replace("\t", "") == strfl

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


def test_decentralized():
    gossip, log = _test_algo("./tests/configs/exp_dec.yaml", "./tests/configs/alg/gossip.yaml", policy="random")
    gossip, log = _test_algo("./tests/configs/exp_dec.yaml", "./tests/configs/alg/gossip.yaml", policy="best")
    gossip, log = _test_algo("./tests/configs/exp_dec.yaml", "./tests/configs/alg/gossip.yaml", policy="aggregate")
    gossip, log = _test_algo("./tests/configs/exp_dec.yaml", "./tests/configs/alg/gossip.yaml", policy="last")



if __name__ == "__main__":
    test_decentralized()
