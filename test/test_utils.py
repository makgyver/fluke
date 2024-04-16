from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import pytest
import json
import tempfile
import sys
sys.path.append(".")
sys.path.append("..")


from fl_bench.client import Client  # NOQA
from fl_bench.utils import (OptimizerConfigurator, import_module_from_str,  # NOQA
           get_class_from_str, get_model, get_class_from_qualified_name,  # NOQA
           get_full_classname, get_loss, get_scheduler, clear_cache, Configuration)  # NOQA


def test_optimcfg():
    opt_cfg = OptimizerConfigurator(
        optimizer_class=SGD,
        scheduler_kwargs={
            "step_size": 1,
            "gamma": 0.1
        },
        lr=0.1,
        momentum=0.9
    )

    assert opt_cfg.optimizer == SGD
    assert opt_cfg.scheduler_kwargs == {
        "step_size": 1,
        "gamma": 0.1
    }

    assert opt_cfg.optimizer_kwargs == {
        "lr": 0.1,
        "momentum": 0.9
    }

    opt, sch = opt_cfg(Linear(10, 10))

    assert isinstance(opt, SGD)
    assert isinstance(sch, StepLR)
    assert opt.defaults["lr"] == 0.1
    assert opt.defaults["momentum"] == 0.9
    assert sch.step_size == 1
    assert sch.gamma == 0.1
    assert str(opt_cfg) == "OptCfg(SGD,lr=0.1,momentum=0.9,StepLR(step_size=1,gamma=0.1))"


def test_functions():
    try:
        client_module = import_module_from_str("fl_bench.client")
        client_class = get_class_from_str("fl_bench.client", "Client")
        model = get_model("MNIST_2NN")
        linear = get_class_from_qualified_name("torch.nn.Linear")
        full_linear = get_full_classname(Linear)
        loss = get_loss("CrossEntropyLoss")
        scheduler = get_scheduler("StepLR")
        clear_cache()
        clear_cache(True)
    except Exception:
        pytest.fail("Unexpected error!")

    assert client_module.__name__ == "fl_bench.client"
    assert client_class == Client
    assert model.__class__.__name__ == "MNIST_2NN"
    assert linear == Linear
    assert full_linear == "torch.nn.modules.linear.Linear"
    assert isinstance(loss, CrossEntropyLoss)
    assert scheduler == StepLR


def test_configuration():

    cfg = dict({
        'protocol': {
            'n_clients': 100,
            'n_rounds': 50,
            'eligible_perc': 0.1
        },
        'data': {
            'dataset': {
                'name': 'mnist'
            },
            'standardize': False,
            'distribution': {
                'name': "iid"
            },
            'client_split': 0.1,
            'sampling_perc': 1
        },
        'exp': {
            'seed': 42,
            'average': 'micro',
            'device': 'cpu'
        },
        'logger': {
            'name': 'local'
        }
    })
    cfg_alg = dict({
        'name': 'fl_bench.algorithms.fedavg.FedAVG',
        'hyperparameters': {
            'server': {
                'weighted': True
            },
            'client': {
                'batch_size': 10,
                'local_epochs': 5,
                'loss': 'CrossEntropyLoss',
                'optimizer': {
                    'lr': 0.01,
                    'momentum': 0.9,
                    'weight_decay': 0.0001
                },
                'scheduler': {
                    'step_size': 1,
                    'gamma': 1
                }
            },
            'model': 'MNIST_2NN'
        }
    })

    temp_cfg = tempfile.NamedTemporaryFile(mode="w")
    temp_cfg_alg = tempfile.NamedTemporaryFile(mode="w")
    json.dump(cfg, open(temp_cfg.name, "w"))
    json.dump(cfg_alg, open(temp_cfg_alg.name, "w"))

    try:
        conf = Configuration(temp_cfg.name, temp_cfg_alg.name)
    except Exception:
        pytest.fail("Unexpected error!")

    assert conf.protocol.n_clients == 100
    # assert conf.data.dataset.name == "mnist"
    assert conf.exp.seed == 42
    # assert conf.logger.name == "local"

    assert str(conf) == "fl_bench.algorithms.fedavg.FedAVG" + \
        "_data(mnist, iid)_proto(C100, R50,E0.1)_seed(42)"


if __name__ == "__main__":
    test_optimcfg()
    test_functions()
    test_configuration()
