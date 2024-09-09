import json
import sys
import tempfile
from unittest.mock import patch

import pytest
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

sys.path.append(".")
sys.path.append("..")


from fluke import DDict  # NOQA
from fluke.algorithms import CentralizedFL  # NOQA
from fluke.client import Client  # NOQA
from fluke.comm import Message  # NOQA
from fluke.data import DataSplitter  # NOQA
from fluke.data.datasets import Datasets  # NOQA
from fluke.nets import MNIST_2NN, VGG9, FedBN_CNN, Shakespeare_LSTM  # NOQA
from fluke.utils import (ClientObserver, Configuration,  # NOQA
                         OptimizerConfigurator, ServerObserver, clear_cache,
                         get_class_from_qualified_name, get_class_from_str,
                         get_full_classname, get_loss, get_model,
                         get_scheduler, import_module_from_str,
                         plot_distribution)
from fluke.utils.log import Log, get_logger  # NOQA
from fluke.utils.model import (STATE_DICT_KEYS_TO_IGNORE,  # NOQA
                               AllLayerOutputModel, MMMixin,
                               batch_norm_to_group_norm, check_model_fit_mem,
                               diff_model, flatten_parameters,
                               get_global_model_dict, get_local_model_dict,
                               merge_models, mix_networks,
                               safe_load_state_dict, set_lambda_model)


def test_optimcfg():
    opt_cfg = OptimizerConfigurator(
        optimizer_cfg=dict(name=SGD,
                           lr=0.1,
                           momentum=0.9),
        scheduler_cfg=dict(
            step_size=1,
            gamma=.1
        ),
    )

    assert opt_cfg.optimizer == SGD
    assert opt_cfg.scheduler_cfg == {
        "step_size": 1,
        "gamma": 0.1
    }

    assert opt_cfg.optimizer_cfg == {
        "lr": 0.1,
        "momentum": 0.9
    }

    opt_cfg = OptimizerConfigurator(
        optimizer_cfg=DDict(name=SGD,
                            lr=0.1,
                            momentum=0.9),
        scheduler_cfg=DDict(
            step_size=1,
            gamma=.1
        ),
    )

    assert opt_cfg.optimizer == SGD
    assert opt_cfg.scheduler == StepLR

    assert opt_cfg.optimizer_cfg == {
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
    assert str(opt_cfg) == "OptCfg(SGD, lr=0.1, momentum=0.9, StepLR(step_size=1, gamma=0.1))"
    assert str(opt_cfg) == opt_cfg.__repr__()

    with pytest.raises(ValueError):
        opt_cfg = OptimizerConfigurator(
            optimizer_cfg=dict(name=1,
                               lr=0.1,
                               momentum=0.9))
    with pytest.raises(ValueError):
        opt_cfg = OptimizerConfigurator(
            optimizer_cfg=DDict(name=SGD,
                                lr=0.1,
                                momentum=0.9),
            scheduler_cfg=[DDict(
                step_size=1,
                gamma=.1
            )]
        )

    with pytest.raises(ValueError):
        opt_cfg = OptimizerConfigurator(
            optimizer_cfg=[DDict(name=SGD,
                                 lr=0.1,
                                 momentum=0.9)],
            scheduler_cfg=DDict(
                step_size=1,
                gamma=.1
            )
        )

    with pytest.raises(ValueError):
        opt_cfg = OptimizerConfigurator(
            optimizer_cfg=DDict(name=SGD,
                                lr=0.1,
                                momentum=0.9),
            scheduler_cfg=[DDict(
                name="Pippo",
                step_size=1,
                gamma=.1
            )]
        )


def test_functions():
    try:
        client_module = import_module_from_str("fluke.client")
        client_class = get_class_from_str("fluke.client", "Client")
        model = get_model("MNIST_2NN")
        model2 = get_model("fluke.nets.MNIST_2NN")
        linear = get_class_from_qualified_name("torch.nn.Linear")
        full_linear = get_full_classname(Linear)
        loss = get_loss("CrossEntropyLoss")
        scheduler = get_scheduler("StepLR")
        logger = get_logger("Log")
        clear_cache()
        clear_cache(True)
    except Exception:
        pytest.fail("Unexpected error!")

    assert client_module.__name__ == "fluke.client"
    assert client_class == Client
    assert model.__class__.__name__ == "MNIST_2NN"
    assert model2.__class__.__name__ == "MNIST_2NN"
    assert linear == Linear
    assert full_linear == "torch.nn.modules.linear.Linear"
    assert isinstance(loss, CrossEntropyLoss)
    assert scheduler == StepLR
    assert isinstance(logger, Log)

    model3 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 3, 1, 1),
        torch.nn.BatchNorm2d(6),
        torch.nn.ReLU()
    )

    model4 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 3, 1, 1),
        torch.nn.BatchNorm2d(6),
        torch.nn.ReLU()
    )

    batch = torch.randn(1, 3, 28, 28)
    model4(batch)

    prev_state_dict = model3.state_dict()
    safe_load_state_dict(model3, model4.state_dict())
    for k in model3.state_dict().keys():
        if k.endswith(STATE_DICT_KEYS_TO_IGNORE):
            assert torch.all(prev_state_dict[k] == model3.state_dict()[k])
        else:
            assert torch.all(model3.state_dict()[k] == model4.state_dict()[k])


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
        'name': 'fluke.algorithms.fedavg.FedAVG',
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
    assert conf.client.local_epochs == 5
    assert conf.server.weighted
    assert conf.model == "MNIST_2NN"

    print(str(conf))
    assert str(conf) == "fluke.algorithms.fedavg.FedAVG_data(mnist, iid())_proto(C100, R50, E0.1)_seed(42)"

    cfg = dict({"protocol": {}, "data": {}, "exp": {}, "logger": {}})
    cfg_alg = dict({"name": "fluke.algorithms.fedavg.FedAVG", "hyperparameters": {
                   "server": {}, "client": {}, "model": "MNIST_2NN"}})

    temp_cfg = tempfile.NamedTemporaryFile(mode="w")
    temp_cfg_alg = tempfile.NamedTemporaryFile(mode="w")
    json.dump(cfg, open(temp_cfg.name, "w"))
    json.dump(cfg_alg, open(temp_cfg_alg.name, "w"))

    with pytest.raises(ValueError):
        conf = Configuration(temp_cfg.name, temp_cfg_alg.name)


def test_log():
    log = Log()
    log.init(test="hello")

    try:
        log.comm_costs[0] = 1  # for testing
        log.start_round(1, None)
        log.comm_costs[0] = 0  # for testing
        log.selected_clients(1, [1, 2, 3])
        log.message_received(Message("test", "test", None))
        log.server_evaluation(1, "global", {"accuracy": 1})
        log.client_evaluation(1, 1, 'pre-fit', {"accuracy": 0.6})
        log.end_round(1)
        log.finished(1)
        temp = tempfile.NamedTemporaryFile(mode="w")
        log.save(temp.name)
    except Exception:
        pytest.fail("Unexpected error!")

    with open(temp.name, "r") as f:
        data = dict(json.load(f))
        assert data == {'perf_global': {'1': {'accuracy': 1}}, 'comm_costs': {
            '0': 0, '1': 4}, 'perf_locals': {}, 'perf_prefit': {'1': {'accuracy': 0.6}},
            'perf_postfit': {}}

    assert log.global_eval == {1: {"accuracy": 1}}
    assert log.locals_eval == {}
    assert log.prefit_eval == {1: {1: {"accuracy": 0.6}}}
    assert log.postfit_eval == {}
    assert log.locals_eval_summary == {}
    assert log.prefit_eval_summary == {1: {"accuracy": 0.6}}
    assert log.postfit_eval_summary == {}
    assert log.comm_costs == {0: 0, 1: 4}
    assert log.current_round == 1


# def test_wandb_log():
#     log2 = WandBLog()
#     log2.init()
#     try:
#         log2.comm_costs[0] = 1  # for testing
#         log2.start_round(1, None)
#         log2.comm_costs[0] = 0  # for testing
#         log2.selected_clients(1, [1, 2, 3])
#         log2.message_received(Message("test", "test", None))
#         log2.error("test")
#         log2.end_round(1, {"accuracy": 1}, [{"accuracy": 0.7}, {"accuracy": 0.5}])
#         log2.finished([{"accuracy": 0.7}, {"accuracy": 0.5}, {"accuracy": 0.6}])
#         temp = tempfile.NamedTemporaryFile(mode="w")
#         log2.save(temp.name)
#     except Exception:
#         pytest.fail("Unexpected error!")

#     with open(temp.name, "r") as f:
#         data = dict(json.load(f))
#         assert data == {'perf_global': {'1': {'accuracy': 1}}, 'comm_costs': {
#             '0': 0, '1': 4}, 'perf_local': {'1': {'accuracy': 0.6}, '2': {'accuracy': 0.6}}}

#     assert log2.history[1] == {"accuracy": 1}
#     assert log2.client_history[1] == {"accuracy": 0.6}
#     assert log2.comm_costs[1] == 4


def test_models():
    model1 = Linear(2, 1)
    model1.weight.data.fill_(1)
    model1.bias.data.fill_(1)

    model2 = Linear(2, 1)
    model2.weight.data.fill_(2)
    model2.bias.data.fill_(2)

    model3 = merge_models(model1, model2, 0.5)
    assert model3.weight.data[0, 0] == 1.5
    assert model3.weight.data[0, 1] == 1.5
    assert model3.bias.data[0] == 1.5

    model4 = merge_models(model1, model2, 0.75)
    assert model4.weight.data[0, 0] == 1.75
    assert model4.weight.data[0, 1] == 1.75
    assert model4.bias.data[0] == 1.75

    diffdict = diff_model(model1.state_dict(), model2.state_dict())
    assert diffdict["weight"].data[0, 0] == -1.0
    assert diffdict["weight"].data[0, 1] == -1.0
    assert diffdict["bias"].data[0] == -1.0

    model = FedBN_CNN()
    model_gn = batch_norm_to_group_norm(model)

    for n1, p1 in model.named_parameters():
        for n2, p2 in model_gn.named_parameters():
            if n1 == n2 and isinstance(p1, torch.nn.BatchNorm2d):
                assert isinstance(p2, torch.nn.GroupNorm)


def test_mixing():
    mixin = MMMixin()
    mixin.set_lambda(0.5)
    assert mixin.get_lambda() == 0.5

    model1 = Linear(2, 1)
    model1.weight.data.fill_(1)
    model1.bias.data.fill_(1)

    model2 = Linear(2, 1)
    model2.weight.data.fill_(2)
    model2.bias.data.fill_(2)
    mixed = mix_networks(model1, model2, 0.5)

    x = torch.FloatTensor([[1, 1]])
    y = mixed(x)
    assert y[0, 0] == 4.5

    assert hasattr(mixed, "lam")
    assert mixed.lam == 0.5
    assert mixed.get_lambda() == 0.5
    assert diff_model(get_local_model_dict(mixed), model2.state_dict())["weight"].data[0, 0] == 0.0
    assert diff_model(get_global_model_dict(mixed), model1.state_dict())["weight"].data[0, 0] == 0.0

    set_lambda_model(mixed, 0.3)

    assert mixed.lam == 0.3
    assert mixed.get_lambda() == 0.3

    weights = mixed.get_weight()
    assert weights[0].data[0, 0] == 0.7 + 0.3 * 2
    assert weights[0].data[0, 1] == 0.7 + 0.3 * 2
    assert weights[1].data[0] == 0.7 + 0.3 * 2

    model1 = MNIST_2NN()
    model2 = MNIST_2NN()
    mixed = mix_networks(model1, model2, 0.2)

    assert mixed.get_lambda() == 0.2

    model1 = VGG9()
    model2 = VGG9()
    mixed = mix_networks(model1, model2, 0.3)
    assert mixed.get_lambda() == 0.3

    x = torch.randn(1, 1, 28, 28)
    mixed(x)

    model1 = Shakespeare_LSTM()
    model2 = Shakespeare_LSTM()
    mixed = mix_networks(model1, model2, 0.4)
    assert mixed.get_lambda() == 0.4

    x = torch.randint(0, 100, (1, 10))
    mixed(x)

    class TestNet(torch.nn.Module):
        def __init__(self):
            super(TestNet, self).__init__()

            # Implement the sequential module for feature extraction
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
                torch.nn.MaxPool2d(2, 2), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(10),
                torch.nn.Conv2d(in_channels=10, out_channels=20,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.MaxPool2d(2, 2), torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(20)
            )

            # Implement the fully connected layer for classification
            self.fc = torch.nn.Linear(in_features=20 * 7 * 7, out_features=10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model1 = TestNet()
    model2 = TestNet()
    mixed = mix_networks(model1, model2, 0.3)
    assert mixed.get_lambda() == 0.3

    x = torch.randn(1, 1, 28, 28)
    mixed(x)


def test_serverobs():
    sobs = ServerObserver()
    sobs.start_round(1, None)
    sobs.server_evaluation(1, "global", {"accuracy": 1})
    sobs.end_round(1)
    sobs.finished(1)
    sobs.selected_clients(1, [1, 2, 3])


def test_clientobs():
    sobs = ClientObserver()
    sobs.start_fit(1, 17, None)
    sobs.client_evaluation(1, 17, "pre-fit", {"accuracy": 1})
    sobs.end_fit(1, 17, None, 0.1)


@patch("matplotlib.pyplot.show")
def test_plot_dist(mock_show):
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
    fl = CentralizedFL(10, splitter, hparams)
    plot_distribution(fl.clients, "ball")
    plot_distribution(fl.clients, "bar")
    plot_distribution(fl.clients, "mat")


def test_check_mem():
    net = MNIST_2NN()
    assert check_model_fit_mem(net, (28 * 28,), 100, "mps", True)

    if torch.cuda.is_available():
        assert check_model_fit_mem(net, (28 * 28,), 100, "cuda")


def test_alllayeroutput():
    net = MNIST_2NN()
    all_out = AllLayerOutputModel(net)
    x = torch.randn(1, 28 * 28)
    all_out(x)
    assert "_encoder.fc1" in all_out.activations_in
    assert "_encoder.fc2" in all_out.activations_in
    assert "_head.fc3" in all_out.activations_in
    assert "_encoder.fc1" in all_out.activations_out
    assert "_encoder.fc2" in all_out.activations_out
    assert "_head.fc3" in all_out.activations_out

    all_out.deactivate()
    all_out(x)
    assert all_out.activations_in == {}
    assert all_out.activations_out == {}


def test_flatten():
    net = MNIST_2NN()
    W = flatten_parameters(net)
    print(W.shape)
    assert W.shape[0] == 178110


if __name__ == "__main__":
    test_optimcfg()
    test_functions()
    test_configuration()
    test_log()
    # test_wandb_log()
    test_models()
    test_mixing()
    test_serverobs()
    test_clientobs()
    test_plot_dist()
    test_check_mem()
    test_alllayeroutput()

    # 91% coverage utils.__init__
    # 95% coverage utils.model
