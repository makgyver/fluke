import json
import shutil
import sys
import tempfile
from unittest.mock import patch

import pytest
import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from fluke.utils import (
    ClientObserver,
    ServerObserver,
    bytes2human,
    cache_obj,
    clear_cuda_cache,
    flatten_dict,
    get_class_from_qualified_name,
    get_class_from_str,
    get_full_classname,
    get_loss,
    get_model,
    get_optimizer,
    get_scheduler,
    import_module_from_str,
    memory_usage,
    plot_distribution,
    retrieve_obj,
)
from fluke.utils.model import (
    AllLayerOutputModel,
    MMMixin,
    aggregate_models,
    batch_norm_to_group_norm,
    check_model_fit_mem,
    diff_model,
    flatten_parameters,
    get_activation_size,
    get_global_model_dict,
    get_local_model_dict,
    get_output_shape,
    get_trainable_keys,
    merge_models,
    mix_networks,
    optimizer_to,
    safe_load_state_dict,
    set_lambda_model,
    state_dict_zero_like,
    unwrap,
)

sys.path.append(".")
sys.path.append("..")


from fluke import DDict, FlukeENV  # NOQA
from fluke.algorithms import CentralizedFL  # NOQA
from fluke.client import Client  # NOQA
from fluke.comm import Message  # NOQA
from fluke.config import Configuration, ConfigurationError, OptimizerConfigurator  # NOQA
from fluke.data import DataSplitter  # NOQA
from fluke.data.datasets import Datasets  # NOQA
from fluke.nets import MNIST_2NN, VGG9, FedBN_CNN, Shakespeare_LSTM  # NOQA
from fluke.server import Server  # NOQA
from fluke.utils.log import DebugLog, Log, TensorboardLog, get_logger  # NOQA
from fluke.utils.model import STATE_DICT_KEYS_TO_IGNORE  # NOQA


def test_optimcfg():
    opt_cfg = OptimizerConfigurator(
        optimizer_cfg=dict(name=SGD, lr=0.1, momentum=0.9),
        scheduler_cfg=dict(step_size=1, gamma=0.1),
    )

    assert opt_cfg.optimizer == SGD
    assert opt_cfg.scheduler_cfg == {"step_size": 1, "gamma": 0.1}

    assert opt_cfg.optimizer_cfg == {"lr": 0.1, "momentum": 0.9}

    opt_cfg = OptimizerConfigurator(
        optimizer_cfg=DDict(name="SGD", lr=0.1, momentum=0.9),
        scheduler_cfg=DDict(name=StepLR, step_size=1, gamma=0.1),
    )

    assert opt_cfg.optimizer == SGD
    assert opt_cfg.scheduler == StepLR

    assert opt_cfg.optimizer_cfg == {"lr": 0.1, "momentum": 0.9}

    opt, sch = opt_cfg(Linear(10, 10))

    assert isinstance(opt, SGD)
    assert isinstance(sch, StepLR)
    assert opt.defaults["lr"] == 0.1
    assert opt.defaults["momentum"] == 0.9
    assert sch.step_size == 1
    assert sch.gamma == 0.1
    assert (
        str(opt_cfg).replace(" ", "").replace("\n", "").replace("\t", "")
        == "OptCfg(SGD,lr=0.1,momentum=0.9,StepLR(step_size=1,gamma=0.1))"
    )
    assert str(opt_cfg) == opt_cfg.__repr__()

    with pytest.raises(ValueError):
        opt_cfg = OptimizerConfigurator(optimizer_cfg=dict(name=1, lr=0.1, momentum=0.9))
    with pytest.raises(ValueError):
        opt_cfg = OptimizerConfigurator(
            optimizer_cfg=DDict(name=SGD, lr=0.1, momentum=0.9),
            scheduler_cfg=[DDict(step_size=1, gamma=0.1)],
        )

    with pytest.raises(ValueError):
        opt_cfg = OptimizerConfigurator(
            optimizer_cfg=[DDict(name=SGD, lr=0.1, momentum=0.9)],
            scheduler_cfg=DDict(step_size=1, gamma=0.1),
        )

    with pytest.raises(ValueError):
        opt_cfg = OptimizerConfigurator(
            optimizer_cfg=DDict(name=SGD, lr=0.1, momentum=0.9),
            scheduler_cfg=[DDict(name="Pippo", step_size=1, gamma=0.1)],
        )


def test_functions():
    # try:
    client_module = import_module_from_str("fluke.client")
    client_class = get_class_from_str("fluke.client", "Client")
    assert client_class == Client
    model = get_model("MNIST_2NN")
    assert model.__class__.__name__ == "MNIST_2NN"
    model2 = get_model("fluke.nets.MNIST_2NN")
    assert model2.__class__.__name__ == "MNIST_2NN"
    linear = get_class_from_qualified_name("torch.nn.Linear")
    assert linear == Linear
    full_linear = get_full_classname(Linear)
    optim = get_optimizer("SGD")
    assert optim == SGD
    loss = get_loss("CrossEntropyLoss")
    assert isinstance(loss, CrossEntropyLoss)
    scheduler = get_scheduler("StepLR")
    assert scheduler == StepLR
    logger = get_logger("Log")
    assert isinstance(logger, Log)
    clear_cuda_cache()
    clear_cuda_cache(True)
    # except Exception:
    #     pytest.fail("Unexpected error!")

    rss, vms, cud = memory_usage()
    assert rss >= 0
    assert vms >= 0
    assert cud >= 0
    if not torch.cuda.is_available():
        assert cud == 0

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
        torch.nn.Conv2d(3, 6, 3, 1, 1), torch.nn.BatchNorm2d(6), torch.nn.ReLU()
    )

    model4 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 3, 1, 1), torch.nn.BatchNorm2d(6), torch.nn.ReLU()
    )

    batch = torch.randn(1, 3, 28, 28)
    model4(batch)

    assert bytes2human(100) == "100 B"

    sd = state_dict_zero_like(model4.state_dict())
    assert list(sd.keys()) == list(model4.state_dict().keys())
    for k in sd.keys():
        assert torch.all(sd[k] == 0)

    prev_state_dict = model3.state_dict()
    safe_load_state_dict(model3, model4.state_dict())
    for k in model3.state_dict().keys():
        if k.endswith(STATE_DICT_KEYS_TO_IGNORE):
            assert torch.all(prev_state_dict[k] == model3.state_dict()[k])
        else:
            assert torch.all(model3.state_dict()[k] == model4.state_dict()[k])

    FlukeENV().set_inmemory(False)
    FlukeENV().open_cache("test_utils")
    cache_obj(model3, "test")
    load_model3 = retrieve_obj("test")

    server = Server(model3, None, [], None, None)
    cache_obj(model3, "model", server)
    load_cmodel = retrieve_obj("model", server)

    assert load_model3 is not None
    assert load_cmodel is not None
    assert isinstance(load_model3, torch.nn.Sequential)
    assert isinstance(load_cmodel, torch.nn.Sequential)

    tkeys = get_trainable_keys(model3)
    assert len(tkeys) == 4
    assert list(tkeys) == ["0.weight", "0.bias", "1.weight", "1.bias"]

    FlukeENV().get_cache().cleanup()
    FlukeENV().close_cache()
    cache_obj(None, "test")

    for k in model3.state_dict().keys():
        assert torch.all(model3.state_dict()[k] == load_model3.state_dict()[k])

    # client = Client(1, None, None, None, None)
    # client.model = model3
    # cache_model(client, ["model"])
    # load_model3 = retrieve_model(client, ["model"])["model"]

    for k in model3.state_dict().keys():
        assert torch.all(model3.state_dict()[k] == load_model3.state_dict()[k])

    d = {"a": 1, "b": 2, "c": {"d": 3}}
    dflat = flatten_dict(d)
    assert dflat == {"a": 1, "b": 2, "c.d": 3}


def test_configuration():

    cfg = dict(
        {
            "protocol": {"n_clients": 100, "n_rounds": 50, "eligible_perc": 0.1},
            "data": {
                "dataset": {"name": "mnist"},
                "distribution": {"name": "iid"},
                "client_split": 0.1,
                "sampling_perc": 1,
            },
            "save": {"path": "temp", "save_every": 1, "global_only": True},
            "exp": {"inmmemory": True, "seed": 42, "device": "cpu"},
            "logger": {"name": "local"},
        }
    )
    cfg_alg = dict(
        {
            "name": "fluke.algorithms.fedavg.FedAVG",
            "hyperparameters": {
                "server": {"weighted": True},
                "client": {
                    "batch_size": 10,
                    "local_epochs": 5,
                    "loss": "CrossEntropyLoss",
                    "optimizer": {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001},
                    "scheduler": {"step_size": 1, "gamma": 1},
                },
                "model": "MNIST_2NN",
            },
        }
    )

    temp_cfg = tempfile.NamedTemporaryFile(mode="w")
    temp_cfg_alg = tempfile.NamedTemporaryFile(mode="w")
    json.dump(cfg, open(temp_cfg.name, "w"))
    json.dump(cfg_alg, open(temp_cfg_alg.name, "w"))

    try:
        conf = Configuration(temp_cfg.name, temp_cfg_alg.name)
    except Exception:
        pytest.fail("Unexpected error!")

    cfg["exp"]["seed"] = [42, 50, 133]
    temp_cfg = tempfile.NamedTemporaryFile(mode="w")
    temp_cfg_alg = tempfile.NamedTemporaryFile(mode="w")
    json.dump(cfg, open(temp_cfg.name, "w"))
    json.dump(cfg_alg, open(temp_cfg_alg.name, "w"))

    gencfg = Configuration.sweep(temp_cfg.name, temp_cfg_alg.name)
    assert len(list(gencfg)) == 3

    cfg["exp"]["seed"] = 42

    assert conf.protocol.n_clients == 100
    # assert conf.data.dataset.name == "mnist"
    assert conf.exp.seed == 42
    # assert conf.logger.name == "local"
    assert conf.client.local_epochs == 5
    assert conf.server.weighted
    assert conf.model == "MNIST_2NN"

    assert conf.__repr__() == str(conf)

    cfg = dict({"protocol": {}, "data": {}, "exp": {}, "logger": {}})
    cfg_alg = dict(
        {
            "name": "fluke.algorithms.fedavg.FedAVG",
            "hyperparameters": {"server": {}, "client": {}, "model": "MNIST_2NN"},
        }
    )

    temp_cfg = tempfile.NamedTemporaryFile(mode="w")
    temp_cfg_alg = tempfile.NamedTemporaryFile(mode="w")
    json.dump(cfg, open(temp_cfg.name, "w"))
    json.dump(cfg_alg, open(temp_cfg_alg.name, "w"))

    with pytest.raises(ConfigurationError):
        conf = Configuration(temp_cfg.name, temp_cfg_alg.name)

    cfg["logger"]["name"] = "WandBLog"
    cfg_ = cfg.copy()
    cfg_["method"] = cfg_alg
    with pytest.raises(ConfigurationError):
        Configuration.from_dict(DDict(cfg_))


class MyLog(Log):
    def __init__(self, value: int):
        super().__init__()
        self.value = value


def test_log():

    log = Log()
    log.init(test="hello")

    try:
        # log.comm_costs[0] = 1  # for testing
        log.start_round(1, None)
        # log.comm_costs[0] = 0  # for testing
        log.selected_clients(1, [1, 2, 3])
        log.message_received("testA", Message("test", "test", None))
        log.server_evaluation(1, "global", {"accuracy": 1})
        log.client_evaluation(1, 1, "pre-fit", {"accuracy": 0.6})
        log.end_round(1)
        log.finished(1)
        log.early_stop(1)
        temp = tempfile.NamedTemporaryFile(mode="w")
        log.save(temp.name)
    except Exception:
        pytest.fail("Unexpected error!")

    with open(temp.name, "r") as f:
        data = dict(json.load(f))
        # assert data["mem_costs"]["1"] > 0
        # del data["mem_costs"]
        print(data)
        assert data == {
            "perf_global": {"1": {"accuracy": 1}},
            "comm_costs": {"0": 0, "1": 4},
            "perf_locals": {},
            "perf_prefit": {"1": {"1": {"accuracy": 0.6}}},
            "perf_postfit": {},
            "custom_fields": {},
        }

    assert log.tracker["global"] == {1: {"accuracy": 1}}
    assert log.tracker["locals"] == {}
    assert log.tracker["pre-fit"] == {1: {1: {"accuracy": 0.6}}}
    assert log.tracker["post-fit"] == {}
    assert log.tracker.summary("locals", round=1) == {}
    assert log.tracker.summary("pre-fit", round=1, include_round=False) == {
        "accuracy": 0.6,
        "support": 1,
    }
    assert log.tracker.summary("post-fit", round=1) == {}
    assert log.tracker["comm"] == {0: 0, 1: 4}
    assert log.current_round == 1

    log.track_item(1, "test", 12)
    assert 1 in log.custom_fields
    assert log.custom_fields[1]["test"] == 12

    log = get_logger("tests.test_utils.MyLog", value=17)
    assert log.value == 17
    assert isinstance(log, MyLog)


def test_tensorboard_log():
    log = TensorboardLog(log_dir="tests/tmp/runs", name="pippo")
    log.init(test="hello")

    model = MNIST_2NN()
    try:
        # log.comm_costs[0] = 1  # for testing
        log.start_round(1, model)
        # log.comm_costs[0] = 0  # for testing
        log.selected_clients(1, [1, 2, 3])
        log.message_received("testA", Message("test", "test", None))
        log.server_evaluation(1, "global", {"accuracy": 1})
        log.client_evaluation(1, 1, "pre-fit", {"accuracy": 0.6})
        log.end_round(1)
        log.add_scalar("test", 1, 1)
        log.add_scalars("test", {"test1": 1, "test2": 2}, 1)
        log.finished(1)
        log.early_stop(1)
        temp = tempfile.NamedTemporaryFile(mode="w")
        log.save(temp.name)
        log.close()
    except Exception:
        pytest.fail("Unexpected error!")

    assert log.custom_fields[1]["test/test1"] == 1
    assert log.custom_fields[1]["test/test2"] == 2
    assert log.custom_fields[1]["test"] == 1

    shutil.rmtree("tests/tmp/runs")


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


def test_debuglog():

    log = DebugLog()
    log.init(test="hello")

    try:
        # log.comm_costs[0] = 1  # for testing
        log.start_round(1, None)
        # log.comm_costs[0] = 0  # for testing
        log.selected_clients(
            1, [Client(1, None, None, None, None), Client(2, None, None, None, None)]
        )
        log.message_received("testA", Message("test", "test", None))
        log.server_evaluation(1, "global", {"accuracy": 1})
        log.client_evaluation(1, 1, "pre-fit", {"accuracy": 0.6})
        log.client_evaluation(1, 2, "pre-fit", {"accuracy": 0.7})
        log.client_evaluation(1, 1, "post-fit", {"accuracy": 0.5})
        log.end_round(1)
        log.finished(2)
        log.message_broadcasted("testB", Message("test", "test", None))
        log.message_sent("testC", Message("test", "test", None))
        log.interrupted()
        log.early_stop(1)
        log.start_fit(1, 1, None)
        log.end_fit(1, 1, None, 0.1)
        temp = tempfile.NamedTemporaryFile(mode="w")
        log.save(temp.name)
    except Exception:
        pytest.fail("Unexpected error!")

    print(log.tracker._performance)
    with open(temp.name, "r") as f:
        data = dict(json.load(f))
        # assert data["mem_costs"]["1"] > 0
        # del data["mem_costs"]
        print(data)
        assert data == {
            "perf_global": {"1": {"accuracy": 1}},
            "comm_costs": {"0": 0, "1": 4},
            "perf_locals": {},
            "perf_prefit": {"1": {"1": {"accuracy": 0.6}, "2": {"accuracy": 0.7}}},
            "perf_postfit": {"1": {"1": {"accuracy": 0.5}}},
            "custom_fields": {},
        }

    assert log.tracker["global"] == {1: {"accuracy": 1}}
    assert log.tracker["locals"] == {}
    assert log.tracker["pre-fit"] == {1: {1: {"accuracy": 0.6}, 2: {"accuracy": 0.7}}}
    assert log.tracker["post-fit"] == {1: {1: {"accuracy": 0.5}}}
    assert log.tracker.summary("locals", round=1) == {}
    assert log.tracker.summary("pre-fit", round=1, include_round=False) == {
        "accuracy": 0.65,
        "support": 2,
    }
    assert log.tracker.summary("post-fit", round=1, include_round=False) == {
        "accuracy": 0.5,
        "support": 1,
    }
    assert log.tracker["comm"] == {0: 0, 1: 4}
    assert log.current_round == 1

    log.track_item(1, "test", 12)
    assert 1 in log.custom_fields
    assert log.custom_fields[1]["test"] == 12

    log = get_logger("tests.test_utils.MyLog", value=17)
    assert log.value == 17
    assert isinstance(log, MyLog)


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

    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.fc1 = Linear(2, 2)
            self.fc2 = Linear(2, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model1 = TestModel()
    model2 = torch.nn.DataParallel(TestModel())

    assert isinstance(unwrap(model2).fc1, torch.nn.Linear)
    assert isinstance(unwrap(model2).fc2, torch.nn.Linear)

    assert get_output_shape(model1, (1, 2)) == (1, 1)

    optimizer = SGD(model1.parameters(), lr=0.01, momentum=0.9)
    model1.to("mps")
    optimizer_to(optimizer, "mps")
    assert optimizer.param_groups[0]["params"][0].device.type == "mps"
    model1.to("cpu")
    optimizer_to(optimizer, "cpu")
    assert optimizer.param_groups[0]["params"][0].device.type == "cpu"


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
                torch.nn.MaxPool2d(2, 2),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(10),
                torch.nn.Conv2d(
                    in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(20),
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


# @patch("matplotlib.pyplot.show")
# def test_plot_dist(mock_show):
#     hparams = DDict(
#         # model="fluke.nets.MNIST_2NN",
#         model=MNIST_2NN(),
#         client=DDict(batch_size=32,
#                      local_epochs=1,
#                      loss=CrossEntropyLoss,
#                      optimizer=DDict(
#                          lr=0.1,
#                          momentum=0.9),
#                      scheduler=DDict(
#                          step_size=1,
#                          gamma=0.1)
#                      ),
#         server=DDict(weighted=True)
#     )
#     mnist = Datasets.MNIST("../data")
#     splitter = DataSplitter(mnist, client_split=0.1)
#     fl = CentralizedFL(10, splitter, hparams)
#     plot_distribution(fl.clients, "ball")
#     plot_distribution(fl.clients, "bar")
#     plot_distribution(fl.clients, "mat")
#     FlukeENV().close_cache()


@patch("matplotlib.pyplot.show")
def test_plot_dist_ball(mock_show):
    hparams = DDict(
        # model="fluke.nets.MNIST_2NN",
        model=MNIST_2NN(),
        client=DDict(
            batch_size=32,
            local_epochs=1,
            loss=CrossEntropyLoss,
            optimizer=DDict(lr=0.1, momentum=0.9),
            scheduler=DDict(step_size=1, gamma=0.1),
        ),
        server=DDict(weighted=True),
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist, client_split=0.1)
    fl = CentralizedFL(10, splitter, hparams)
    plot_distribution(fl.clients, plot_type="ball")
    FlukeENV().close_cache()


@patch("matplotlib.pyplot.show")
def test_plot_dist_bar(mock_show):
    hparams = DDict(
        # model="fluke.nets.MNIST_2NN",
        model=MNIST_2NN(),
        client=DDict(
            batch_size=32,
            local_epochs=1,
            loss=CrossEntropyLoss,
            optimizer=DDict(lr=0.1, momentum=0.9),
            scheduler=DDict(step_size=1, gamma=0.1),
        ),
        server=DDict(weighted=True),
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist, client_split=0.1)
    fl = CentralizedFL(10, splitter, hparams)
    plot_distribution(fl.clients, plot_type="bar")
    FlukeENV().close_cache()


@patch("matplotlib.pyplot.show")
def test_plot_dist_mat(mock_show):
    hparams = DDict(
        # model="fluke.nets.MNIST_2NN",
        model=MNIST_2NN(),
        client=DDict(
            batch_size=32,
            local_epochs=1,
            loss=CrossEntropyLoss,
            optimizer=DDict(lr=0.1, momentum=0.9),
            scheduler=DDict(step_size=1, gamma=0.1),
        ),
        server=DDict(weighted=True),
    )
    mnist = Datasets.MNIST("../data")
    splitter = DataSplitter(mnist, client_split=0.1)
    fl = CentralizedFL(10, splitter, hparams)
    plot_distribution(fl.clients, plot_type="mat")
    FlukeENV().close_cache()


def test_check_mem():
    net = MNIST_2NN()

    if torch.backends.mps.is_available():
        assert check_model_fit_mem(net, (28 * 28,), 100, "mps", True)

    if torch.cuda.is_available():
        assert check_model_fit_mem(net, (28 * 28,), 100, "cuda")

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            check_model_fit_mem(net, (28 * 28,), 100, "cuda")


def test_get_activation_size():
    net = MNIST_2NN()
    x = torch.randn(1, 28 * 28)
    assert 10 == get_activation_size(net, None)
    assert 10 == get_activation_size(net, x)

    net = FedBN_CNN()
    x = torch.randn(1, 1, 28, 28)
    with pytest.raises(ValueError):
        get_activation_size(net.encoder, None)
    assert 10 == get_activation_size(net, x)
    assert 6272 == get_activation_size(net.encoder, x)


def test_agg():
    loss = CrossEntropyLoss()
    net1 = FedBN_CNN()
    net2 = FedBN_CNN()
    optimizer = SGD(net1.parameters(), lr=0.01)
    optimizer2 = SGD(net2.parameters(), lr=0.01)
    x = torch.randn(10, 1, 28, 28)
    net1.zero_grad()
    net2.zero_grad()
    loss1 = loss(net1(x), torch.randint(0, 10, (10,)))
    loss2 = loss(net2(x), torch.randint(0, 10, (10,)))
    loss1.backward()
    loss2.backward()
    optimizer.step()
    optimizer2.step()
    net = FedBN_CNN()
    _ = aggregate_models(net, [net1, net2], [0.5, 0.5], eta=1.0)


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
    # test_plot_dist()
    test_check_mem()
    test_alllayeroutput()

    # 91% coverage utils.__init__
    # 95% coverage utils.model
