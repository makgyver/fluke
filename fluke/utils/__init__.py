"""This module contains utility functions and classes used in ``fluke``."""
from __future__ import annotations

# from enum import Enum
import importlib
import inspect
import sys
import warnings
from typing import TYPE_CHECKING, Any, Iterable, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import seaborn as sns
import torch
import yaml
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

sys.path.append(".")
sys.path.append("..")

if TYPE_CHECKING:
    from client import Client  # NOQA

from .. import DDict  # NOQA

__all__ = [
    'log',
    'model',
    'Configuration',
    'OptimizerConfigurator',
    'ClientObserver',
    'ServerObserver',
    'clear_cache',
    'get_class_from_str',
    'get_class_from_qualified_name',
    'get_full_classname',
    'get_loss',
    'get_model',
    'get_optimizer',
    'get_scheduler',
    'import_module_from_str',
    'plot_distribution'
]


class ClientObserver():
    """Client observer interface.
    This interface is used to observe the client during the federated learning process.
    For example, it can be used to log the performance of the local model, as it is done by the
    ``Log`` class.
    """

    def start_fit(self, round: int, client_id: int, model: Module, **kwargs: dict[str, Any]):
        """This method is called when the client starts the local training process.

        Args:
            round (int): The round number.
            client_id (int):  The client ID.
            model (Module): The local model before training.
            **kwargs (dict): Additional keyword arguments.
        """
        pass

    def end_fit(self,
                round: int,
                client_id: int,
                model: Module,
                loss: float,
                **kwargs: dict[str, Any]):
        """This method is called when the client ends the local training process.

        Args:
            round (int): The round number.
            client_id (int): The client ID.
            model (Module): The local model after the local training.
            loss (float): The average loss incurred by the local model during training.
            **kwargs (dict): Additional keyword arguments.
        """
        pass

    def client_evaluation(self,
                          round: int,
                          client_id: int,
                          phase: Literal["pre-fit", "post-fit"],
                          evals: dict[str, float],
                          **kwargs: dict[str, Any]):
        """This method is called when the client evaluates the local model.
        The evaluation can be done before ('pre-fit') and/or after ('post-fit') the local
        training process. The 'pre-fit' evlauation is usually the evaluation of the global model on
        the local test set, and the 'post-fit' evaluation is the evaluation of the just updated
        local model on the local test set.

        Args:
            round (int): The round number.
            client_id (int): The client ID.
            phase (Literal['pre-fit', 'post-fit']): Whether the evaluation is done before or after
                the local training process.
            evals (dict[str, float]): The evaluation results.
            **kwargs (dict): Additional keyword arguments.
        """
        pass


class ServerObserver():
    """Server observer interface.
    This interface is used to observe the server during the federated learning process.
    For example, it can be used to log the performance of the global model and the communication
    costs, as it is done by the ``Log`` class.
    """

    def start_round(self, round: int, global_model: Any) -> None:
        """This method is called when a new round starts.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
        """
        pass

    def end_round(self, round: int) -> None:
        """This method is called when a round ends.

        Args:
            round (int): The round number.
        """
        pass

    def selected_clients(self, round: int, clients: Iterable) -> None:
        """This method is called when the clients have been selected for the current round.

        Args:
            round (int): The round number.
            clients (Iterable): The clients selected for the current round.
        """
        pass

    def server_evaluation(self,
                          round: int,
                          type: Literal["global", "locals"],
                          evals: Union[dict[str, float], dict[int, dict[str, float]]],
                          **kwargs: dict[str, Any]) -> None:
        """This method is called when the server evaluates the global or the local models on its
        test set.

        Args:
            round (int): The round number.
            type (Literal['global', 'locals']): The type of evaluation. If 'global', the evaluation
                is done on the global model. If 'locals', the evaluation is done on the local models
                of the clients on the test set of the server.
            evals (dict[str, float] | dict[int, dict[str, float]]): The evaluation metrics. In case
                of 'global' evaluation, it is a dictionary with the evaluation metrics. In case of
                'locals' evaluation, it is a dictionary of dictionaries where the keys are the
                client IDs and the values are the evaluation metrics.
        """
        pass

    def finished(self, round: int) -> None:
        """This method is called when the federated learning process has ended.

        Args:
            round (int): The last round number.
        """
        pass


class OptimizerConfigurator:
    """This class is used to configure the optimizer and the learning rate scheduler.

    Attributes:
        optimizer (type[Optimizer]): The optimizer class.
        scheduler (type[LRScheduler]): The learning rate scheduler class.
        optimizer_cfg (DDict): The optimizer keyword arguments.
        scheduler_cfg (DDict): The scheduler keyword arguments.
    """

    def __init__(self,
                 optimizer_cfg: DDict | dict,
                 scheduler_cfg: DDict | dict = None):
        """Initialize the optimizer configurator. In both the ``optimizer_cfg`` and the
        ``scheduler_cfg`` dictionaries, the key "name" is used to specify the optimizer and the
        scheduler, respectively. If not present, the default optimizer is the ``SGD`` optimizer, and
        the default scheduler is the ``StepLR`` scheduler. The other keys are the keyword arguments
        for the optimizer and the scheduler, respectively.

        Note:
            In the key "name" of the ``optimizer_cfg`` dictionary, you can specify the optimizer
            class directly, instead of its string name. Same for the scheduler.

        Args:
            optimizer_cfg (dict, DDict): The optimizer class.
            scheduler_cfg (dict or DDict, optional): The scheduler keyword arguments. If not
                specified, the default scheduler is the ``StepLR`` scheduler with a step size of 1
                and a gamma of 1. Defaults to ``None``.
        """

        self.optimizer_cfg: DDict = None
        self.scheduler_cfg: DDict = None
        self.optimizer: type[Optimizer] = None
        self.scheduler: type[LRScheduler] = None

        if isinstance(optimizer_cfg, DDict):
            self.optimizer_cfg = optimizer_cfg
        elif isinstance(optimizer_cfg, dict):
            self.optimizer_cfg = DDict(**optimizer_cfg)
        else:
            raise ValueError("Invalid optimizer configuration.")

        if scheduler_cfg is None:
            self.scheduler_cfg = DDict(name="StepLR", step_size=1, gamma=1)
        elif isinstance(scheduler_cfg, DDict):
            self.scheduler_cfg = scheduler_cfg
        elif isinstance(scheduler_cfg, dict):
            self.scheduler_cfg = DDict(**scheduler_cfg)
        else:
            raise ValueError("Invalid scheduler configuration.")

        if isinstance(self.optimizer_cfg.name, str):
            self.optimizer = get_optimizer(self.optimizer_cfg.name)
        elif inspect.isclass(self.optimizer_cfg.name) and \
                issubclass(self.optimizer_cfg.name, Optimizer):
            self.optimizer = self.optimizer_cfg.name
        else:
            raise ValueError("Invalid optimizer name. Must be a string or an optimizer class.")

        if "name" not in self.scheduler_cfg:
            self.scheduler = get_scheduler("StepLR")
        elif isinstance(self.scheduler_cfg.name, str):
            self.scheduler = get_scheduler(self.scheduler_cfg.name)
        elif inspect.isclass(self.scheduler_cfg.name) and \
                issubclass(self.scheduler_cfg.name, LRScheduler):
            self.scheduler = self.scheduler_cfg.name
        else:
            raise ValueError("Invalid scheduler name. Must be a string or a scheduler class.")

        self.scheduler_cfg = self.scheduler_cfg.exclude("name")
        self.optimizer_cfg = self.optimizer_cfg.exclude("name")

    def __call__(self, model: Module, **override_kwargs):
        """Creates the optimizer and the scheduler.

        Args:
            model (Module): The model whose parameters will be optimized.
            override_kwargs (dict): The optimizer's keyword arguments to override the default ones.

        Returns:
            tuple[Optimizer, StepLR]: The optimizer and the scheduler.
        """
        if override_kwargs:
            self.optimizer_cfg.update(**override_kwargs)
        optimizer = self.optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                   **self.optimizer_cfg)
        scheduler = self.scheduler(optimizer, **self.scheduler_cfg)
        return optimizer, scheduler

    def __str__(self) -> str:
        strsched = self.scheduler.__name__
        to_str = f"OptCfg({self.optimizer.__name__}, "
        to_str += ", ".join([f"{k}={v}" for k, v in self.optimizer_cfg.items()])
        to_str += f", {strsched}(" + ", ".join([f"{k}={v}" for k, v in self.scheduler_cfg.items()])
        to_str += "))"
        return to_str

    def __repr__(self) -> str:
        return str(self)


def import_module_from_str(name: str) -> Any:
    """Import a module from its name.

    Args:
        name (str): The name of the module.

    Returns:
        Any: The module.
    """
    components = name.split('.')
    mod = importlib.import_module(".".join(components[:-1]))
    mod = getattr(mod, components[-1])
    return mod


def get_class_from_str(module_name: str, class_name: str) -> Any:
    """Get a class from its name.
    This function is used to get a class from its name and the name of the module where it is
    defined. It is used to dynamically import classes.

    Args:
        module_name (str): The name of the module where the class is defined.
        class_name (str): The name of the class.

    Returns:
        Any: The class.
    """
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_class_from_qualified_name(qualname: str) -> Any:
    """Get a class from its fully qualified name.

    Args:
        qualname (str): The fully qualified name of the class.

    Returns:
        Any: The class.
    """
    module_name = ".".join(qualname.split(".")[:-1])
    class_name = qualname.split(".")[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_loss(lname: str) -> Module:
    """Get a loss function from its name.
    The supported loss functions are the ones defined in the ``torch.nn`` module.

    Args:
        lname (str): The name of the loss function.

    Returns:
        Module: The loss function.
    """
    return get_class_from_str("torch.nn", lname)()


def get_model(mname: str, **kwargs: dict[str, Any]) -> Module:
    """Get a model from its name.
    This function is used to get a torch model from its name and the name of the module where it is
    defined. It is used to dynamically import models. If ``mname`` is not a fully qualified name,
    the model is assumed to be defined in the ``fluke.nets`` module.

    Args:
        mname (str): The name of the model.
        **kwargs: The keyword arguments to pass to the model's constructor.

    Returns:
        Module: The model.
    """
    if "." in mname:
        module_name = ".".join(mname.split(".")[:-1])
        mname = mname.split(".")[-1]
    else:
        module_name: str = "fluke.nets"
    return get_class_from_str(module_name, mname)(**kwargs)


def get_full_classname(classtype: type) -> str:
    """Get the fully qualified name of a class.

    Args:
        classtype (type): The class.

    Returns:
        str: The fully qualified name of the class.

    Example:
        Let ``A`` be a class defined in the module ``fluke.utils``

        .. code-block:: python
            :linenos:

            # This is the content of the file fluke/utils.py
            class A:
                pass

            get_full_classname(A) # 'fluke.utils.A'

        If the class is defined in the ``__main__`` module, then:

        .. code-block:: python
            :linenos:

            if __name__ == "__main__":
                class B:
                    pass

                get_full_classname(B) # '__main__.B'
    """
    return f"{classtype.__module__}.{classtype.__name__}"


def get_optimizer(oname: str) -> type[Optimizer]:
    """Get an optimizer from its name.
    This function is used to get an optimizer from its name. It is used to dynamically import
    optimizers. The supported optimizers are the ones defined in the ``torch.optim`` module.

    Args:
        oname (str): The name of the optimizer.

    Returns:
        type[Optimizer]: The optimizer class.
    """
    return get_class_from_str("torch.optim", oname)


def get_scheduler(sname: str) -> type[LRScheduler]:
    """Get a learning rate scheduler from its name.
    This function is used to get a learning rate scheduler from its name. It is used to dynamically
    import learning rate schedulers. The supported schedulers are the ones defined in the
    ``torch.optim.lr_scheduler`` module.

    Args:
        sname (str): The name of the scheduler.

    Returns:
        torch.nn.Module: The learning rate scheduler.
    """
    return get_class_from_str("torch.optim.lr_scheduler", sname)


def clear_cache(ipc: bool = False):
    """Clear the CUDA cache. This function should be used to free the GPU memory after the training
    process has ended. It is usually used after the local training of the clients.

    Args:
        ipc (bool, optional): Whether to force collecting GPU memory after it has been released by
            CUDA IPC.
    """
    torch.cuda.empty_cache()
    if ipc and torch.cuda.is_available():
        torch.cuda.ipc_collect()


class Configuration(DDict):
    """Fluke configuration class.
    This class is used to store the configuration of an experiment. The configuration must adhere to
    a specific structure. The configuration is validated when the class is instantiated.

    Args:
        config_exp_path (str): The path to the experiment configuration file.
        config_alg_path (str): The path to the algorithm configuration file.

    Raises:
        ValueError: If the configuration is not valid.
    """

    def __init__(self, config_exp_path: str, config_alg_path: str):
        with open(config_exp_path) as f:
            config_exp = yaml.safe_load(f)
        with open(config_alg_path) as f:
            config_alg = yaml.safe_load(f)

        self.update(**config_exp)
        self.update(method=config_alg)
        self._validate()

    @property
    def client(self) -> DDict:
        """Get quick access to the client hyperparameters.

        Returns:
            DDict: The client hyperparameters.
        """
        return self.method.hyperparameters.client

    @property
    def server(self) -> DDict:
        """Get quick access to the server hyperparameters.

        Returns:
            DDict: The server hyperparameters.
        """
        return self.method.hyperparameters.server

    @property
    def model(self) -> DDict:
        """Get quick access to the model hyperparameters.

        Returns:
            DDict: The model hyperparameters.
        """
        return self.method.hyperparameters.model

    def _validate(self) -> bool:

        EXP_OPT_KEYS = {
            "device": "cpu",
            "seed": 42
        }

        LOG_OPT_KEYS = {
            "name": "Log"
        }

        EVAL_OPT_KEYS = {
            "task": "classification",
            "eval_every": 1,
            "pre_fit": False,
            "post_fit": True,
            "server": True,
            "locals": False
        }

        FIRST_LVL_KEYS = ["data", "protocol", "method"]
        FIRST_LVL_OPT_KEYS = {
            "eval": EVAL_OPT_KEYS,
            "exp": EXP_OPT_KEYS,
            "logger": LOG_OPT_KEYS
        }
        PROTO_REQUIRED_KEYS = ["n_clients", "eligible_perc", "n_rounds"]
        DATA_REQUIRED_KEYS = ["distribution", "dataset"]
        DATA_OPT_KEYS = {
            "sampling_perc": 1.0,
            "client_split": 0.0,
            "keep_test": True,
            "server_test": True,
            "server_split": 0.0,
            "uniform_test": False
        }

        ALG_1L_REQUIRED_KEYS = ["name", "hyperparameters"]
        HP_REQUIRED_KEYS = ["model", "client", "server"]
        CLIENT_HP_REQUIRED_KEYS = ["loss", "batch_size", "local_epochs", "optimizer"]
        WANDB_REQUIRED_KEYS = ["project", "entity"]

        error = False
        for k in FIRST_LVL_KEYS:
            if k not in self:
                f = "experiment" if k != "method" else "algorithm"
                rich.print(f"Error: {k} is required in the {f} configuration file")
                error = True

        for k, v in FIRST_LVL_OPT_KEYS.items():
            if k not in self:
                self[k] = DDict(**v)

        for k in PROTO_REQUIRED_KEYS:
            if k not in self.protocol:
                rich.print(f"Error: {k} is required for key 'protocol'.")
                error = True

        for k in DATA_REQUIRED_KEYS:
            if k not in self.data:
                rich.print(f"Error: {k} is required for key 'data'.")
                error = True

        for k, v in DATA_OPT_KEYS.items():
            if k not in self.data:
                self.data[k] = v

        for k, v in EVAL_OPT_KEYS.items():
            if k not in self.eval:
                self.eval[k] = v

        for k, v in EXP_OPT_KEYS.items():
            if k not in self.exp:
                self.exp[k] = v

        for k in ALG_1L_REQUIRED_KEYS:
            if k not in self.method:
                rich.print(f"Error: {k} is required for key 'method'.")
                error = True

        for k in HP_REQUIRED_KEYS:
            if k not in self.method.hyperparameters:
                rich.print(f"Error: {k} is required for key 'hyperparameters' in 'method'.")
                error = True

        for k in CLIENT_HP_REQUIRED_KEYS:
            if k not in self.method.hyperparameters.client:
                rich.print(f"Error: {k} is required as hyperparameter of 'client'.")
                error = True

        if 'logger' in self and self.logger.name == "wandb":
            for k in WANDB_REQUIRED_KEYS:
                if k not in WANDB_REQUIRED_KEYS:
                    rich.print(f"Error: {k} is required for key 'logger' when using 'WandBLog'.")
                    error = True

        # if not error:
        #     self.data.dataset.name = DatasetsEnum(self.data.dataset.name)
            # self.exp.device = DeviceEnum(self.exp.device) if self.exp.device else DeviceEnum.CPU
            # self.logger.name = LogEnum(self.logger.name)

        if error:
            raise ValueError("Configuration validation failed.")

    def __str__(self) -> str:
        return f"{self.method.name}_data({self.data.dataset.name}, " + \
            f"{self.data.distribution.name}(" + \
            ", ".join([f"{k}={v}" for k, v in self.data.distribution.exclude('name').items()]) +\
            f"))_proto(C{self.protocol.n_clients}, R{self.protocol.n_rounds}, " + \
            f"E{self.protocol.eligible_perc})" + \
            f"_seed({self.exp.seed})"

    def __repr__(self) -> str:
        return str(self)


def plot_distribution(clients: list[Client],
                      train: bool = True,
                      type: str = "ball") -> None:
    """Plot the distribution of classes for each client.
    This function is used to plot the distribution of classes for each client. The plot can be a
    scatter plot, a heatmap, or a bar plot. The scatter plot (``type='ball'``) shows filled circles
    whose size is proportional to the number of examples of a class. The heatmap (``type='mat'``)
    shows a matrix where the rows represent the classes and the columns represent the clients with
    a color intensity proportional to the number of examples of a class. The bar plot
    (``type='bar'``) shows a stacked bar plot where the height of the bars is proportional to the
    number of examples of a class.

    Warning:
        If the number of clients is greater than 30, the type is automatically switched to
        ``'bar'`` for better visualization.

    Args:
        clients (list[Client]): The list of clients.
        train (bool, optional): Whether to plot the distribution on the training set. If ``False``,
            the distribution is plotted on the test set. Defaults to ``True``.
        type (str, optional): The type of plot. It can be ``'ball'``, ``'mat'``, or ``'bar'``.
            Defaults to ``'ball'``.
    """
    assert type in ["bar", "ball", "mat"], "Invalid plot type. Must be 'bar', 'ball' or 'mat'."
    if len(clients) > 30 and type != "bar":
        warnings.warn("Too many clients to plot. Switching to 'bar' plot.")
        type = "bar"

    client = {}
    for c in clients:
        client[c.index] = c.train_set.tensors[1] if train else c.test_set.tensors[1]

    # Count the occurrences of each class for each client
    class_counts = {client_idx: torch.bincount(client_data).tolist()
                    for client_idx, client_data in enumerate(client.values())}

    # Maximum number of classes
    num_classes = max(len(counts) for counts in class_counts.values())

    _, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(range(len(client)))
    ax.set_yticks(range(num_classes))

    class_matrix = np.zeros((num_classes, len(client)))
    for client_idx, counts in class_counts.items():
        for class_idx, count in enumerate(counts):
            class_matrix[class_idx, client_idx] = count
            # Adjusting size based on the count
            if type == "ball":
                size = count * 1  # Adjust the scaling factor as needed
                ax.scatter(client_idx, class_idx, s=size, alpha=0.6)
                ax.text(client_idx, class_idx, str(count), va='center',
                        ha='center', color='black', fontsize=9)
    plt.title('Number of Examples per Class for Each Client', fontsize=12)
    ax.grid(False)
    if type == "mat":
        sns.heatmap(class_matrix, ax=ax, cmap="viridis", annot=class_matrix, fmt='g',
                    cbar=False, annot_kws={"fontsize": 6})
    elif type == "bar":
        df = pd.DataFrame(class_matrix.T, index=[f'{i}' for i in range(len(client))],
                          columns=[f'{i}' for i in range(num_classes)])
        df.plot(ax=ax, kind='bar', stacked=True, color=sns.color_palette('viridis', num_classes))
    plt.xlabel('clients')
    plt.ylabel('classes')
    plt.show()
