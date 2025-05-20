"""This module contains utility functions and classes used in ``fluke``."""
from __future__ import annotations

import importlib
import os
import sys
import warnings
from typing import TYPE_CHECKING, Any, Collection, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

sys.path.append(".")
sys.path.append("..")

if TYPE_CHECKING:
    from client import Client  # NOQA
    from server import Server  # NOQA

from .. import FlukeCache, FlukeENV, custom_formatwarning  # NOQA

warnings.formatwarning = custom_formatwarning

__all__ = [
    'log',
    'model',
    'ClientObserver',
    'ServerObserver',
    'bytes2human',
    'cache_obj',
    'clear_cuda_cache',
    'get_class_from_str',
    'get_class_from_qualified_name',
    'get_full_classname',
    'get_loss',
    'get_model',
    'get_optimizer',
    'get_scheduler',
    'flatten_dict',
    'import_module_from_str',
    'memory_usage',
    'plot_distribution',
    'retrieve_obj',
    'safe_train_test_split'
]


class ClientObserver:
    """Client observer interface.
    This interface is used to observe the client during the federated learning process.
    For example, it can be used to log the performance of the local model, as it is done by the
    :class:`fluke.utils.log.Log` class.
    """

    def start_fit(self, round: int, client_id: int, model: Module, **kwargs):
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
                **kwargs):
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
                          **kwargs):
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

    def track_item(self,
                   round: int,
                   client_id: int,
                   item: str,
                   value: float,
                   **kwargs) -> None:
        """This method is called when the client aims to log an item.

        Args:
            round (int): The round number.
            client_id (int): The client ID.
            item (str): The name of the log item.
            value (float): The value of the log item.
            **kwargs (dict): Additional keyword arguments.
        """
        pass


class ServerObserver:
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

    def selected_clients(self, round: int, clients: Collection) -> None:
        """This method is called when the clients have been selected for the current round.

        Args:
            round (int): The round number.
            clients (Collection): The clients selected for the current round.
        """
        pass

    def server_evaluation(self,
                          round: int,
                          eval_type: Literal["global", "locals"],
                          evals: Union[dict[str, float], dict[int, dict[str, float]]],
                          **kwargs) -> None:
        """This method is called when the server evaluates the global or the local models on its
        test set.

        Args:
            round (int): The round number.
            eval_type (Literal['global', 'locals']): The type of evaluation. If 'global', the
                evaluation is done on the global model. If 'locals', the evaluation is done on the
                local models of the clients on the test set of the server.
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

    def interrupted(self) -> None:
        """This method is called when the federated learning process has been interrupted."""
        pass

    def early_stop(self, round: int) -> None:
        """This method is called when the federated learning process has been stopped due to an
        early stopping criterion.

        Args:
            round (int): The last round number.
        """
        pass

    def track_item(self,
                   round: int,
                   item: str,
                   value: float) -> None:
        """This method is called when the server aims log an item.

        Args:
            round (int): The round number.
            item (str): The name of the log item.
            value (float): The value of the log item.
        """
        pass


def safe_train_test_split(X: torch.Tensor,
                          y: torch.Tensor,
                          test_size: float,
                          client_id: int | None = None) -> tuple[torch.Tensor,
                                                                 Optional[torch.Tensor],
                                                                 torch.Tensor,
                                                                 Optional[torch.Tensor]]:
    try:
        if test_size == 0.0:
            return X, None, y, None
        else:
            return train_test_split(X, y, test_size=test_size, stratify=y)
    except ValueError:
        client_str = f"[Client {client_id}]" if client_id is not None else ""
        warnings.warn(
            f"Stratified split failed for {client_str}. Falling back to random split."
        )
        return train_test_split(X, y, test_size=test_size)


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


def get_model(mname: str, **kwargs) -> Module:
    """Get a model from its name.
    This function is used to get a torch model from its name and the name of the module where it is
    defined. It is used to dynamically import models. If ``mname`` is not a fully qualified name,
    the model is assumed to be defined in the :mod:`fluke.nets` module.

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
    optimizers. The supported optimizers are the ones defined in the :mod:`torch.optim` module.

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
    :mod:`torch.optim.lr_scheduler` module.

    Args:
        sname (str): The name of the scheduler.

    Returns:
        torch.nn.Module: The learning rate scheduler.
    """
    return get_class_from_str("torch.optim.lr_scheduler", sname)


def clear_cuda_cache(ipc: bool = False):
    """Clear the CUDA cache. This function should be used to free the GPU memory after the training
    process has ended. It is usually used after the local training of the clients.

    Args:
        ipc (bool, optional): Whether to force collecting GPU memory after it has been released by
            CUDA IPC.
    """
    torch.cuda.empty_cache()
    if ipc and torch.cuda.is_available():
        torch.cuda.ipc_collect()


def plot_distribution(clients: list[Client],
                      train: bool = True,
                      plot_type: str = "ball") -> None:
    """Plot the distribution of classes for each client.
    This function is used to plot the distribution of classes for each client. The plot can be a
    scatter plot, a heatmap, or a bar plot. The scatter plot (``plot_type='ball'``) shows filled
    circles whose size is proportional to the number of examples of a class. The heatmap
    (``plot_type='mat'``) shows a matrix where the rows represent the classes and the columns
    represent the clients with a color intensity proportional to the number of examples of a class.
    The bar plot (``plot_type='bar'``) shows a stacked bar plot where the height of the bars is
    proportional to the number of examples of a class.

    Warning:
        If the number of clients is greater than 30, the type is automatically switched to
        ``'bar'`` for better visualization.

    Args:
        clients (list[Client]): The list of clients.
        train (bool, optional): Whether to plot the distribution on the training set. If ``False``,
            the distribution is plotted on the test set. Defaults to ``True``.
        plot_type (str, optional): The type of plot. It can be ``'ball'``, ``'mat'``, or ``'bar'``.
            Defaults to ``'ball'``.
    """
    assert plot_type in ["bar", "ball", "mat"], "Invalid plot type. Must be 'bar', 'ball' or 'mat'."
    if len(clients) > 30 and plot_type != "bar":
        warnings.warn("Too many clients to plot. Switching to 'bar' plot.")
        plot_type = "bar"

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

    class_matrix = np.zeros((num_classes, len(client)))
    for client_idx, counts in class_counts.items():
        for class_idx, count in enumerate(counts):
            class_matrix[class_idx, client_idx] = count
            # Adjusting size based on the count
            if plot_type == "ball":
                size = count * 1  # Adjust the scaling factor as needed
                ax.scatter(client_idx, class_idx, s=size, alpha=0.6)
                ax.set_yticks(range(num_classes))
                ax.text(client_idx, class_idx, str(count), va='center',
                        ha='center', color='black', fontsize=9)
    plt.title('Number of Examples per Class for Each Client', fontsize=12)
    ax.grid(False)
    if plot_type == "mat":
        ax.set_yticks(range(num_classes))
        sns.heatmap(class_matrix, ax=ax, cmap="viridis", annot=class_matrix, fmt='g',
                    cbar=False, annot_kws={"fontsize": 6})
    elif plot_type == "bar":
        df = pd.DataFrame(class_matrix.T, index=[f'{i}' for i in range(len(client))],
                          columns=[f'{i}' for i in range(num_classes)])
        step = int(max(1, np.ceil(len(clients) / 50)))
        df.plot(ax=ax,
                kind='bar',
                stacked=True,
                xticks=[x for x in range(0, len(clients), step)],
                color=sns.color_palette('viridis', num_classes)).legend(loc='upper right')
    plt.xlabel('clients')
    plt.ylabel('classes')
    plt.show()


def bytes2human(n: int) -> str:
    """Convert bytes to human-readable format.

    See:
        https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

    Args:
        n (int): The number of bytes.

    Returns:
        str: The number of bytes in human-readable format.

    Example:
        .. code-block:: python
            :linenos:

            >>> bytes2human(10000)
            '9.8K'
            >>> bytes2human(100001221)
            '95.4M'
            >>> bytes2human(1024 ** 3)
            '1.0G'

    """
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f %sB' % (value, s)
    return "%s B" % n


def memory_usage() -> tuple[int, int, int]:
    """Get the memory usage of the current process.

    Returns:
        tuple[int, int, int]: The resident set size (RSS), the virtual memory size (VMS), and the
            current CUDA reserved memory. If the device is not a CUDA device, the reserved memory is
            0.
    """
    proc = psutil.Process(os.getpid())

    if FlukeENV().get_device().type == "cuda":
        cuda_device = torch.device(FlukeENV().get_device())
        current_reserved = torch.cuda.memory_reserved(cuda_device)
    else:
        current_reserved = 0
    return proc.memory_info().rss, proc.memory_info().vms, current_reserved


def retrieve_obj(key: str,
                 party: Client | Server | None = None,
                 pop: bool = True) -> Any:
    """Load an object from the cache (disk).
    If the object is not found in the cache, it returns ``None``.

    Warning:
        This method assumes the cache is already initialized and opened.
        If it is not, it will raise an exception.

    Args:
        key (str, optional): specific key to add to the default filename.
        party (Client | Server, optional): the party for which to load the object. Defaults to
            ``None``.
        pop (bool, optional): Whether to remove the object from the cache after loading it.
            Defaults to ``True``.

    Returns:
        Any: The object retrieved from the cache.
    """
    prefix = ""
    if party is not None:
        prefix = f"c{party.index}_" if hasattr(party, "index") else "server_"
    cache = FlukeENV().get_cache()
    obj = cache.pop(f"{prefix}{key}", copy=(party is None)) if pop else cache.get(f"{prefix}{key}")
    return obj


def cache_obj(obj: Any,
              key: str,
              party: Server | Client | None = None) -> FlukeCache.ObjectRef | None:
    """Move the object from the RAM to the disk cache to free up memory.
    If the object is ``None``, it returns ``None`` without caching it.

    Warning:
        This method assumes the cache is already initialized and opened.
        If it is not, it will raise an exception.

    Args:
        obj (Any): The object to cache.
        key (str, optional): The key associated with the object.
        party (Client | Server): the party for which to unload the model. Defaults to ``None``.

    Returns:
        FlukeCache.ObjectRef: The object reference identifier. If the object is ``None``,
        it returns ``None``.
    """
    if obj is None:
        return None
    prefix = ""
    if party is not None:
        prefix = f"c{party.index}_" if hasattr(party, "index") else "server_"
    cache = FlukeENV().get_cache()
    return cache.push(f"{prefix}{key}", obj)


def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_dict(nested_dict: dict, sep: str = '.') -> dict:
    """Flatten a nested dictionary.
    The flatten dictionary is a dictionary where the keys are the concatenation of the keys of the
    nested dictionary separated by a separator.

    Args:
        nested_dict (dict): Nested dictionary.
        sep (str, optional): Separator. Defaults to '.'.

    Returns:
        dict: Flattened dictionary.

    Example:
        .. code-block:: python
            :linenos:

            d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
            flatten_dict(d)
            # Output: {'a': 1, 'b.c': 2, 'b.d.e': 3}

    """
    return _flatten_dict(nested_dict, sep=sep)
