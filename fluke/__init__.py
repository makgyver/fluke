"""
The ``fluke`` module is the entry module of the ``fluke`` framework. Here are defined generic
classes used by the other modules.
"""
from __future__ import annotations

import random
import os
import re
import shutil
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Iterable, Union

import numpy as np
import torch
from rich.console import Group
from rich.progress import Live, Progress

if TYPE_CHECKING:
    from .evaluation import Evaluator


__all__ = [
    'algorithms',
    'client',
    'comm',
    'data',
    'evaluation',
    'get',
    'nets',
    'run',
    'server',
    'utils',
    'DDict',
    'GlobalSettings',
    'ObserverSubject',
    'Singleton'
]


class Singleton(type):
    """This metaclass is used to create singleton classes. A singleton class is a class that can
    have only one instance. If the instance does not exist, it is created; otherwise, the existing
    instance is returned.

    Example:
        .. code-block:: python
            :linenos:

            class MyClass(metaclass=Singleton):
                pass
            a = MyClass()
            b = MyClass()
            print(a is b)  # True

    """
    _instances = {}

    def __call__(cls, *args, **kwargs: dict[str, Any]):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DDict(dict):
    """A dictionary that can be accessed with dot notation recursively.

    Example:
        .. code-block:: python
            :linenos:

            d = DDict(a=1, b=2, c={'d': 3, 'e': 4})
            print(d.a)  # 1
            print(d.b)  # 2
            print(d.c.d)  # 3
            print(d.c.e)  # 4

    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, *args: dict, **kwargs: dict[str, Any]):
        self.update(*args, **kwargs)

    def update(self, *args: dict, **kwargs: dict[str, Any]):
        """Update the ``DDict`` with the specified key-value pairs.

        Args:
            *args (dict): Dictionary with the key-value pairs.
            **kwargs: The key-value pairs.

        Example:
            .. code-block:: python
                :linenos:

                d = DDict(a=1)
                print(d) # {'a': 1}
                d.update(b=2, c=3)
                print(d) # {'a': 1, 'b': 2, 'c': 3}

        """
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = DDict(**v)
                    else:
                        self[k] = v
            else:
                warnings.warn(f"Argument {arg} is not a dictionary and will be ignored.")

        for k, v in kwargs.items():
            if isinstance(v, dict):
                self[k] = DDict(**v)
            else:
                self[k] = v

    def exclude(self, *keys: str):
        """Create a new ``DDict`` excluding the specified keys.

        Args:
            *keys: The keys to be excluded.

        Returns:
            DDict: The new DDict.

        Example:
            .. code-block:: python
                :linenos:

                d = DDict(a=1, b=2, c=3)
                e = d.exclude('b', 'c')
                print(e) # {'a': 1}

        """
        return DDict(**{k: v for k, v in self.items() if k not in keys})

    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


class ObserverSubject():
    """Subject class for the observer pattern. The subject is the class that is observed and thus
    it holds the observers.

    Example:
        .. code-block:: python
            :linenos:

            class MySubject(ObserverSubject):
                def __init__(self):
                    super().__init__()
                    self._data = 0

                @property
                def data(self):
                    return self._data

                @data.setter
                def data(self, value):
                    self._data = value
                    self.notify()

            class MyObserver:
                def __init__(self, subject):
                    subject.attach(self)

                def update(self):
                    print("Data changed.")

            subject = MySubject()
            observer = MyObserver(subject)
            subject.data = 1  # "Data changed."

    """

    def __init__(self):
        self._observers: list[Any] = []

    def attach(self, observer: Union[Any, Iterable[Any]]):
        """Attach one or more observers.

        Args:
            observer (Union[Any, Iterable[Any]]): The observer or a list of observers.
        """
        if observer is None:
            return

        if not isinstance(observer, (list, tuple, set)):
            observer = [observer]

        for obs in observer:
            if obs not in self._observers:
                self._observers.append(obs)

    def detach(self, observer: Any):
        """Detach an observer.

        Args:
            observer (Any): The observer to be detached.
        """
        try:
            self._observers.remove(observer)
        except ValueError:
            pass


class GlobalSettings(metaclass=Singleton):
    """Global settings for ``fluke``.
    This class is a singleton that holds the global settings for ``fluke``. The settings include:

    - The device (``"cpu"``, ``"cuda[:N]"``, ``"auto"``, ``"mps"``);
    - The ``seed`` for reproducibility;
    - If the models are stored in memory or on disk (when not in use);
    - The evaluation configuration;
    - The saving settings;
    - The progress bars for the federated learning process, clients and the server;
    - The live renderer, which is used to render the progress bars.

    """

    # general settings
    _device: torch.device = torch.device('cpu')
    _seed: int = 0
    _inmemory: bool = True

    # saving settings
    _save_path: str = None
    _save_every: int = 0
    _global_only: bool = False
    _temp_path: str = None

    # evaluation settings
    _evaluator: Evaluator = None
    _eval_cfg: dict = {
        "pre_fit": False,
        "post_fit": False,
        "locals": False,
        "server": True
    }

    # progress bars
    _rich_progress_FL: Progress = None
    _rich_progress_clients: Progress = None
    _rich_progress_server: Progress = None
    _live_renderer: Live = None

    def __init__(self):
        super().__init__()
        self._rich_progress_FL: Progress = Progress(transient=True)
        self._rich_progress_clients: Progress = Progress(transient=True)
        self._rich_progress_server: Progress = Progress(transient=True)
        self._rich_live_renderer: Live = Live(Group(self._rich_progress_FL,
                                                    self._rich_progress_clients,
                                                    self._rich_progress_server))

    def get_seed(self) -> int:
        """Get the seed.

        Returns:
            int: The seed.
        """
        return self._seed

    def get_eval_cfg(self) -> DDict:
        """Get the evaluation configuration.

        Returns:
            DDict: The evaluation configuration.
        """
        return DDict(self._eval_cfg)

    def set_eval_cfg(self, cfg: DDict) -> None:
        """Set the evaluation configuration.

        Args:
            cfg (DDict): The evaluation configuration.
        """
        for key, value in cfg.items():
            self._eval_cfg[key] = value

    def get_evaluator(self) -> Evaluator:
        """Get the evaluator.

        Returns:
            Evaluator: The evaluator.
        """
        return self._evaluator

    def set_evaluator(self, evaluator: Evaluator) -> None:
        """Set the evaluator.

        Args:
            evaluator (Evaluator): The evaluator.
        """
        self._evaluator = evaluator

    def set_seed(self, seed: int) -> None:
        """Set seed for reproducibility. The seed is used to set the random seed for the following
        libraries: ``torch``, ``torch.cuda``, ``numpy``, ``random``.

        Args:
            seed (int): The seed.
        """
        self._seed = seed
        torch.manual_seed(seed)
        gen = torch.Generator()
        gen.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def auto_device(self) -> torch.device:
        """Set device to ``cuda`` or ``mps`` if available, otherwise ``cpu``.

        Returns:
            torch.device: The device.
        """
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self._device = torch.device('mps')
        else:
            self._device = torch.device('cpu')
        return self._device

    def set_device(self, device: str) -> torch.device:
        """Set the device. The device can be ``cpu``, ``auto``, ``mps``, ``cuda`` or ``cuda:N``,
        where ``N`` is the GPU index.

        Args:
            device (str): The device as string.

        Returns:
            torch.device: The selected device as torch.device.
        """
        assert device in ['cpu', 'auto', 'mps', 'cuda'] or re.match(r'^cuda:\d+$', device), \
            f"Invalid device {device}."

        if device == "auto":
            return GlobalSettings().auto_device()

        if device.startswith('cuda') and ":" in device:
            idx = int(device.split(":")[1])
            self._device = torch.device("cuda", idx)
        else:
            self._device = torch.device(device)
        return self._device

    def get_device(self) -> torch.device:
        """Get the current device.

        Returns:
            torch.device: The device.
        """
        return self._device

    def get_progress_bar(self, progress_type: str) -> Progress:
        """Get the progress bar.
        The possible progress bar types are:

        - ``FL``: The progress bar for the federated learning process.
        - ``clients``: The progress bar for the clients.
        - ``server``: The progress bar for the server.

        Args:
            progress_type (str): The type of progress bar.

        Returns:
            Progress: The progress bar.

        Raises:
            ValueError: If the progress bar type is invalid.
        """
        if progress_type == 'FL':
            return self._rich_progress_FL
        elif progress_type == 'clients':
            return self._rich_progress_clients
        elif progress_type == 'server':
            return self._rich_progress_server
        else:
            raise ValueError(f'Invalid type of progress bar type {progress_type}.')

    def get_live_renderer(self) -> Live:
        """Get the live renderer. The live renderer is used to render the progress bars.

        Returns:
            Live: The live renderer.
        """
        return self._rich_live_renderer

    def get_save_options(self) -> tuple[str, int, bool]:
        """Get the save options.

        Returns:
            tuple[str, int, bool]: The save path, the save frequency and the global only flag.
        """
        return self._save_path, self._save_every, self._global_only

    def set_save_options(self,
                         path: str | None = None,
                         save_every: int | None = None,
                         global_only: bool | None = None) -> None:
        """Set the save options.

        Args:
            path (str): The path to save the checkpoints.
            save_every (int): The frequency of saving the checkpoints.
            global_only (bool): If ``True``, only the global model is saved.
        """
        if path is not None:
            self._save_path = path
        if save_every is not None:
            self._save_every = save_every
        if global_only is not None:
            self._global_only = global_only

    def get_temp_path(self) -> str:
        """Get/Generate the temporary path to store data on disk.
        The path has the format ``./tmp/fluke_UUID64`` where ``UUID64`` is a random UUID.

        Returns:
            str: The temporary path.
        """
        if self._temp_path is None:
            self._temp_path = f"./tmp/fluke_{uuid.uuid4()}"

        os.makedirs(self._temp_path, exist_ok=True)
        return self._temp_path

    def empty_temp_path(self) -> None:
        """Empty the temporary folder."""
        if self._temp_path is not None:
            shutil.rmtree(self._temp_path, ignore_errors=True)
            self._temp_path = None

    def set_inmemory(self, inmemory: bool) -> None:
        """Set if the data is stored in memory.

        Args:
            inmemory (bool): If ``True``, the data is stored in memory, otherwise it is stored on
                disk.
        """
        self._inmemory = inmemory

    def is_inmemory(self) -> bool:
        """Check if the data is stored in memory.

        Returns:
            bool: If ``True``, the data is stored in memory, otherwise it is stored on disk.
        """
        return self._inmemory

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_rich')}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
