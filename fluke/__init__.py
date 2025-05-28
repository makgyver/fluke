"""
The :mod:`fluke` module is the entry module of the :mod:`fluke` framework. Here are defined generic
classes used by the other modules.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import uuid
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Collection, Union

import numpy as np
import torch
from diskcache import Cache
from omegaconf import DictConfig, ListConfig
from rich.console import Group
from rich.progress import Live, Progress

if TYPE_CHECKING:
    from .evaluation import Evaluator


def custom_formatwarning(msg: str, category: type, filename: str, lineno: int, *args) -> str:
    # return f"[{category.__name__}] {filename}:{lineno} - {msg}\n"

    # ANSI color codes
    # red = '\033[91m'
    yellow = "\033[93m"
    blue = "\033[94m"
    reset = "\033[0m"

    return (
        f"{yellow}[{category.__name__}]{reset} "
        f"{blue}{filename}:{lineno}{reset}\n"
        f"{yellow}{msg}{reset}\n"
    )


warnings.formatwarning = custom_formatwarning


__all__ = [
    "algorithms",
    "client",
    "comm",
    "config",
    "data",
    "distr",
    "evaluation",
    "get",
    "nets",
    "run",
    "server",
    "utils",
    "DDict",
    "FlukeCache",
    "FlukeENV",
    "ObserverSubject",
    "Singleton",
]

__version__ = "0.7.9"
__author__ = "Mirko Polato"
__email__ = "mirko.polato@unito.it"
__license__ = "LGPLv2.1"
__copyright__ = "Copyright (c) 2025, Mirko Polato"
__status__ = "Development"


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

    def __call__(cls, *args, **kwargs) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def clear(cls) -> None:
        """Clear the singleton instances."""
        cls._instances = {}


class DDict(dict):
    """A dictionary that can be accessed with dot notation recursively.

    Important:
        The :class:`DDict` is a subclass of the built-in :class:`dict` class and it behaves like a
        dictionary. However, the keys must be strings.

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

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs) -> None:
        """Update the :class:`DDict` with the specified key-value pairs.

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
                    if isinstance(v, (dict, DictConfig)):
                        self[k] = DDict(**v)
                    elif isinstance(v, ListConfig):
                        self[k] = list(v)
                    else:
                        self[k] = v
            else:
                warnings.warn(f"Argument {arg} is not a dictionary and will be ignored.")

        for k, v in kwargs.items():
            if isinstance(v, (dict, DictConfig)):
                self[k] = DDict(**v)
            elif isinstance(v, ListConfig):
                self[k] = list(v)
            else:
                self[k] = v

    def exclude(self, *keys: str) -> DDict:
        """Create a new :class:`DDict` excluding the specified keys.

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

    def match(self, other: DDict, full: bool = True) -> bool:
        """Check if the two :class:`DDict` match.

        Args:
            other (DDict): The other :class:`DDict`.
            full (bool): If ``True``, the two :class:`DDict` must match exactly. If ``False``, the
                `other` :class:`DDict` must be a subset of the current :class:`DDict`.

        Returns:
            bool: Whether the two :class:`DDict` match.
        """
        if full:
            return self == other
        return all(
            k in self
            and (
                self[k] == other[k]
                if not isinstance(self[k], DDict)
                else self[k].match(other[k], False)
            )
            for k in other.keys()
        )

    def diff(self, other: DDict) -> DDict:
        """Get the difference between two :class:`DDict`.

        Args:
            other (DDict): The other :class:`DDict`.

        Returns:
            DDict: The difference between the two :class:`DDict`.

        Example:
            .. code-block:: python
                :linenos:

                d = DDict(a=1, b=2, c=3)
                e = DDict(a=1, b=3, c=4)
                print(d.diff(e)) # {'b': 3, 'c': 4}
        """
        diff = DDict()
        for k, v in other.items():
            if k in self:
                if isinstance(self[k], DDict):
                    d = self[k].diff(v)
                    if d:
                        diff[k] = d
                elif v != self[k]:
                    diff[k] = v
            else:
                diff[k] = v
        return diff

    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def hash(self) -> str:
        """Returns a SHA-256 hash of the dictionary contents.

        This is useful to check if the dictionary has changed or not, for example, when
        comparing configurations or parameters.

        Returns:
            str: The SHA-256 hash of the dictionary contents.
        """
        dict_str = json.dumps(self, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dict_str.encode("utf-8")).hexdigest()


class ObserverSubject:
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

    def attach(self, observer: Union[Any, Collection[Any]]):
        """Attach one or more observers.

        Args:
            observer (Union[Any, Collection[Any]]): The observer or a list of observers.
        """
        if observer is None:
            return

        if not isinstance(observer, (list, tuple, set)):
            observer = [observer]

        for obs in observer:
            if obs not in self._observers:
                self._observers.append(obs)

    def detach(self, observer: Any) -> None:
        """Detach an observer.

        Args:
            observer (Any): The observer to be detached.
        """
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    def notify(self, event: str, **kwargs) -> None:
        for obs in self._observers:
            if hasattr(obs, event):
                getattr(obs, event)(**kwargs)


class FlukeENV(metaclass=Singleton):
    """Environment class for the :mod:`fluke` framework.
    This class is a singleton and it contains environment settings that are used by the other
    classes. The environment includes:

    - The device (``"cpu"``, ``"cuda[:N]"``, ``"auto"``, ``"mps"``);
    - The ``seed`` for reproducibility;
    - If the models are stored in memory or on disk (when not in use);
    - The evaluation configuration;
    - The saving settings;
    - The progress bars for the federated learning process, clients and the server;
    - The live renderer, which is used to render the progress bars.

    """

    # general settings
    _device: torch.device = torch.device("cpu")
    _device_ids: list[int] = []
    _seed: int = 0
    _inmemory: bool = True
    _cache: FlukeCache | None = None

    # saving settings
    _save_path: str = None
    _save_every: int = 0
    _global_only: bool = False

    # evaluation settings
    _evaluator: Evaluator = None
    _eval_cfg: dict = {
        "pre_fit": False,
        "post_fit": False,
        "locals": False,
        "server": True,
    }

    # progress bars
    _rich_progress_FL: Progress = None
    _rich_progress_clients: Progress = None
    _rich_progress_server: Progress = None
    _live_renderer: Live = None

    # global logger
    _logger: Any = None

    def __init__(self):
        super().__init__()
        self._rich_progress_FL: Progress = Progress(transient=True)
        self._rich_progress_clients: Progress = Progress(transient=True)
        self._rich_progress_server: Progress = Progress(transient=True)
        self._rich_live_renderer: Live = Live(
            Group(
                self._rich_progress_FL,
                self._rich_progress_clients,
                self._rich_progress_server,
            )
        )

    def configure(self, cfg: DDict) -> None:
        """Configure the global settings.

        Args:
            cfg (DDict): The configuration.
        """
        self.set_seed(cfg.exp.seed)
        self.set_device(cfg.exp.device)
        self.set_inmemory(cfg.exp.inmemory)
        self.set_save_options(**cfg.save)
        self.set_eval_cfg(**cfg.eval)

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

    def set_eval_cfg(self, **cfg: DDict | dict) -> None:
        """Set the evaluation configuration.

        Args:
            **cfg (DDict | dict): The evaluation configuration.
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
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        gen = torch.Generator()
        gen.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def auto_device(self) -> torch.device:
        """Set device to ``cuda`` or ``mps`` if available, otherwise ``cpu``.

        Returns:
            torch.device: The device.
        """
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        return self._device

    def set_device(self, device: str | list) -> torch.device:
        """Set the device. The device can be ``cpu``, ``auto``, ``mps``, ``cuda`` or ``cuda:N``,
        where ``N`` is the GPU index.

        Args:
            device (str): The device as string.

        Returns:
            torch.device: The selected device as torch.device.
        """
        assert (
            device in ["cpu", "auto", "mps", "cuda"]
            or isinstance(device, list)
            or re.match(r"^cuda:\d+$", device)
        ), f"Invalid device {device}."

        if device == "auto":
            return FlukeENV().auto_device()

        if isinstance(device, list):
            self._device_ids = []
            for d in device:
                if isinstance(d, int):
                    self._device_ids.append(d)
                elif re.match(r"^cuda:\d+$", d):
                    self._device_ids.append(int(d.split(":")[1]))
                else:
                    raise ValueError(f"Invalid device/device_id {d}.")

            self._device = torch.device(device[0])
            if len(self._device_ids) > 1:
                warn_msg = (
                    "[EXPERIMENTAL Feature] Multi-GPU training is experimental"
                    " and may not work as expected. Please report any issues to the developers."
                )
                warnings.warn(warn_msg)

        elif device.startswith("cuda") and ":" in device:
            self._device_ids = []
            idx = int(device.split(":")[1])
            self._device = torch.device("cuda", idx)
            self._device_ids.append(idx)

        else:
            self._device = torch.device(device)
        return self._device

    def get_device(self) -> torch.device:
        """Get the current device.

        Returns:
            torch.device: The device.
        """
        return self._device

    def get_device_ids(self) -> list[int]:
        """Get the device ids if the device is ``cuda``.

        Note:
            The device ids are the indices of the GPUs that are used by the client. If the device is
            ``cpu`` or ``mps``, an empty list is returned.

        Returns:
            list[int]: The device ids.
        """
        return self._device_ids

    def is_parallel_client(self):
        """
        Check if the client runs in parallel mode, i.e., using multiple GPUs.

        Returns:
            bool: True if the client runs in parallel mode, False otherwise.
        """
        return len(self._device_ids) > 1

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
        if progress_type == "FL":
            return self._rich_progress_FL
        elif progress_type == "clients":
            return self._rich_progress_clients
        elif progress_type == "server":
            return self._rich_progress_server
        else:
            raise ValueError(f"Invalid type of progress bar type {progress_type}.")

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

    def set_save_options(
        self,
        path: str | None = None,
        save_every: int | None = None,
        global_only: bool | None = None,
    ) -> None:
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

    def get_logger(self) -> Any:
        """Get the global logger.

        Returns:
            Any: The logger.
        """
        return self._logger

    def set_logger(self, logger: Any) -> None:
        """Set the global logger.

        Args:
            logger (Any): The logger.
        """
        self._logger = logger

    def set_inmemory(self, inmemory: bool) -> None:
        """Set if the data is stored in memory.

        Args:
            inmemory (bool): If ``True``, the data is stored in memory, otherwise it is stored on
                disk.
        """
        self._inmemory = inmemory

    def get_cache(self) -> FlukeCache:
        """Get the cache.

        Returns:
            Cache: The cache.
        """
        return self._cache

    def open_cache(self, path: str) -> None:
        """Open the cache at the specified path if the ``inmemory`` flag is ``False``.

        Note:
            The full path to the cache is ``tmp/path`` where ``path`` is the specified path.
            We suggest to use as path the UUID of the experiment.

        Args:
            path (str): The path to the cache.
        """
        if not self._inmemory and self._cache is None:
            self._cache = FlukeCache(path)
        elif self._cache is not None:
            warnings.warn("Cache already open.")

    def close_cache(self) -> None:
        """Close the cache."""
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def is_inmemory(self) -> bool:
        """Check if the data is stored in memory.

        Returns:
            bool: If ``True``, the data is stored in memory, otherwise it is stored on disk.
        """
        return self._inmemory

    def force_close(self) -> None:
        """Force close the progress bars and the live renderer."""

        task_ids = [task.id for task in self._rich_progress_FL.tasks]
        for tid in task_ids:
            self._rich_progress_FL.remove_task(tid)

        task_ids = [task.id for task in self._rich_progress_clients.tasks]
        for tid in task_ids:
            self._rich_progress_clients.remove_task(tid)

        task_ids = [task.id for task in self._rich_progress_server.tasks]
        for tid in task_ids:
            self._rich_progress_server.remove_task(tid)

        self._rich_live_renderer.refresh()
        self._rich_live_renderer.stop()

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_rich")}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


class FlukeCache:
    """A cache class that can store data on disk."""

    class ObjectRef:
        """A reference to an object in the cache.
        The reference is a unique identifier that is used to store and retrieve the object from the
        cache.
        """

        def __init__(self):
            self._id = str(uuid.uuid4().hex)

        @property
        def id(self) -> str:
            """Get the unique identifier of the reference.

            Returns:
                str: The unique identifier.
            """
            return self._id

        def __str__(self) -> str:
            return f"ObjectRef({self._id})"

        def __repr__(self) -> str:
            return self.__str__()

    class _RefCounter:
        """A reference counter for an object in the cache."""

        def __init__(self, value: Any, refs: int = 1):
            self._value = value
            self._refs = refs
            self._id = FlukeCache.ObjectRef()

        @property
        def id(self) -> FlukeCache.ObjectRef:
            """Get the unique identifier of the reference.

            Returns:
                str: The unique identifier.
            """
            return self._id

        @property
        def value(self) -> Any:
            """Get the value pointed by the reference.

            Returns:
                Any: The value.
            """
            return self._value

        @property
        def refs(self) -> int:
            """Get the number of references to the object in the cache.

            Returns:
                int: The number of references.
            """
            return self._refs

        def dec(self) -> Any:
            """Decrement the number of references to the object in the cache.

            Returns:
                FlukeCache._RefCounter: The reference counter.
            """
            self._refs -= 1
            return self

        def inc(self) -> FlukeCache._RefCounter:
            """Increment the number of references to the object in the cache.

            Returns:
                FlukeCache._RefCounter: The reference counter.
            """
            self._refs += 1
            return self

    def __init__(self, path: str, **kwargs):
        if "size_limit" not in kwargs:
            kwargs["size_limit"] = 2**34
        self._cache: Cache = Cache(f"tmp/{path}", **kwargs)
        self._key2ref: dict[str, FlukeCache.ObjectRef] = {}

    def __getitem__(self, key: str) -> Any:
        return self._cache[self._key2ref[key].id].value

    @property
    def cache_dir(self) -> str:
        """Get the cache directory.

        Returns:
            str: The cache directory.
        """
        return self._cache.directory

    def get(self, key: str, default: Any = None):
        """Get the object identified by the key from the cache.

        If the object is not in the cache, the default value is returned.

        Note:
            The object is still in the cache after this operation.

        Args:
            key (str): The key of the object.
            default (Any, optional): The default value to return if the object is not in the cache.
                Defaults to None.

        Returns:
            Any: The object in the cache or the default value.
        """
        if key not in self._key2ref:
            return default
        obj = self._cache.get(self._key2ref[key].id, default=default)
        if obj is not default:
            return obj.value
        return default

    def push(self, key: str, value: Any) -> FlukeCache.ObjectRef:
        """Push an object to the cache.

        Note:
            If the object that is pushed is already a cache reference, then the referenced object is
            already in the cache and its reference counter is incremented.

        Args:
            key (str): The key of the object.
            value (Any): The object to store in the cache.

        Returns:
            FlukeCache.ObjectRef: The reference to the object in the cache.
        """
        if isinstance(value, FlukeCache.ObjectRef):
            assert value.id in self._cache, f"Reference {value.id} not in cache."
            self._key2ref[key] = value
            self._cache[value.id] = self._cache[value.id].inc()
            return value
        else:
            ref = self._RefCounter(value)
            self._key2ref[key] = ref.id
            self._cache[ref.id.id] = ref
            return ref.id

    def pop(self, key: str, copy: bool = True) -> Any:
        """Pop an object from the cache given its key.

        If the key is not in the cache, ``None`` is returned.

        Args:
            key (str): The key of the object.
            copy (bool, optional): If ``True``, a copy of the object is returned.
                Defaults to ``True``.

        Returns:
            Any: The object in the cache or its copy.
        """
        if key not in self._key2ref:
            return None
        ref = self._key2ref[key]
        del self._key2ref[key]
        self._cache[ref.id] = self._cache[ref.id].dec()
        obj = self._cache[ref.id].value
        if self._cache[ref.id].refs == 0:
            self._cache.delete(ref.id)
        return obj if not copy else deepcopy(obj)

    def delete(self, key: str) -> None:
        """Remove an object from the cache without returning it.

        If the key is not in the cache, nothing happens.

        Args:
            key (str): The key of the object.
        """
        if key in self._key2ref:
            ref = self._key2ref[key]
            del self._key2ref[key]
            self._cache[ref.id] = self._cache[ref.id].dec()
            if self._cache[ref.id].refs == 0:
                self._cache.delete(ref.id)

    def close(self) -> None:
        """Close the cache."""
        if self._cache is not None:
            self._cache.clear()
            self._cache.close()
            try:
                shutil.rmtree(self._cache.directory)
            except OSError:  # Windows wonkiness
                pass
            self._cache = None
            self._key2ref = {}

    @property
    def occupied(self) -> int:
        """Get the number of different objects in the cache.

        Returns:
            int: The number of objects in the cache.
        """
        return len(list(self._cache.iterkeys()))

    def cleanup(self) -> None:
        """Clean up the cache by removing the objects that are not referenced.

        This operation should not be necessary if the cache is used correctly.
        """
        keys = set([v.id for v in self._key2ref.values()])
        for key in self._cache.iterkeys():
            if key not in keys:
                self._cache.pop(key)
