import re
import torch
import random
import numpy as np
from rich.console import Group
from rich.progress import Progress, Live
from typing import Any, Union, Iterable


__all__ = [
    'comm',
    'algorithms',
    'data',
    'evaluation',
    'net',
    'client',
    'server',
    'utils',
    'run',
    'Singleton',
    'ObserverSubject',
    'GlobalSettings',
    'DDict'
]


class Singleton(type):
    """Singleton metaclass."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DDict(dict):
    """A dictionary that can be accessed with dot notation recursively."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                self[k] = DDict(**v)
            else:
                self[k] = v

    def exclude(self, *keys: str):
        """Create a new DDict excluding the specified keys.

        Returns:
            DDict: The new DDict.
        """
        return DDict(**{k: v for k, v in self.items() if k not in keys})


class ObserverSubject():
    """Observer subject class."""

    def __init__(self):
        self._observers = []

    def attach(self, observer: Union[Any, Iterable[Any]]):
        """Attach an observer.

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


# class DeviceEnum(Enum):
#     """Device enumerator."""
#     CPU: str = "cpu"    #: CPU
#     CUDA: str = "cuda"  #: CUDA
#     AUTO: str = "auto"  #: AUTO - automatically selects CUDA if available, otherwise CPU
#     MPS: str = "mps"    #: MPS - for Apple M1/M2 GPUs


class GlobalSettings(metaclass=Singleton):
    """Global settings for FLUKE.

    This class is a singleton that holds the global settings for FLUKE. The settings include:
    - The device (CPU, CUDA, AUTO, MPS);
    - The seed for reproducibility;
    - The progress bars for the federated learning process, clients and server;
    - The live renderer, which is used to render the progress bars.
    """

    _device: str = 'cpu'
    _seed: int = 0

    _progress_FL: Progress = None
    _progress_clients: Progress = None
    _progress_server: Progress = None
    _live_renderer: Live = None

    def __init__(self):
        super().__init__()
        self._progress_FL = Progress()
        self._progress_clients = Progress()
        self._progress_server = Progress()
        self._live_renderer = Live(Group(self._progress_FL,
                                         self._progress_clients,
                                         self._progress_server))

    def set_seed(self, seed: int) -> None:
        """Set seed for reproducibility.

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
        """Set device to cuda if available, otherwise cpu.

        Returns:
            torch.device: The device.
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def set_device(self, device: str) -> torch.device:
        """Set the device.

        Args:
            device (str): The device as string.

        Returns:
            torch.device: The device as torch.device.
        """
        assert device in ['cpu', 'auto', 'mps', 'cuda'] or re.match(r'^cuda:\d+$', device), \
            f"Invalid device {device}."

        if device == "auto":
            return GlobalSettings().auto_device()

        self._device = torch.device(device)
        return self._device

    def get_device(self):
        """Get the device.

        Returns:
            torch.device: The device.
        """
        return self._device

    def get_progress_bar(self, progress_type: str) -> Progress:
        """Get the progress bar.

        The progress bar types are:
        - FL: The progress bar for the federated learning process.
        - clients: The progress bar for the clients.
        - server: The progress bar for the server.

        Args:
            progress_type (str): The type of progress bar.

        Returns:
            Progress: The progress bar.

        Raises:
            ValueError: If the progress bar type is invalid.
        """
        if progress_type == 'FL':
            return self._progress_FL
        elif progress_type == 'clients':
            return self._progress_clients
        elif progress_type == 'server':
            return self._progress_server
        else:
            raise ValueError(f'Invalid type of progress bar type {progress_type}.')

    def get_live_renderer(self) -> Live:
        """Get the live renderer.

        Returns:
            Live: The live renderer.
        """
        return self._live_renderer

    def get_seed(self):
        """Get the seed.

        Returns:
            int: The seed.
        """
        return self._seed
