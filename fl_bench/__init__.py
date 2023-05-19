from abc import ABC, abstractmethod
import sys
import pickle
from typing import Any, Optional, Tuple, Union, Iterable
import torch
import multiprocessing as mp
from rich.progress import Progress, Live
from rich.console import Group

class Singleton(type):
    """Singleton metaclass."""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ObserverSubject():
    """Observer subject class."""
    def __init__(self):
        self._observers = []

    def attach(self, observer: Union[Any, Iterable[Any]]):
        if observer is None:
            return
        
        if not isinstance(observer, Iterable):
            observer = [observer]

        for observer in observer:
            if not observer in self._observers:
                self._observers.append(observer)

    def detach(self, observer: Any):
        try:
            self._observers.remove(observer)
        except ValueError:
            pass


class GlobalSettings(metaclass=Singleton):
    """Global settings for the library.""" 
    
    _device = 'cpu'
    _workers = 1

    _progress_FL = None
    _progress_clients = None
    _progress_server = None
    _live_renderer = None

    def __init__(self):
        super().__init__()
        self._progress_FL = Progress()
        self._progress_clients = Progress()
        self._progress_server = Progress()
        self._live_renderer = Live(Group(self._progress_FL,
                                         self._progress_clients,
                                         self._progress_server))

    def auto_device(self) -> torch.device:
        """Set device to cuda if available, otherwise cpu.
        
        Returns
        -------
        torch.device
            The device.
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    def set_device(self, device_name: str) -> torch.device:
        """Set the device.
        
        Parameters
        ----------
        device_name: name of the device to set (possible values are, cuda, cpu, and auto). When device_name is auto the cuda is used if available, otherwise cpu.

        
        Returns
        -------
        torch.device
            The device.
        """

        if device_name == "auto":
            return GlobalSettings().auto_device()

        self._device = torch.device(device_name)
        return self._device
    
    def get_device(self):
        """Get the device.

        Returns
        -------
        torch.device
            The device.
        """
        return self._device

    def set_workers(self, workers: int) -> None:
        """Set the number of workers.
        
        Parameters
        ----------
        workers : int
            The number of workers.
        """
        self._workers = max(1, min(workers, mp.cpu_count()))
    
    def auto_workers(self) -> int:
        """Set the number of workers to the number of cpu cores.
        
        Returns
        -------
        int
            The number of workers.
        """
        self._workers = mp.cpu_count()
        return self._workers

    def get_workers(self) -> int:
        """Get the number of workers.
        
        Returns
        -------
        int
            The number of workers.
        """
        return self._workers
    
    def get_progress_bar(self, progress_type: str) -> Progress:
        """Get the progress bar.
        
        Parameters
        ----------
        type : str
            The type of progress bar.
        
        Returns
        -------
        Progress
            The progress bar.
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
        
        Returns
        -------
        Live
            The live renderer.
        """
        return self._live_renderer


class Message:
    def __init__(self,
                 payload: Any,
                 msg_type: str="model",
                 sender: Optional[Any]=None):
        self.msg_type: str = msg_type
        self.payload: Any = payload
        self.sender: Optional[Any] = sender
    
    def get_size(self) -> int:
        if self.payload is None:
            return 1
        return sys.getsizeof(pickle.dumps(self.payload))
