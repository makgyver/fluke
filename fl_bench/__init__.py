from typing import Any, Union, Iterable
import torch
import multiprocessing as mp

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