import torch

class Singleton(type):
    """Singleton metaclass."""
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GlobalSettings(metaclass=Singleton):
    """Global settings for the library.""" 
    
    _device = 'cpu'

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