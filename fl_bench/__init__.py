import torch

__all__ = [
    'algorithms'
]

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GlobalSettings(metaclass=Singleton):    
    _device = 'cpu'

    def auto_device(self):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    def set_device(self, device):
        self._device = device
        return self._device
    
    def get_device(self):
        return self._device