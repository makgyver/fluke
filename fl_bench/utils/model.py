"""This submodule provides utilities for pytorch model manipulation."""
import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module


def diff_model(model_dict1: dict, model_dict2: dict):
    """Compute the difference between two model state dictionaries.

    The difference is computed at the level of the parameters.

    Args:
        model_dict1 (dict): The state dictionary of the first model.
        model_dict2 (dict): The state dictionary of the second model.

    Returns:
        OrderedDict: The state dictionary of the difference between the two models.
    
    Raises:
        AssertionError: If the two models have different architectures.
    """
    assert model_dict1.keys() == model_dict2.keys(), "Models have not the same architecture"
    return OrderedDict({key: model_dict1[key] - model_dict2[key] for key in model_dict1.keys()})


def merge_models(model_1: Module, model_2: Module, lam: float):
    """Merge two models using a linear interpolation.

    The interpolation is done at the level of the parameters using the formula:
    `merged_model = (1 - lam) * model_1 + lam * model_2`.

    Args:
        model_1 (Module): The first model.
        model_2 (Module): The second model.
        lam (float): The interpolation constant.
    
    Returns:
        Module: The merged model.
    """
    merged_model = deepcopy(model_1)
    for name, param in merged_model.named_parameters():
        param.data = (1 - lam) * model_1.get_parameter(name).data + lam  * model_2.get_parameter(name).data
    return merged_model


class MMMixin:
    """Mixin class for model interpolation.
    
    This class provides the necessary methods to interpolate between two models. This mixin class
    must be used as a parent class for the pytorch modules that need to be interpolated.

    Attributes:
        lam (float): The interpolation constant.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = None

    def set_lambda(self, lam) -> None:
        """Set the interpolation constant.

        Args:
            lam (float): The interpolation constant.
        """
        self.lam = lam
    
    def get_lambda(self) -> float:
        """Get the interpolation constant.

        Returns:
            float: The interpolation constant.
        """
        return self.lam

    # @abstractmethod
    def get_weight(self) -> torch.Tensor:
        """Get the interpolated weights.

        Returns:
            torch.Tensor: The interpolated weights.
        """
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        return w


class SubspaceLinear(MMMixin, nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        w, b = self.get_weight()
        x = F.linear(input=x, weight=w, bias=b)
        return x

class TwoParamLinear(SubspaceLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))
        if self.bias is not None:
            self.bias_local = nn.Parameter(torch.zeros_like(self.bias))
                                     
class LinesLinear(TwoParamLinear):
    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        if self.bias is not None:
            b = (1 - self.lam) * self.bias + self.lam * self.bias_local
        else:
            b = None
        return w, b


class SubspaceConv(MMMixin, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        w, b = self.get_weight()
        x = F.conv2d(input=x, 
                     weight=w, 
                     bias=b, 
                     stride=self.stride, 
                     padding=self.padding, 
                     dilation=self.dilation, 
                     groups=self.groups)
        return x

class TwoParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))
        if self.bias is not None:
            self.bias_local = nn.Parameter(torch.zeros_like(self.bias))

class LinesConv(TwoParamConv):
    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        if self.bias is not None:
            b = (1 - self.lam) * self.bias + self.lam * self.bias_local
        else:
            b = None
        return w, b

class SubspaceLSTM(MMMixin, nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        w = self.get_weight()
        h = (
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device), 
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        )
        with torch.no_grad():
            torch._cudnn_rnn_flatten_weight(
                weight_arr=w, 
                weight_stride0=(4 if self.bias else 2),
                input_size=self.input_size,
                mode=torch.backends.cudnn.rnn.get_cudnn_mode('LSTM'),
                hidden_size=self.hidden_size,
                proj_size=0,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False
            )
        result = torch._VF.lstm(x, h, w, self.bias, self.num_layers, 0.0, self.training, self.bidirectional, self.batch_first) 
        return result[0], result[1:]
    
class TwoParamLSTM(SubspaceLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for l in range(self.num_layers):
            setattr(self, f'weight_hh_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'weight_hh_l{l}'))))
            setattr(self, f'weight_ih_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'weight_ih_l{l}'))))
            if self.bias:
                setattr(self, f'bias_hh_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'bias_hh_l{l}'))))
                setattr(self, f'bias_ih_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'bias_ih_l{l}'))))
            
class LinesLSTM(TwoParamLSTM):
    def get_weight(self):
        weight_list = []
        for l in range(self.num_layers):
            weight_list.append((1 - self.lam) * getattr(self, f'weight_ih_l{l}') + self.lam * getattr(self, f'weight_ih_l{l}_local'))
            weight_list.append((1 - self.lam) * getattr(self, f'weight_hh_l{l}') + self.lam * getattr(self, f'weight_hh_l{l}_local'))
        return weight_list


class SubspaceEmbedding(MMMixin, nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        w = self.get_weight()
        x = F.embedding(input=x, weight=w)
        return x

class TwoParamEmbedding(SubspaceEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))
        
class LinesEmbedding(TwoParamEmbedding):
    pass    


class SubspaceBN(MMMixin, nn.BatchNorm2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()

        # The rest is code in the PyTorch source forward pass for batchnorm.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
                    
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        return F.batch_norm(
            x,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
    
class TwoParamBN(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.Tensor(self.num_features))
        self.bias_local = nn.Parameter(torch.Tensor(self.num_features))
        
class LinesBN(TwoParamBN):
    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        if self.bias is not None:
            b = (1 - self.lam) * self.bias + self.lam * self.bias_local
        else:
            b = None
        return w, b
    

def _recursive_mix_networks(merged_net: Module, global_model: Module, local_model: Module):
    layers = {}
    for n, x in merged_net.named_children():
        if isinstance(x, torch.nn.Linear):
            layer = LinesLinear(x.in_features, x.out_features, bias=x.bias is not None)
        elif isinstance(x, torch.nn.Conv2d):
            layer = LinesConv(x.in_channels, 
                              x.out_channels, 
                              x.kernel_size, 
                              x.stride, 
                              x.padding, 
                              x.dilation, 
                              x.groups, 
                              x.bias is not None)
        elif isinstance(x, torch.nn.BatchNorm2d):
            layer = LinesBN(x.num_features)
        elif isinstance(x, torch.nn.Embedding):
            layer = LinesEmbedding(x.num_embeddings, x.embedding_dim)
        elif isinstance(x, torch.nn.LSTM):
            layer = LinesLSTM(x.input_size, 
                              x.hidden_size,
                              x.num_layers,
                              x.bias,
                              x.batch_first,
                              x.dropout,
                              x.bidirectional)
        else:
            layers[n] = _recursive_mix_networks(x, 
                                                getattr(global_model, n), 
                                                getattr(local_model, n))
            continue

        for namep, _ in x.named_parameters():
            setattr(layer, namep, getattr(global_model, n).get_parameter(namep))
            setattr(layer, namep + "_local", getattr(local_model, n).get_parameter(namep))
        layers[n] = layer
    
    return layers


def _recursive_set_layer(module: Module, layers: dict):
    for n, l in layers.items():
        if isinstance(l, dict):
            _recursive_set_layer(getattr(module, n), l)
        else:
            setattr(module, n, l)


def mix_networks(global_model: Module, local_model: Module, lamda: float):
    merged_net = deepcopy(global_model)
    layers = _recursive_mix_networks(merged_net, global_model, local_model)
    _recursive_set_layer(merged_net, layers)
    set_lambda_model(merged_net, lamda)
    return merged_net


def _set_lambda(module: Module, lam: float, layerwise: bool=False):
    """Set model interpolation constant.
    
    Args:
        module (torch.nn.Module): module
        lam (float): constant used for interpolation (0 means a retrieval of a global model, 1 means a retrieval of a local model)
        layerwise (bool): set different lambda layerwise or not
    """
    if (
        isinstance(module, torch.nn.Conv2d) 
        or isinstance(module, torch.nn.BatchNorm2d)
        or isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.LSTM)
        or isinstance(module, torch.nn.Embedding)
    ):
        if layerwise:
            lam = np.random.uniform(0.0, 1.0)
        setattr(module, 'lam', lam)


def set_lambda_model(model: Module, lam: float, layerwise: bool=False):
    model.apply(partial(_set_lambda, lam=lam, layerwise=layerwise))


def get_local_model_dict(model):
    return {k.replace("_local", ""): v for k, v in model.state_dict().items() if "_local" in k}


def get_global_model_dict(model):
    return {k: v for k, v in model.state_dict().items() if "_local" not in k}
