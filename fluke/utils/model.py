"""This submodule provides utilities for pytorch model manipulation."""
import sys
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F

sys.path.append(".")
sys.path.append("..")


__all__ = [
    "MMMixin",
    "LinesLinear",
    "LinesConv2d",
    "LinesLSTM",
    "LinesEmbedding",
    "LinesBN2d",
    "diff_model",
    "merge_models",
    "set_lambda_model",
    "get_output_shape",
    "get_local_model_dict",
    "get_global_model_dict",
    "mix_networks",
    "batch_norm_to_group_norm",
    "safe_load_state_dict",
    "state_dict_zero_like",
    "flatten_parameters",
    "check_model_fit_mem",
    "AllLayerOutputModel"
]

STATE_DICT_KEYS_TO_IGNORE = tuple()  # ("num_batches_tracked", "running_mean", "running_var")


class MMMixin:
    """Mixin class for model interpolation.
    This class provides the necessary methods to interpolate between two models. This mixin class
    must be used as a parent class for the PyTorch modules that need to be interpolated.

    Tip:
        Ideally, when using this mixin to implement a new class ``M``, this should be mixed with
        a class ``C`` that extends :class:`torch.nn.Module` and the interpolation of the parameters
        should happen between the parameters in ``C`` and a new set of parameters defined in ``A``.
        This type of multiple inheritance must have as first parent the class :class:`MMMixin` and
        as second parent a class that extends :class:`torch.nn.Module`.
        For example:

        .. code-block:: python
            :linenos:

            # C is a class that extends torch.nn.Module
            class M(MMMixin, C):
                def __init__(self, *args, **kwargs: dict[str, Any]):
                    super().__init__(*args, **kwargs)
                    self.weight_local = nn.Parameter(torch.zeros_like(self.weight))

        In this case, the default implementation of the method :meth:`get_weight` will work and
        will interpolate between the ``weight`` and the ``weight_local`` attribute of the module.

    Attributes:
        lam (float): The interpolation constant.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam: float = None

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
        """Get the interpolated weights of the layer or module according to the interpolation
        constant :attr:`lam`. The default implementation assumes that the layer or module has a
        ``weight`` attribute and a ``weight_local`` attribute that are both tensors of the same
        shape. The interpolated weights are computed as:
        ``w = (1 - self.lam) * self.weight + self.lam * self.weight_local``.

        Returns:
            torch.Tensor: The interpolated weights.
        """
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        return w


class LinesLinear(MMMixin, nn.Linear):
    """Linear layer with gloabl and local weights. The weights are interpolated using the
    interpolation constant ``lam``. Thus, the forward pass of this layer will use the interpolated
    weights.

    Note:
        The global weights are the "default" weights of the :class:`torch.nn.Linear` layer,
        while the local ones are in the submodule ``weight_local`` (and ``bias_local``).

    Attributes:
        weight_local (torch.Tensor): The local weights.
        bias_local (torch.Tensor): The local bias.
    """

    def __init__(self, *args, **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))
        if self.bias is not None:
            self.bias_local = nn.Parameter(torch.zeros_like(self.bias))

    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        if self.bias is not None:
            b = (1 - self.lam) * self.bias + self.lam * self.bias_local
        else:
            b = None
        return w, b

    def forward(self, x):
        w, b = self.get_weight()
        x = F.linear(input=x, weight=w, bias=b)
        return x


class LinesConv2d(MMMixin, nn.Conv2d):
    """Conv2d layer with gloabl and local weights. The weights are interpolated using the
    interpolation constant ``lam``. Thus, the forward pass of this layer will use the interpolated
    weights.

    Note:
        The global weights are the "default" weights of the :class:`torch.nn.Conv2d` layer, while
        the local ones are in the submodule ``weight_local`` (and ``bias_local``).

    Attributes:
        weight_local (torch.Tensor): The local weights.
        bias_local (torch.Tensor): The local bias.
    """

    def __init__(self, *args, **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))
        if self.bias is not None:
            self.bias_local = nn.Parameter(torch.zeros_like(self.bias))

    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        if self.bias is not None:
            b = (1 - self.lam) * self.bias + self.lam * self.bias_local
        else:
            b = None
        return w, b

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


class LinesLSTM(MMMixin, nn.LSTM):
    """LSTM layer with gloabl and local weights. The weights are interpolated using the
    interpolation constant ``lam``. Thus, the forward pass of this layer will use the interpolated
    weights.

    Note:
        The global weights are the "default" weights of the :class:`torch.nn.LSTM` layer, while the
        local ones are in the submodules ``weight_hh_l{layer}_local`` and
        ``weight_ih_l{layer}_local``, where ``layer`` is the layer number. Similar considerations
        apply to the biases.

    Caution:
        This class may not work properly an all devices. If you encounter any issues, please open
        an issue in the repository.


    Attributes:
        weight_hh_l{layer}_local (torch.Tensor): The local hidden-hidden weights of layer ``layer``.
        weight_ih_l{layer}_local (torch.Tensor): The local input-hidden weights of layer ``layer``.
        bias_hh_l{layer}_local (torch.Tensor): The local hidden-hidden biases of layer ``layer``.
        bias_ih_l{layer}_local (torch.Tensor): The local input-hidden biases of layer ``layer``.
    """

    def __init__(self, *args, **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)
        for layer in range(self.num_layers):
            setattr(self, f'weight_hh_l{layer}_local', nn.Parameter(
                torch.zeros_like(getattr(self, f'weight_hh_l{layer}'))))
            setattr(self, f'weight_ih_l{layer}_local', nn.Parameter(
                torch.zeros_like(getattr(self, f'weight_ih_l{layer}'))))
            if self.bias:
                setattr(self, f'bias_hh_l{layer}_local', nn.Parameter(
                    torch.zeros_like(getattr(self, f'bias_hh_l{layer}'))))
                setattr(self, f'bias_ih_l{layer}_local', nn.Parameter(
                    torch.zeros_like(getattr(self, f'bias_ih_l{layer}'))))

    def get_weight(self):
        weight_list = []
        for layer in range(self.num_layers):
            weight_list.append((1 - self.lam) * getattr(self,
                               f'weight_ih_l{layer}') +
                               self.lam * getattr(self, f'weight_ih_l{layer}_local'))
            weight_list.append((1 - self.lam) * getattr(self,
                               f'weight_hh_l{layer}') +
                               self.lam * getattr(self, f'weight_hh_l{layer}_local'))
        return weight_list

    def forward(self, x):
        w = self.get_weight()
        h = (
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device),
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        )
        with torch.no_grad():
            # if torch._use_cudnn_rnn_flatten_weight():
            #     torch._cudnn_rnn_flatten_weight(
            #         weight_arr=w,
            #         weight_stride0=(4 if self.bias else 2),
            #         input_size=self.input_size,
            #         mode=2,  # torch.backends.cudnn.rnn.get_cudnn_mode('LSTM'),
            #         hidden_size=self.hidden_size,
            #         proj_size=0,
            #         num_layers=self.num_layers,
            #         batch_first=True,
            #         bidirectional=False
            #     )
            # else:
            self._flat_weights = w
            self.flatten_parameters()
        result = torch._VF.lstm(x, h, w, self.bias, self.num_layers, 0.0,
                                self.training, self.bidirectional, self.batch_first)
        return result[0], result[1:]


class LinesEmbedding(MMMixin, nn.Embedding):
    """Embedding layer with gloabl and local weights. The weights are interpolated using the
    interpolation constant ``lam``. Thus, the forward pass of this layer will use the interpolated
    weights.

    Note:
        The global weights are the "default" weights of the class:`torch.nn.Embedding` layer,
        while the local ones are in the submodule ``weight_local``.

    Attributes:
        weight_local (torch.Tensor): The local weights.
    """

    def __init__(self, *args, **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x):
        w = self.get_weight()
        x = F.embedding(input=x, weight=w)
        return x


class LinesBN2d(MMMixin, nn.BatchNorm2d):
    """BatchNorm2d layer with gloabl and local weights. The weights are interpolated using the
    interpolation constant ``lam``. Thus, the forward pass of this layer will use the interpolated
    weights.

    Note:
        The global weights are the "default" weights of the ``nn.BatchNorm2d`` layer, while the
        local ones are in the submodules ``weight_local`` and ``bias_local``.

    Attributes:
        weight_local (torch.Tensor): The local weights.
        bias_local (torch.Tensor): The local bias.
    """

    def __init__(self, *args, **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.Tensor(self.num_features))
        self.bias_local = nn.Parameter(torch.Tensor(self.num_features))

    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        if self.bias is not None:
            b = (1 - self.lam) * self.bias + self.lam * self.bias_local
        else:
            b = None
        return w, b

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


def _recursive_mix_networks(merged_net: Module, global_model: Module, local_model: Module) -> dict:
    layers = {}
    if next(merged_net.named_children(), None) is None:
        named_modules = merged_net.named_modules()
    else:
        named_modules = merged_net.named_children()
    for n, x in named_modules:
        if isinstance(x, torch.nn.Linear):
            layer = LinesLinear(x.in_features, x.out_features, bias=x.bias is not None)
        elif isinstance(x, torch.nn.Conv2d):
            layer = LinesConv2d(x.in_channels,
                                x.out_channels,
                                x.kernel_size,
                                x.stride,
                                x.padding,
                                x.dilation,
                                x.groups,
                                x.bias is not None)
        elif isinstance(x, torch.nn.BatchNorm2d):
            layer = LinesBN2d(x.num_features)
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
        elif isinstance(x, (torch.nn.BatchNorm1d, torch.nn.BatchNorm3d)):
            raise NotImplementedError("BatchNorm1d and BatchNorm3d are not supported")
        elif next(x.parameters(), None) is None:
            layers[n] = x
            continue
        else:
            layers[n] = _recursive_mix_networks(x,
                                                getattr(global_model, n),
                                                getattr(local_model, n))
            continue

        for namep, _ in x.named_parameters():
            setattr(layer, namep, getattr(global_model, n).get_parameter(
                namep) if n else getattr(global_model, namep))
            setattr(layer, namep + "_local", getattr(local_model, n).get_parameter(namep)
                    if n else getattr(local_model, namep))
        layers[n] = layer

    return layers


def _recursive_set_layer(module: Module, layers: dict):
    for n, l in layers.items():
        if isinstance(l, dict):
            _recursive_set_layer(getattr(module, n), l)
        else:
            setattr(module, n, l)


def mix_networks(global_model: Module, local_model: Module, lamda: float) -> MMMixin:
    """Mix two networks using a linear interpolation.
    This method takes two models and a lambda value and returns a new model that is a linear
    interpolation of the two input models. It transparenly handles the interpolation of the
    different layers of the models. The returned model implements the :class:`MMMixin` class and
    has all the layers swapped with the corresponding interpolated layers.

    See Also:
        - :class:`MMMixin`
        - :class:`LinesLinear`
        - :class:`LinesConv2d`
        - :class:`LinesBN2d`
        - :class:`LinesEmbedding`
        - :class:`LinesLSTM`

    Args:
        global_model (torch.nn.Module): The global model.
        local_model (torch.nn.Module): The local model.
        lamda (float): The interpolation constant.

    Returns:
        Module: The merged/interpolated model that implements the ``MMMixin`` class.
    """
    merged_net = deepcopy(global_model)
    layers = _recursive_mix_networks(merged_net, global_model, local_model)
    if len(layers) == 1 and "" in layers:
        merged_net = layers[""]
    else:
        _recursive_set_layer(merged_net, layers)
    set_lambda_model(merged_net, lamda)
    return merged_net


def _set_lambda(module: MMMixin, lam: float, layerwise: bool = False) -> None:
    """Set model interpolation constant.

    Args:
        module (torch.nn.Module): module
        lam (float): constant used for interpolation (0 means a retrieval of a global model, 1
          means a retrieval of a local model)
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


def set_lambda_model(model: MMMixin, lam: float, layerwise: bool = False) -> None:
    """Set model interpolation constant.

    Warning:
        This function performs an inplace operation on the model, and
        it assumes that the model has been built using the :class:`MMMixin` classes.

    Args:
        model (torch.nn.Module): model
        lam (float): constant used for interpolation (0 means a retrieval of a global model, 1
          means a retrieval of a local model)
        layerwise (bool): set different lambda layerwise or not
    """
    setattr(model.__class__, 'get_lambda', lambda self: lam)
    model.apply(partial(_set_lambda, lam=lam, layerwise=layerwise))


def get_local_model_dict(model: MMMixin) -> OrderedDict:
    """Get the local model state dictionary.

    Args:
        model (torch.nn.Module): the model.

    Returns:
        OrderedDict: the local model state dictionary.
    """
    return OrderedDict({k.replace("_local", ""): deepcopy(v)
                        for k, v in model.state_dict().items() if "_local" in k})


def get_global_model_dict(model: MMMixin) -> OrderedDict:
    """Get the global model state dictionary.

    Args:
        model (torch.nn.Module): the model.

    Returns:
        OrderedDict: the global model state dictionary.
    """
    return OrderedDict({k: deepcopy(v) for k, v in model.state_dict().items() if "_local" not in k})


def get_output_shape(model: Module, input_dim: tuple[int, ...]) -> tuple[int, ...]:
    """Get the output shape of a model given the shape of the input.

    Args:
        model (torch.nn.Module): The model to get the output shape.
        input_dim (tuple[int, ...]): The expected input shape of the model.

    Returns:
        tuple[int, ...]: The output shape of the model.
    """
    return model(torch.rand(*(input_dim))).data.shape


def diff_model(model_dict1: dict, model_dict2: dict) -> OrderedDict:
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


def merge_models(model_1: Module, model_2: Module, lam: float) -> Module:
    """Merge two models using a linear interpolation.
    The interpolation is done at the level of the parameters using the formula:
    ``merged_model = (1 - lam) * model_1 + lam * model_2``.

    Args:
        model_1 (torch.nn.Module): The first model.
        model_2 (torch.nn.Module): The second model.
        lam (float): The interpolation constant.

    Returns:
        Module: The merged model.
    """
    merged_model = deepcopy(model_1)
    for name, param in merged_model.named_parameters():
        param.data = (1 - lam) * model_1.get_parameter(name).data + \
            lam * model_2.get_parameter(name).data
    return merged_model


def safe_load_state_dict(model1: Module, model2_state_dict: dict) -> None:
    """Load a state dictionary into a model.
    This function is a safe version of ``model.load_state_dict`` that handles the case in which the
    state dictionary has keys that match with ``STATE_DICT_KEYS_TO_IGNORE`` and thus have to be
    ignored.

    Caution:
        This function performs an inplace operation on ``model1``.

    Args:
        model1 (torch.nn.Module): The model to load the state dictionary.
        model2_state_dict (dict): The state dictionary.
    """
    model1_state_dict = model1.state_dict()
    new_state_dict = OrderedDict()
    for key, value in model2_state_dict.items():
        if not key.endswith(STATE_DICT_KEYS_TO_IGNORE):
            new_state_dict[key] = value
        else:
            new_state_dict[key] = model1_state_dict[key]
    model1.load_state_dict(new_state_dict)


def batch_norm_to_group_norm(layer: Module) -> Module:
    r"""Iterates over a whole model (or layer of a model) and replaces every
    batch norm 2D with a group norm

    Args:
        layer (torch.nn.Module): model or one layer of a model.

    Returns:
        torch.nn.Module: model with group norm layers instead of batch norm layers.

    Raises:
        ValueError: If the number of channels :math:`\notin \{2^i\}_{i=4}^{11}`
    """

    GROUP_NORM_LOOKUP = {
        16: 2,  # -> channels per group: 8
        32: 4,  # -> channels per group: 8
        64: 8,  # -> channels per group: 8
        128: 8,  # -> channels per group: 16
        256: 16,  # -> channels per group: 16
        512: 32,  # -> channels per group: 16
        1024: 32,  # -> channels per group: 32
        2048: 32,  # -> channels per group: 64
    }

    for name, _ in layer.named_modules():
        if name:
            try:
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    if num_channels in GROUP_NORM_LOOKUP:
                        layer._modules[name] = torch.nn.GroupNorm(
                            GROUP_NORM_LOOKUP[num_channels], num_channels)
                    else:
                        raise ValueError(
                            f"GroupNorm not implemented for {num_channels} channels")
            except AttributeError:
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def state_dict_zero_like(state_dict: OrderedDict) -> OrderedDict:
    output = OrderedDict()
    for k, v in state_dict.items():
        output[k] = torch.zeros_like(v)
    return output


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    """Returns the model parameters as a contiguous tensor.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        torch.Tensor: The model parameters as a contiguous tensor of shape (n,), where n is the
            number of parameters in the model.
    """
    n = sum(p.numel() for p in model.parameters())
    params = torch.zeros(n)
    i = 0
    for p in model.parameters():
        params_slice = params[i:i + p.numel()]
        params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        i += p.numel()
    return params


def check_model_fit_mem(model: torch.nn.Module,
                        input_size: tuple[int, ...],
                        num_clients: int,
                        device: str = 'cuda',
                        mps_default: bool = True):
    """Check if the models fit in the memory of the device.
    The method estimates the memory usage of the models, when all clients and the server own a
    single neural network, on the device and checks if the models fit in the memory of the device.

    Attention:
        This function only works for CUDA devices. For MPS devices, the function will
        always return the value of ``mps_default``. To date, PyTorch does not provide
        a way to estimate the memory usage of a model on an MPS device.

    Args:
        model (torch.nn.Module): The model to check.
        input_size (tuple[int, ...]): The input size of the model.
        num_clients (int): The number of clients in the federation.
        device (str, optional): The device to check. Defaults to 'cuda'.
        mps_default (bool, optional): The default value to return if the device is MPS.
    """

    # Ensure the device is available
    assert device in ["cuda", "mps"] or device.startswith("cuda:"), \
        "Invalid argument 'device'. Must be 'cuda', 'mps' or 'cuda:<device_id>'."

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available.")
    elif not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available on device: {device}")

    if device == "mps":
        return mps_default
    else:
        return _check_model_fit_mem_cuda(model, input_size, num_clients, device)


def _check_model_fit_mem_cuda(model: torch.nn.Module,
                              input_size: tuple[int, ...],
                              num_clients: int,
                              device: torch.device):

    # Get the current CUDA device
    cuda_device = torch.device(device)

    # Get the free memory in bytes
    free_mem = torch.cuda.mem_get_info(cuda_device.index)[0]
    # current_allocation = torch.cuda.memory_allocated(cuda_device)
    current_reserved = torch.cuda.memory_reserved(cuda_device)

    # Transfer the model to CUDA
    model = model.to(cuda_device)

    # Estimate the model memory
    # param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # Create a dummy input tensor with the specified size
    dummy_input = torch.randn(*input_size, device=cuda_device)

    # Forward pass to estimate memory for activations
    with torch.no_grad():
        model(dummy_input)

    # Get the current GPU memory allocated and cached
    # allocated_mem = torch.cuda.memory_allocated(cuda_device)
    reserved_mem = torch.cuda.memory_reserved(cuda_device)

    # Total estimated memory usage
    total_estimated_mem = (num_clients + 1) * (reserved_mem - current_reserved)

    # Compare the total estimated memory with the free memory
    return total_estimated_mem <= free_mem


def _recursive_register_hook(module: Module, hook: callable, name: str = "", handles: list = None):
    named_modules = module.named_children()
    empty = True
    if handles is None:
        handles = []
    for n, sub_module in named_modules:
        empty = False
        current_name = name + "." + n if name else n
        leaf = _recursive_register_hook(sub_module, hook, current_name, handles)
        if leaf:
            handles.append(sub_module.register_forward_hook(hook(current_name)))
    return empty


class AllLayerOutputModel(nn.Module):
    """Wrapper class to get the output of all layers in a model.
    Once the model is wrapped with this class, the activations of all layers can be accessed through
    the attributes ``activations_in`` and ``activations_out``.

    ``activations_in`` is a dictionary that contains the input activations of all layers.
    ``activations_out`` is a dictionary that contains the output activations of all layers.

    Note:
        The activations are stored in the order in which they are computed during the forward pass.

    Important:
        If you need to access the activations of a specific layer after a potential activation
        function, you should use the ``activations_in`` of the next layer. For example, if you
        need the activations of the first layer after the ReLU activation, you should use the
        activations_in of the second layer. These attribute may not include the activations of the
        last layer if it includes an activation function.

    Important:
        If your model includes as submodule all the activations functions (e.g., of type
        torch.nn.ReLU), then you can use the ``activations_out`` attribute to get all the
        activations (i.e., before and after the activation functions).

    Attributes:
        model (torch.nn.Module): The model to wrap.
        activations_in (OrderedDict): The input activations of all layers.
        activations_out (OrderedDict): The output activations of all layers.

    Args:
        model (torch.nn.Module): The model to wrap.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.activations_in = OrderedDict()
        self.activations_out = OrderedDict()
        self._handles = []
        self.activate()

    def is_active(self) -> bool:
        """Returns whether the all layer output model is active.

        Returns:
            bool: Whether the all layer output model is active.
        """
        return bool(self._handles)

    def activate(self) -> None:
        """Activate the all layer output functionality."""
        _recursive_register_hook(self.model, self._get_activation, handles=self._handles)
        # for layer in self.model.modules():
        #     self._handles.append(layer.register_forward_hook(self._get_activation(layer)))

    def deactivate(self, clear_activations: bool = True) -> None:
        """Deactivate the all layer output functionality."""
        for h in self._handles:
            h.remove()
        self._handles = []
        if clear_activations:
            self.activations_in = OrderedDict()
            self.activations_out = OrderedDict()

    def _get_activation(self, name):
        def hook(model, input, output):
            # if name not in self.activations_in:
            self.activations_in[name] = input[0].detach()
            self.activations_out[name] = output.detach()
        return hook

    def forward(self, x):
        return self.model(x)


# if __name__ == "__main__":
#     from fluke.nets import MNIST_2NN
#     from fluke.data.datasets import Datasets
#     from fluke.data import FastDataLoader
#     model = MNIST_2NN(softmax=True)
#     model_all = AllLayerOutputModel(model)
#     data = Datasets.MNIST()
#     dl = FastDataLoader(*data.train, num_labels=10, batch_size=2, shuffle=True)

#     yy = None
#     for x, y in dl:
#         model_all.model(x)
#         yy = model(x)
#         break

#     print(model_all.activations_in.keys())
#     print()
#     print(model_all.activations_out.keys())

#     print(yy)

#     model_all.deactivate()
#     for x, y in dl:
#         model_all.model(x)
#         break

#     print(model_all.activations_in.keys())
#     print()
#     print(model_all.activations_out.keys())

#     model_all.activate()
#     for x, y in dl:
#         model_all.model(x)
#         break

#     print(model_all.activations_in.keys())
#     print()
#     print(model_all.activations_out.keys())
