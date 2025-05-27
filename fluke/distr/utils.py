"""
Utility module for `fluke.distr`.
"""

from torch.nn import Module

from ..utils import get_model

__all__ = ["ModelBuilder"]


class ModelBuilder:
    """A class to build and return instances of a model class with specified arguments.

    This class is useful for creating model instances in a flexible way, allowing for
    different configurations of the model without directly instantiating it.

    Attributes:
        model_class (type): The class of the model to be instantiated.
        args (tuple): Positional arguments to pass to the model constructor.
        kwargs (dict): Keyword arguments to pass to the model constructor.
    """

    def __init__(self, model_class: type, **kwargs):
        self.model_class = model_class
        self.kwargs = kwargs

    def build(self) -> Module:
        """Build and return an instance of the model.

        Returns:
            Module: An instance of the model class initialized with the provided arguments.
        """
        return get_model(mname=self.model_class, **self.kwargs)
