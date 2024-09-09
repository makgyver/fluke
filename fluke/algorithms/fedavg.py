"""Implementation of the Federated Averaging [FedAVG17]_ algorithm.

References:
    .. [FedAVG17] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y
       Arcas. Communication-efficient learning of deep networks from decentralized data.
       In AISTATS (2017). URL: https://arxiv.org/abs/1602.05629

"""
import sys

sys.path.append(".")
sys.path.append("..")

from . import CentralizedFL  # NOQA

__all__ = [
    "FedAVG"
]


class FedAVG(CentralizedFL):
    pass
