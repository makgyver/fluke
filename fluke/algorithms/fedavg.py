"""Implementation of the Federated Averaging [FedAVG]_ algorithm.

References:
    .. [FedAVG] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y
       Arcas. Communication-efficient learning of deep networks from decentralized data.
       In AISTATS, 2017. URL: https://arxiv.org/abs/1602.05629

"""
import sys
sys.path.append(".")
sys.path.append("..")

from . import CentralizedFL  # NOQA


class FedAVG(CentralizedFL):
    pass
