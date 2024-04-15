import sys
sys.path.append(".")
sys.path.append("..")

from . import CentralizedFL  # NOQA


class FedAVG(CentralizedFL):
    """Federated Averaging algorithm.

    This class implements the Federated Averaging (FedAvg) algorithm from the paper:
    H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.
    Communication-efficient learning of deep networks from decentralized data. In AISTATS, 2017.
    URL: https://arxiv.org/abs/1602.05629.
    """
    pass
