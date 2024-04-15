from copy import deepcopy
import torch
import sys
sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA


class FedBNClient(Client):

    def receive(self, message: Message) -> None:
        if message.msg_type == "model":
            global_model = message.payload
            if self.model is None:
                self.model = deepcopy(global_model)
            else:
                with torch.no_grad():
                    for key in global_model.state_dict().keys():
                        if not key.startswith("bn"):
                            self.model.state_dict()[key].data.copy_(global_model.state_dict()[key])


class FedBN(CentralizedFL):
    """

    This class implements the FedBN algorithm from the paper:
    Xiaoxiao Li, Meirui JIANG, Xiaofei Zhang, Michael Kamp, and Qi Dou. FedBN: Federated Learning
    on Non-IID Features via Local Batch Normalization. ICLR 2021.
    URL: https://openreview.net/pdf?id=6YEQUn0QICG

    Warning:
        To use this algorithm, you need to use a network with batch normalization layers whose names
        start with "bn".
    """

    def get_client_class(self) -> Client:
        return FedBNClient
