"""Implementation of the FedBN [FedBN21]_ algorithm.

References:
    .. [FedBN21] Xiaoxiao Li, Meirui JIANG, Xiaofei Zhang, Michael Kamp, and Qi Dou. FedBN:
       Federated Learning on Non-IID Features via Local Batch Normalization. In ICLR (2021).
       URL: https://openreview.net/pdf?id=6YEQUn0QICG
"""
import sys

import torch

sys.path.append(".")
sys.path.append("..")

from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA

# Same idea in https://link.springer.com/chapter/10.1007/978-3-030-60548-3_13

__all__ = [
    "FedBNClient",
    "FedBN"
]


class FedBNClient(Client):

    def _get_bn_keys(self, module: torch.nn.Module, running_key: str = "") -> list:
        keys = []
        for key, value in module.named_children():
            if isinstance(value, torch.nn.modules.batchnorm._BatchNorm):
                keys.append(f"{running_key}{key}")
            else:
                keys += self._get_bn_keys(value, running_key=key + ".")
        return keys

    def receive_model(self) -> None:
        global_model = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.model = global_model

            to_skip = self._get_bn_keys(self.model)
            self._to_keep = []
            for key in self.model.state_dict():
                ok = True
                for skip in to_skip:
                    if key.startswith(skip):
                        ok = False
                        break
                if ok:
                    self._to_keep.append(key)

        else:
            with torch.no_grad():
                for key in global_model.state_dict().keys():
                    if key in self._to_keep:
                        self.model.state_dict()[key].data.copy_(
                            global_model.state_dict()[key].clone())


class FedBN(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedBNClient
