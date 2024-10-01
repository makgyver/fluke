"""Implementation of the Kafè [Kafe24]_ algorithm.

References:
    .. [Kafe24] Pian Qi, Diletta Chiaro, Fabio Giampaolo, and Francesco Piccialli.
       KAFÈ: Kernel Aggregation for FEderated. In ECML-PKDD (2024).
       URL: https://link.springer.com/content/pdf/10.1007/978-3-031-70359-1_4.pdf

"""
from collections import OrderedDict
import sys
from typing import Iterable

import torch
import numpy as np
from sklearn.neighbors import KernelDensity

sys.path.append(".")
sys.path.append("..")

from . import CentralizedFL  # NOQA
from ..data import FastDataLoader  # NOQA
from ..client import Client  # NOQA
from ..server import Server  # NOQA
from ..utils.model import STATE_DICT_KEYS_TO_IGNORE  # NOQA

__all__ = [
    "KafeServer",
    "Kafe"
]


class KafeServer(Server):

    def __init__(self,
                 model: torch.nn.Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False,
                 bandwidth: float = 1.0):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.hyper_params.update(bandwidth=bandwidth)

    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_sd = self.get_client_models(eligible)
        weights = self._get_client_weights(eligible)

        # get last layer of m clients' weights
        last_layer_weight_name = list(clients_sd[0].keys())[-2]
        last_layer_bias_name = list(clients_sd[0].keys())[-1]

        for key in self.model.state_dict().keys():
            if key in (last_layer_weight_name, last_layer_bias_name):
                continue

            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue
            for i, client_sd in enumerate(clients_sd):
                if key not in avg_model_sd:
                    avg_model_sd[key] = weights[i] * client_sd[key]
                else:
                    avg_model_sd[key] = avg_model_sd[key] + weights[i] * client_sd[key]

        w_last_layer = []
        b_last_layer = []

        for csd in clients_sd:
            w_last_layer.append(np.array(csd[last_layer_weight_name]))
            b_last_layer.append(np.array(csd[last_layer_bias_name]))

        w_last_layer = np.array(w_last_layer).reshape(len(w_last_layer), -1)
        b_last_layer = np.array(b_last_layer).reshape(len(b_last_layer), -1)

        # using KDE get the kernel density of last layers
        kde_w = KernelDensity(kernel='gaussian',
                              bandwidth=self.hyper_params.bandwidth).fit(w_last_layer,
                                                                         sample_weight=weights)
        kde_b = KernelDensity(kernel='gaussian',
                              bandwidth=self.hyper_params.bandwidth).fit(b_last_layer,
                                                                         sample_weight=weights)

        # sample m samples and average, then obtain a new last layer for the global model
        w_last_layer_new = np.mean(kde_w.sample(len(w_last_layer)), axis=0)
        b_last_layer_new = np.mean(kde_b.sample(len(b_last_layer)), axis=0)

        # update last layer
        avg_model_sd[last_layer_weight_name] = torch.tensor(w_last_layer_new.reshape(
            clients_sd[0][last_layer_weight_name].shape))
        avg_model_sd[last_layer_bias_name] = torch.tensor(b_last_layer_new.reshape(
            clients_sd[0][last_layer_bias_name].shape))

        self.model.load_state_dict(avg_model_sd)


class Kafe(CentralizedFL):

    def get_server_class(self) -> Server:
        return KafeServer
