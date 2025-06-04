"""Implementation of the Kafè [Kafe24]_ algorithm.

References:
    .. [Kafe24] Pian Qi, Diletta Chiaro, Fabio Giampaolo, and Francesco Piccialli.
       KAFÈ: Kernel Aggregation for FEderated. In ECML-PKDD (2024).
       URL: https://link.springer.com/content/pdf/10.1007/978-3-031-70359-1_4.pdf

"""

import sys
from typing import Collection, Sequence

import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from . import CentralizedFL  # NOQA

__all__ = ["KafeServer", "Kafe"]


class KafeServer(Server):

    def __init__(
        self,
        model: torch.nn.Module,
        test_set: FastDataLoader,
        clients: Sequence[Client],
        weighted: bool = False,
        bandwidth: float = 1.0,
    ):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.hyper_params.update(bandwidth=bandwidth)

    def aggregate(self, eligible: Sequence[Client], client_models: Collection[Module]) -> None:
        weights = self._get_client_weights(eligible)

        # get last layer of m clients' weights
        last_layer_weight_name = list(self.model.state_dict().keys())[-2]
        last_layer_bias_name = list(self.model.state_dict().keys())[-1]

        # Get model parameters and buffers
        model_params = dict(self.model.named_parameters())
        model_buffers = dict(self.model.named_buffers())  # Includes running_mean, running_var, etc.

        # Initialize accumulators for parameters
        avg_params = {key: torch.zeros_like(param.data) for key, param in model_params.items()}
        avg_buffers = {
            key: torch.zeros_like(buffer.data)
            for key, buffer in model_buffers.items()
            if "num_batches_tracked" not in key
        }

        max_num_batches_tracked = 0  # Track the max num_batches_tracked
        w_last_layer = []
        b_last_layer = []

        # Compute weighted sum (weights already sum to 1, so no division needed)
        for m, w in zip(client_models, weights):
            for key, param in m.named_parameters():
                if key == last_layer_weight_name:
                    w_last_layer.append(param.data.numpy().copy())
                    continue

                if key == last_layer_bias_name:
                    b_last_layer.append(param.data.numpy().copy())
                    continue

                avg_params[key].add_(param.data, alpha=w)

            for key, buffer in m.named_buffers():
                if "num_batches_tracked" not in key:
                    avg_buffers[key].add_(buffer.data, alpha=w)
                else:
                    max_num_batches_tracked = max(max_num_batches_tracked, buffer.item())

        for key in model_params.keys():
            if key in (last_layer_weight_name, last_layer_bias_name):
                continue
            model_params[key].data.lerp_(avg_params[key], self.hyper_params.lr)  # Soft update

        for key in model_buffers.keys():
            if "num_batches_tracked" not in key:
                model_buffers[key].data.lerp_(avg_buffers[key], self.hyper_params.lr)  # Soft update

        # Assign max num_batches_tracked
        for key in model_buffers.keys():
            if "num_batches_tracked" in key:
                model_buffers[key].data.fill_(max_num_batches_tracked)

        w_last_layer = np.array(w_last_layer).reshape(len(w_last_layer), -1)
        b_last_layer = np.array(b_last_layer).reshape(len(b_last_layer), -1)

        # using KDE get the kernel density of last layers
        kde_w = KernelDensity(kernel="gaussian", bandwidth=self.hyper_params.bandwidth).fit(
            w_last_layer, sample_weight=weights
        )
        kde_b = KernelDensity(kernel="gaussian", bandwidth=self.hyper_params.bandwidth).fit(
            b_last_layer, sample_weight=weights
        )

        # sample m samples and average, then obtain a new last layer for the global model
        w_last_layer_new = np.mean(kde_w.sample(len(w_last_layer)), axis=0)
        b_last_layer_new = np.mean(kde_b.sample(len(b_last_layer)), axis=0)

        model_sd = self.model.state_dict()
        model_sd[last_layer_weight_name].data = torch.tensor(
            w_last_layer_new.reshape(model_sd[last_layer_weight_name].shape)
        )
        model_sd[last_layer_bias_name].data = torch.tensor(
            b_last_layer_new.reshape(model_sd[last_layer_bias_name].shape)
        )


class Kafe(CentralizedFL):

    def get_server_class(self) -> type[Server]:
        return KafeServer
