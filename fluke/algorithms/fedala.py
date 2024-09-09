"""Implementation of the FedALA [FedALA23]_ algorithm.

References:
    .. [FedALA23] Jianqing Zhang, Yang Hua, Hao Wang, Tao Song, Zhengui Xue, Ruhui Ma, Haibing Guan.
       FedALA: Adaptive Local Aggregation for Personalized Federated Learning
       In AAAI (2023). URL: https://arxiv.org/pdf/2212.01197v4

"""
import sys
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from fluke.client import Client

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..nets import EncoderHeadNet  # NOQA
from ..utils import OptimizerConfigurator  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from . import CentralizedFL  # NOQA

__all__ = [
    "FedALAClient",
    "FedALA"
]


class FedALAClient(Client):

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: nn.Module,
                 local_epochs: int = 3,
                 ala_sample_size: float = 0.8,
                 eta: float = 1.0,
                 conergence_threshold: float = 0.001,
                 loss_window_size: int = 10,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.hyper_params.update(ala_sample_size=ala_sample_size,
                                 eta=eta,
                                 conergence_threshold=conergence_threshold,
                                 loss_window_size=loss_window_size)
        self.weights = None
        self.start_phase = True

    def adaptive_local_aggregation(self, server_model: EncoderHeadNet):

        # keep the server encoder weights
        safe_load_state_dict(self.model.encoder, server_model.encoder.state_dict())

        random_sample_loader = FastDataLoader(*self.train_set.tensors,
                                              num_labels=self.train_set.num_labels,
                                              batch_size=self.train_set.batch_size,
                                              shuffle=True,
                                              percentage=self.hyper_params.ala_sample_size)

        temp_model = deepcopy(self.model)

        # frozen the encoder weights
        for param in temp_model.encoder.parameters():
            param.requires_grad = False

        optimizer = torch.optim.SGD(temp_model.head.parameters(), lr=0)

        if self.weights is None:
            self.weights = [torch.ones_like(p.data).to(self.device)
                            for p in self.model.head.parameters()]

        # initialize the temp model
        for param_t, param, param_g, weight in zip(temp_model.head.parameters(),
                                                   self.model.head.parameters(),
                                                   server_model.head.parameters(),
                                                   self.weights):
            param_t.data = param + (param_g - param) * weight

        converged = False
        losses = []
        while not converged and len(losses) < 100:

            for X, y in random_sample_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_hat = temp_model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                # optimizer.step()

                losses.append(loss.item())

                # update weight in this batch
                for param_t, param, param_g, weight in zip(temp_model.head.parameters(),
                                                           self.model.head.parameters(),
                                                           server_model.head.parameters(),
                                                           self.weights):
                    weight.data = torch.clamp(
                        weight - self.hyper_params.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(temp_model.head.parameters(),
                                                           self.model.head.parameters(),
                                                           server_model.head.parameters(),
                                                           self.weights):
                    param_t.data = param + (param_g - param) * weight

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.hyper_params.loss_window_size:
                loss_std = np.std(losses[-self.hyper_params.loss_window_size:])
                if loss_std < self.hyper_params.conergence_threshold:
                    converged = True

        self.start_phase = False
        safe_load_state_dict(self.model.head, temp_model.head.state_dict())

    def receive_model(self) -> None:
        server_model = self.channel.receive(self, self.server, msg_type="model").payload
        if self.model is None:
            self.model = server_model
        else:
            self.adaptive_local_aggregation(server_model)


class FedALA(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedALAClient
