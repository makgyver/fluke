"""Implementation of the [Moon21]_ algorithm.

References:
    .. [Moon21] Qinbin Li, Bingsheng He, and Dawn Song. Model-Contrastive Federated Learning.
       In CVPR (2021). URL: https://arxiv.org/abs/2103.16257
"""
import sys
from copy import deepcopy

import torch
from torch.nn import CosineSimilarity

sys.path.append(".")
sys.path.append("..")

from .. import FlukeENV  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils import clear_cuda_cache  # NOQA
from ..utils.model import safe_load_state_dict, unwrap  # NOQA

__all__ = [
    "MOONClient",
    "MOON"
]


class MOONClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int,
                 mu: float,
                 tau: float,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 0,
                 **kwargs):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, clipping=clipping, **kwargs)
        self.hyper_params.update(
            mu=mu,
            tau=tau
        )
        self.prev_model = None
        self.server_model = None
        self._attr_to_cache.extend(["prev_model", "server_model"])

    def receive_model(self) -> None:
        model = self.channel.receive(self.index, "server", msg_type="model").payload
        if self.model is None:
            # one of these deepcopy is not needed
            self.model = deepcopy(model)
            self.prev_model = deepcopy(model)
        else:
            safe_load_state_dict(self.prev_model, deepcopy(self.model.state_dict()))
            safe_load_state_dict(self.model, deepcopy(model.state_dict()))
        self.server_model = model

    def _model_to_dataparallel(self):
        super()._model_to_dataparallel()
        self.prev_model = torch.nn.DataParallel(self.prev_model,
                                                device_ids=FlukeENV().get_device_ids())
        self.server_model = torch.nn.DataParallel(self.server_model,
                                                  device_ids=FlukeENV().get_device_ids())

    def _dataparallel_to_model(self):
        super()._dataparallel_to_model()
        self.prev_model = self.prev_model.module
        self.server_model = self.server_model.module

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (override_local_epochs if override_local_epochs > 0
                       else self.hyper_params.local_epochs)
        cos = CosineSimilarity(dim=-1).to(self.device)
        self.model.to(self.device)
        self.prev_model.to(self.device)
        self.server_model.to(self.device)
        self.model.train()
        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                z_local = unwrap(self.model).encoder(X)  # , -1)
                y_hat = unwrap(self.model).head(z_local)
                loss_sup = self.hyper_params.loss_fn(y_hat, y)

                z_prev = unwrap(self.prev_model).encoder(X)  # , -1)
                z_global = unwrap(self.server_model).encoder(X)  # , -1)

                sim_lg = cos(z_local, z_global).reshape(-1, 1) / self.hyper_params.tau
                sim_lp = cos(z_local, z_prev).reshape(-1, 1) / self.hyper_params.tau
                loss_con = -torch.log(torch.exp(sim_lg) /
                                      (torch.exp(sim_lg) + torch.exp(sim_lp))).mean()

                loss = loss_sup + self.hyper_params.mu * loss_con
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.prev_model.cpu()
        self.server_model.cpu()
        self.model.cpu()
        clear_cuda_cache()
        return running_loss


class MOON(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return MOONClient
