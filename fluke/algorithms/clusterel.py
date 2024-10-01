import sys
from collections import OrderedDict
from typing import Any, Iterable
from copy import deepcopy

import torch

sys.path.append(".")
sys.path.append("..")

from ..comm import Message  # NOQA
from ..client import Client  # NOQA
from ..data import FastDataLoader  # NOQA
from ..utils.model import STATE_DICT_KEYS_TO_IGNORE, AllLayerOutputModel   # NOQA
from ..server import Server  # NOQA
from . import CentralizedFL  # NOQA


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # ||X.T Y||_F / (||X.T X||_F * ||Y.T Y||_F)
    # compute the gram matrix
    X_gram = X @ X.T
    Y_gram = Y @ Y.T
    XY_gram = X @ Y.T
    # centralize the gram matrix
    # H = I - 1/n 11^T
    # Kc = H K H
    n = X_gram.shape[0]
    I_n = torch.eye(n, device=X.device)
    H = I_n - torch.ones(n, n, device=X.device) / n
    X_gram = H @ X_gram @ H
    Y_gram = H @ Y_gram @ H
    XY_gram = H @ XY_gram @ H
    # compute the norms
    X_norm = torch.norm(X_gram, p='fro')
    Y_norm = torch.norm(Y_gram, p='fro')
    XY_norm = torch.norm(XY_gram, p='fro') ** 2
    return XY_norm / (X_norm * Y_norm)


class ClusterelServer(Server):
    def __init__(self,
                 model: torch.nn.Module,
                 test_data: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = True,
                 layer_name: str = None,
                 **kwargs: dict[str, Any]):
        assert test_data is not None, "'test_data' is a required argument for ClusterelServer."
        super().__init__(model, test_data, clients, weighted)

        if layer_name is None:
            layer_name = ".".join(list(model.state_dict())[-1].split(".")[:-1])  # CHECKME

        self.hyper_params.update(layer_name=layer_name)
        self.temp_models = None

    def _sample_data(self,
                     X: torch.Tensor,
                     y: torch.Tensor,
                     sample_x_class: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        samp_X, samp_y = None, None
        for c in set(y.tolist()):
            idx = torch.where(y == c)[0]
            idx = idx[torch.randperm(idx.size(0))[:sample_x_class]]
            if samp_X is None:
                samp_X, samp_y = X[idx], y[idx]
            else:
                samp_X = torch.cat((samp_X, X[idx]), dim=0)
                samp_y = torch.cat((samp_y, y[idx]), dim=0)

        return samp_X, samp_y

    def _compute_clusters(self, eligible: Iterable[Client]) -> torch.Tensor:
        sim_score = torch.ones(len(eligible), len(eligible))
        X, y = self._sample_data(self.test_set.tensors[0], self.test_set.tensors[1], 10)
        for i, c1 in enumerate(eligible):
            c1_model = AllLayerOutputModel(c1.model)
            for j, c2 in enumerate(eligible):
                if i >= j:
                    continue
                c2_model = AllLayerOutputModel(c2.model)

                _ = c1_model(X)
                _ = c2_model(X)

                repr1 = c1_model.activations_out[self.hyper_params.layer_name]
                repr2 = c2_model.activations_out[self.hyper_params.layer_name]

                sim_score[i, j] = linear_CKA(repr1, repr2)
                sim_score[j, i] = sim_score[i, j]
                c2_model.deactivate()
            c1_model.deactivate()
        return sim_score

    def broadcast_model(self, eligible: Iterable[Client]) -> None:
        if self.temp_models is None:
            return super().broadcast_model(eligible)

        for i, m in enumerate(self.temp_models):
            self._channel.send(message=Message(m, "model", self), mbox=eligible[i])
        self.temp_models = None

    def finalize(self) -> None:
        self.temp_models = None
        return super().finalize()

    def aggregate(self, eligible: Iterable[Client]) -> None:
        avg_model_sd = OrderedDict()
        clients_model = self.get_client_models(eligible, state_dict=False)
        clients_sd = [m.state_dict() for m in clients_model]
        weights = self._get_client_weights(eligible)
        sim_score = self._compute_clusters(eligible)

        self.temp_models = [None for _ in range(len(eligible))]
        weights = torch.FloatTensor(weights)
        for i, client_model in enumerate(clients_model):
            self.temp_models[i] = deepcopy(client_model)
            avg_model_sd = OrderedDict()

            w = torch.nn.functional.softmax(sim_score[i] * weights)
            for j, client_sd in enumerate(clients_sd):
                for key in self.model.state_dict().keys():
                    if key not in avg_model_sd:
                        avg_model_sd[key] = w[j] * client_sd[key]
                    else:
                        avg_model_sd[key] = avg_model_sd[key] + w[j] * client_sd[key]

            self.temp_models[i].load_state_dict(avg_model_sd)

        # Update the global model
        avg_model_sd = OrderedDict()
        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue
            for i, client_sd in enumerate(clients_sd):
                if key not in avg_model_sd:
                    avg_model_sd[key] = weights[i] * client_sd[key]
                else:
                    avg_model_sd[key] = avg_model_sd[key] + weights[i] * client_sd[key]

        self.model.load_state_dict(avg_model_sd)


class Clusterel(CentralizedFL):

    def get_server_class(self) -> Server:
        return ClusterelServer
