"""Implementation of the FedLD [FedLD24]_ algorithm.

References:
    .. [FedLD24] Shuang Zeng, Pengxin Guo, Shuai Wang, Jianbo Wang, Yuyin Zhou and Liangqiong Qu.
       Tackling Data Heterogeneity in Federated Learning via Loss Decomposition.
       In MICCAI (2024). URL: https://papers.miccai.org/miccai-2024/paper/1348_paper.pdf

"""
import sys
from typing import Collection

import torch
from torch.nn import CrossEntropyLoss, Module

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from . import CentralizedFL  # NOQA

__all__ = [
    "FedLD",
    "FedLDClient",
    "FedLDServer"
]


class MarginalLogLoss(Module):

    def __init__(self,
                 base_loss: Module = CrossEntropyLoss(),
                 lam: float = 0.2):
        super(MarginalLogLoss, self).__init__()
        self.base_loss = base_loss
        self.lam = lam

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_loss(y_pred, y_true)
        softmax_output = torch.softmax(y_pred, dim=1)
        magnitude = torch.norm(softmax_output, p=2)
        marginal_base_loss = base_loss + self.lam * torch.log(1 + magnitude ** 2)
        return marginal_base_loss

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    def __str__(self, indent: int = 0) -> str:
        indent_str = " " * indent
        return f"{indent_str}MarginalLogLoss(base_loss={self.base_loss}, lambda={self.lam})"

    def __repr__(self, indent: int = 0) -> str:
        return self.__str__(indent=indent)


class FedLDClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Module,
                 local_epochs: int = 3,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 0,
                 margin_lam: float = 0.2,
                 **kwargs):
        super().__init__(index=index,
                         train_set=train_set,
                         test_set=test_set,
                         optimizer_cfg=optimizer_cfg,
                         loss_fn=loss_fn,
                         local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs,
                         clipping=clipping,
                         **kwargs)
        self.hyper_params.update(margin_lam=margin_lam,
                                 loss_fn=MarginalLogLoss(base_loss=self.hyper_params.loss_fn,
                                                         lam=margin_lam))


def _get_client_grads(client_model: Module, server_model: Module) -> torch.Tensor:
    grads = []
    for key in server_model.state_dict().keys():
        grads.append(client_model.state_dict()[key].data.clone().detach().flatten(
        ) - server_model.state_dict()[key].data.clone().detach().flatten())
    return torch.cat(grads)


def _set_client_grads(client_model: Module,
                      server_model: Module,
                      new_grads: torch.Tensor):
    start = 0
    for key in server_model.state_dict().keys():
        dims = client_model.state_dict()[key].shape
        end = start + dims.numel()
        client_model.state_dict()[key].data.copy_(server_model.state_dict()[
            key].data.clone().detach() + new_grads[start:end].reshape(dims).clone())
        start = end
    return client_model


def pcgrad_svd(client_grads: Collection[Module],
               grad_history: dict,
               k_proportion: float = 0.1) -> tuple[torch.Tensor, dict]:
    """ Projecting conflicting gradients using SVD"""
    client_num = len(client_grads)
    client_grads_ = torch.stack(client_grads)
    grads = []
    grad_len = grad_history['grad_len']
    start = 0
    for key in grad_len.keys():
        g_len = grad_len[key]
        end = start + g_len
        layer_grad_history = grad_history[key]
        client_grads_layer = client_grads_[:, start:end]
        naive_avg_grad = torch.mean(client_grads_layer, dim=0, keepdim=True)
        if layer_grad_history is not None:
            if not torch.all(layer_grad_history == 0):
                hessian = 1 / client_grads_layer.size(0) * client_grads_layer @ client_grads_layer.T
                u, s, e = torch.svd(hessian)
                v = client_grads_layer.T @ e  # (d*m)*(m*m) = (d*m)
                v = v.T  # (m*d)
                k = int(len(s) * k_proportion)
                w = v[:k]
                for j in range(k):
                    num_pos = 0
                    num_neg = 0
                    for i in range(client_grads_layer.size(0)):
                        if torch.dot(client_grads_layer[i], w[j]) >= 0:
                            num_pos += 1
                        else:
                            num_neg += 1
                    if num_pos < num_neg:
                        w[j] = -w[j]
                grad_agg = []
                for i in range(client_num):
                    grad_pc = client_grads_layer[i]
                    grad_revise = torch.zeros_like(grad_pc)
                    for j in range(k):
                        grad_revise_j = torch.dot(grad_pc, w[j])/torch.dot(w[j], w[j]) * w[j]
                        grad_revise_j = grad_revise_j * \
                            s[j]/s.sum() * torch.norm(grad_pc) / torch.norm(grad_revise_j)
                        grad_revise += grad_revise_j
                    grad_agg.append(grad_revise)
                grad_new = torch.mean(torch.stack(grad_agg), dim=0, keepdim=True)
                grad_new = torch.squeeze(grad_new)
                if g_len == 1:
                    grad_new = torch.squeeze(naive_avg_grad)
                    grad_new = torch.unsqueeze(grad_new, 0)
            else:
                client_grads_layer = client_grads_[:, start:end]
                naive_avg_grad = torch.mean(client_grads_layer, dim=0, keepdim=True)
                grad_new = naive_avg_grad
                grad_new = torch.squeeze(grad_new)
                if g_len == 1:
                    grad_new = torch.squeeze(naive_avg_grad)
                    grad_new = torch.unsqueeze(grad_new, 0)
            gamma = 0.99
            grad_history[key] = gamma * grad_history[key] + (1 - gamma) * grad_new
            grads.append(grad_new)
        else:
            grad_new = client_grads_[:, start:end].mean(0)
            grad_history[key] = grad_new
            grads.append(grad_new)
        start = end
    grad_new = torch.cat(grads)

    return grad_new, grad_history


class FedLDServer(Server):

    def __init__(self,
                 model: torch.nn.Module,
                 test_set: FastDataLoader,
                 clients: Collection[Client],
                 weighted: bool = False,
                 lr: float = 1.0,
                 k_proportion: float = 0.8,
                 **kwargs):
        super().__init__(model=model,
                         test_set=test_set,
                         clients=clients,
                         weighted=weighted,
                         lr=lr,
                         **kwargs)
        self.hyper_params.update(k_proportion=k_proportion)
        self.grad_history = {key: None for key in model.state_dict().keys()}
        self.grad_history['grad_len'] = {key: value.numel()
                                         for key, value in model.state_dict().items()}

    @torch.no_grad()
    def aggregate(self, eligible: Collection[Client], client_models: Collection[Module]) -> None:
        client_models = list(client_models)
        local_clients_grads = [_get_client_grads(c_model, self.model) for c_model in client_models]
        grad_new, self.grad_history = pcgrad_svd(local_clients_grads,
                                                 self.grad_history,
                                                 self.hyper_params.k_proportion)

        for c_model in client_models:
            c_model = _set_client_grads(c_model, self.model, grad_new)

        return super().aggregate(eligible, client_models)


class FedLD(CentralizedFL):

    def get_client_class(self):
        return FedLDClient

    def get_server_class(self):
        return FedLDServer
