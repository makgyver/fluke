"""Implementation of the [FedAwS]_ algorithm.

References:
    .. [FedAwS] TODO
"""
import sys
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F
sys.path.append(".")
sys.path.append("..")

from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from ..client import Client  # NOQA
from . import CentralizedFL  # NOQA


class SpreadModel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = nn.Parameter(weights)

    def forward(self):
        return F.normalize(self.weights, dim=1)


class SpreadLoss(nn.Module):

    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, weights: torch.Tensor):
        ws_norm = F.normalize(weights, dim=1)
        cos_dis = 0.5 * (1.0 - torch.mm(ws_norm, ws_norm.transpose(0, 1)))

        d_mat = torch.diag(torch.ones(weights.shape[0]))
        d_mat = d_mat.to(weights.device)

        cos_dis = cos_dis * (1.0 - d_mat)

        indx = ((self.margin - cos_dis) > 0.0).float()
        loss = (((self.margin - cos_dis) * indx) ** 2).mean()
        return loss


class FedAwSServer(Server):

    def __init__(self,
                 model: torch.nn.Module,
                 test_data: FastDataLoader,
                 clients: Iterable[Client],
                 eval_every: int = 1,
                 weighted: bool = False,
                 aws_lr: float = 0.1,
                 aws_steps: int = 100,
                 margin: float = 0.5,
                 last_layer_name: str = "classifier",
                 **kwargs):
        super().__init__(model=model, test_data=test_data,
                         clients=clients, eval_every=eval_every, weighted=weighted)
        print(model.state_dict().keys())
        assert (last_layer_name + ".weight") in model.state_dict().keys(), \
            f"Invalid last_layer_name: {last_layer_name}. Make sure that the last layer \
                is named as {last_layer_name}"
        self.hyper_params.update(aws_lr=aws_lr,
                                 margin=margin,
                                 aws_steps=aws_steps,
                                 last_layer_name=last_layer_name + ".weight")

    def _compute_spreadout(self):
        ws = self.model.state_dict()[self.hyper_params.last_layer_name].data
        spread_model = SpreadModel(ws)

        optimizer = torch.optim.SGD(
            spread_model.parameters(),
            lr=self.hyper_params.aws_lr,
            momentum=0.9
        )
        criterion = SpreadLoss(margin=self.hyper_params.margin)

        for _ in range(self.hyper_params.aws_steps):
            optimizer.zero_grad()
            loss = criterion(spread_model.forward())
            loss.backward()
            optimizer.step()

        self.model.load_state_dict(
            {self.hyper_params.last_layer_name: spread_model.weights.data},
            strict=False
        )

    def aggregate(self, eligible: Iterable[Client]) -> None:
        super().aggregate(eligible)

        self._compute_spreadout()


class FedAwS(CentralizedFL):

    def get_server_class(self) -> Server:
        return FedAwSServer