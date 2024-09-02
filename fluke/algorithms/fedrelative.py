from torch import nn
import torch
import torch.nn.functional as F
from typing import Iterable
import sys
sys.path.append(".")
sys.path.append("..")

from . import CentralizedFL  # NOQA
from ..nets import EncoderHeadNet  # NOQA
from ..client import Client  # NOQA
from ..server import Server  # NOQA
from ..data import FastDataLoader  # NOQA


class RelativeProjectionModel(nn.Module):

    def __init__(self, model: EncoderHeadNet, anchors: torch.Tensor):
        super().__init__()
        self.model = model
        self.anchors = anchors
        self.relative_linear = nn.Linear(
            self.model.encoder.output_size, anchors.shape[0], bias=False)

        right_shape = None
        for layer in self.model.head.children():
            if isinstance(layer, nn.Linear):
                right_shape = layer.state_dict()['weight'].shape[1]
                break

        if right_shape is None:
            raise ValueError("No linear layer found in the head of the model.")

        self.bridge_layer = nn.Linear(
            anchors.shape[0], right_shape, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Z = self.model.encoder(x)
        Z = F.normalize(Z, p=2, dim=0)
        with torch.no_grad():
            H = self.model.encoder(self.anchors)
            H = F.normalize(H, p=2, dim=0)
            self.relative_linear.weight.copy_(H)

        dotZH = self.relative_linear(Z)
        x = self.bridge_layer(dotZH)
        return self.model.head(x)

    def to(self, *args, **kwargs):
        self.anchors = self.anchors.to(*args, **kwargs)
        return super().to(*args, **kwargs)


# class FedRelClient(Client):

#     def receive_model(self) -> None:
#         msg = self.channel.receive(self, self.server, msg_type="model")
#         if self.model is None:
#             self.model = msg.payload
#         else:
#             self.model.model.head.load_state_dict(msg.payload.model.head.state_dict())


class FedRelativeServer(Server):
    def __init__(self,
                 model: torch.nn.Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 n_anchors_class: int = 5,
                 weighted: bool = True):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.hyper_params.update(n_anchors_class=n_anchors_class)

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1, finalize: bool = True) -> None:
        if self.rounds == 0:
            # Select random anchors from test set. n_anchors per class
            X, y = self.test_set.tensors
            anch_X, anch_y = None, None
            for c in set(y.tolist()):
                idx = torch.where(y == c)[0]
                idx = idx[torch.randperm(idx.size(0))[:self.hyper_params.n_anchors_class]]
                if anch_X is None:
                    anch_X, anch_y = X[idx], y[idx]
                else:
                    anch_X = torch.cat((anch_X, X[idx]), dim=0)
                    anch_y = torch.cat((anch_y, y[idx]), dim=0)

            self.anchors = anch_X
            # Anchors are sent with the model
            self.model = RelativeProjectionModel(self.model, self.anchors)
        super().fit(n_rounds=n_rounds, eligible_perc=eligible_perc, finalize=finalize)


class FedRelative(CentralizedFL):

    # def get_client_class(self) -> Client:
    #     return FedRelClient

    def get_server_class(self) -> Server:
        return FedRelativeServer
