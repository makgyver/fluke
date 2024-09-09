"""Implementation of the CCVR [CCVR21]_ algorithm.

References:
    .. [CCVR21] Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, Jiashi Feng. No Fear of
       Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data. In NeurIPS
       (2021). URL: https://arxiv.org/abs/2106.05001
"""
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.modules import Module

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..comm import Message  # NOQA
from ..data import FastDataLoader  # NOQA
from ..server import Server  # NOQA
from . import CentralizedFL  # NOQA

__all__ = [
    "CCVRClient",
    "CCVRServer",
    "CCVR"
]


class CCVRClient(Client):
    @torch.no_grad()
    def compute_mean_cov(self) -> None:
        """Computes the label-wise mean and covariance of the data. After the computation, the
        client send (through the channel) a message to the server containing the computed values and
        the number of examples per class.
        """
        self.model.to(self.device)
        list_z, list_y = [], []
        for _, (X, y) in enumerate(self.train_set):
            X, y = X.to(self.device), y.to(self.device)
            Z = self.model.forward_encoder(X)
            list_z.append(Z)
            list_y.append(y)

        Z = torch.cat(list_z, dim=0)
        Y = torch.cat(list_y, dim=0)
        n_feats = Z.shape[-1]

        classes_mean = []
        classes_cov = []
        ex_x_class = []
        for c in range(self.train_set.num_labels):
            idx = torch.where(Y == c)[0]
            if idx.shape[0] == 0:
                classes_mean.append(torch.zeros(n_feats))
                classes_cov.append(torch.zeros(n_feats, n_feats))
                ex_x_class.append(0)
                continue

            Z_c = Z[idx]
            mean_c = Z_c.mean(dim=0)
            cov_c = Z_c.t().cov(correction=0)
            classes_mean.append(mean_c.to("cpu"))
            classes_cov.append(cov_c.to("cpu"))
            ex_x_class.append(Z_c.size(0))

        payload = (classes_mean, classes_cov, ex_x_class)
        self.model.to("cpu")
        self.channel.send(Message(payload, "mean_cov", self), self.server)


class CCVRServer(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False,
                 lr: float = 0.1,
                 batch_size: int = 64,
                 sample_per_class: int = 100,
                 **kwargs: dict[str, Any]):
        super().__init__(model=model, test_set=test_set, clients=clients, weighted=weighted)
        self.hyper_params.update(
            lr=lr,
            batch_size=batch_size,
            sample_per_class=sample_per_class
        )

    def _compute_mean_cov(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        means, covs, ns = [], [], []
        for client in self.clients:
            client.receive_model()
            client.compute_mean_cov()
            mean, cov, n = self.channel.receive(self, client, msg_type="mean_cov").payload
            means.append(mean)
            covs.append(cov)
            ns.append(n)

        num_classes = len(means[0])
        classes_mean = [None for _ in range(num_classes)]
        ex_x_class = [sum(n) for n in zip(*ns)]

        # loop over classes
        for c, (mu, n) in enumerate(zip(zip(*means), zip(*ns))):
            if ex_x_class[c] > 0:
                classes_mean[c] = torch.sum(torch.stack(
                    mu) * torch.tensor(n).reshape(-1, 1), dim=0) / ex_x_class[c]

        classes_cov = [None for _ in range(num_classes)]
        for c in range(num_classes):
            if ex_x_class[c] > 1:
                for k in range(self.n_clients):
                    if classes_cov[c] is None:
                        classes_cov[c] = torch.zeros_like(covs[k][c])

                    classes_cov[c] += ((ns[k][c] - 1) / (ex_x_class[c] - 1)
                                       ) * covs[k][c] + (ns[k][c] / (ex_x_class[c] - 1)) * (
                        torch.outer(means[k][c], means[k][c])
                    )

                classes_cov[c] -= (ex_x_class[c] / (ex_x_class[c] - 1)) * (
                    torch.outer(classes_mean[c], classes_mean[c])
                )

        return classes_mean, classes_cov

    def _generate_virtual_repr(self,
                               classes_mean: Iterable[torch.Tensor],
                               classes_cov: Iterable[torch.Tensor]) -> tuple[torch.Tensor,
                                                                             torch.Tensor]:
        data, targets = [], []
        for c, (mean, cov) in enumerate(zip(classes_mean, classes_cov)):
            if mean is not None and cov is not None:
                samples = np.random.multivariate_normal(
                    mean.cpu().numpy(),
                    cov.cpu().numpy(),
                    self.hyper_params.sample_per_class,
                )
                data.append(torch.tensor(samples, dtype=torch.float))
                targets.append(
                    torch.ones(
                        self.hyper_params.sample_per_class,
                        dtype=torch.long
                    ) * c
                )

        data = torch.cat(data)
        targets = torch.cat(targets)
        return data, targets

    def _calibrate(self, Z_train: torch.FloatTensor, y_train: torch.LongTensor) -> None:
        self.model.train()
        self.model.to(self.device)

        # FIXME: loss, optimizer and scheduler are fixed for now
        optimizer = torch.optim.SGD(self.model.head.parameters(), lr=self.hyper_params.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        train_set = FastDataLoader(Z_train,
                                   y_train,
                                   num_labels=len(set(y_train.cpu().numpy())),
                                   batch_size=self.hyper_params.batch_size,
                                   shuffle=True)

        for Z, y in train_set:
            Z, y = Z.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_hat = self.model.forward_head(Z)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
        self.model.to("cpu")

    def finalize(self) -> None:
        """In the CCVR Server, at the end of the learning the model is calibrated according to the
        data distributions of the clients.
        """
        self.broadcast_model(self.clients)
        classes_mean, classes_cov = self._compute_mean_cov()
        Z, y = self._generate_virtual_repr(classes_mean, classes_cov)
        self._calibrate(Z, y)
        super().finalize()


class CCVR(CentralizedFL):

    def get_client_class(self) -> Client:
        return CCVRClient

    def get_server_class(self) -> Server:
        return CCVRServer
