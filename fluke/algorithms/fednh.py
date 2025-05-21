
"""Implementation of the [FedNH23]_ algorithm.

References:
    .. [FedNH23] Yutong Dai, Zeyuan Chen, Junnan Li, Shelby Heinecke, Lichao Sun, Ran Xu.
       Tackling Data Heterogeneity in Federated Learning with Class Prototypes. In AAAI (2023).
       URL: https://arxiv.org/abs/2212.02758
"""
import sys
from collections import defaultdict
from typing import Collection

import torch
from torch.nn import CrossEntropyLoss, Module, Parameter
from torch.nn import functional as F

sys.path.append(".")
sys.path.append("..")


from .. import FlukeENV  # NOQA
from ..client import Client  # NOQA
from ..config import OptimizerConfigurator  # NOQA
from ..data import FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..server import Server  # NOQA
from ..utils import clear_cuda_cache, get_model  # NOQA
from ..utils.model import get_activation_size, unwrap  # NOQA
from . import CentralizedFL  # NOQA

__all__ = [
    "ProtoNet",
    "ArgMaxModule",
    "FedNHClient",
    "FedNHServer",
    "FedNH"
]


class ProtoNet(Module):
    """Neural network with prototypes.
    The network is composed of an encoder and a set of prototypes. The prototypes are
    initialized as orthogonal vectors and are not trainable. In the forward pass, the network
    computes the cosine similarity between the normalized embeddings (output of the encoder) and
    the prototypes.

    Args:
        encoder (Module): Encoder network.
        n_protos (int): Number of prototypes.
        proto_size (int): Size of the prototypes.
    """

    def __init__(self, encoder: Module, n_protos: int, proto_size: int):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.prototypes = Parameter(torch.rand((n_protos, proto_size)), requires_grad=False)
        self.prototypes.data = torch.nn.init.orthogonal_(torch.rand(n_protos, proto_size))
        self.temperature = Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        embeddings = torch.div(embeddings, embeddings_norm)
        logits = torch.matmul(embeddings, self.prototypes.T)
        logits = self.temperature * logits
        return embeddings, logits


class ArgMaxModule(Module):
    def __init__(self,
                 model: Module):
        super().__init__()
        self.model: Module = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.model(x)[1], dim=1)


class FedNHClient(Client):

    def __init__(self,
                 index: int,
                 model: Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,  # Not used
                 local_epochs: int,
                 n_protos: int,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 5,
                 **kwargs):
        fine_tuning_epochs = fine_tuning_epochs if fine_tuning_epochs > 0 else local_epochs
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=CrossEntropyLoss(),
                         local_epochs=local_epochs, fine_tuning_epochs=fine_tuning_epochs,
                         clipping=clipping, **kwargs)
        self.hyper_params.update(
            n_protos=n_protos
        )
        if isinstance(model, str):
            model = get_model(model)
        embedding_size = get_activation_size(model, train_set.tensors[0][0])
        self.model = ProtoNet(model, n_protos, embedding_size)
        self.count_by_class = torch.bincount(self.train_set.tensors[1],
                                             minlength=self.train_set.num_labels)
        self._save_to_cache()

    def _update_protos(self, protos: Collection[torch.Tensor]) -> None:
        prototypes = unwrap(self.model).prototypes.data
        for label, prts in protos.items():
            if prts.shape[0] > 0:
                prototypes[label] = torch.sum(prts, dim=0) / prts.shape[0]
                prototypes[label] /= torch.norm(prototypes[label]).clamp(min=1e-12)
                prototypes[label] = prototypes[label] * prts.shape[0]
            else:
                prototypes[label] = torch.zeros_like(prototypes[label])

    def fit(self, override_local_epochs: int = 0) -> float:

        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                _, logits = self.model(X)
                loss = self.hyper_params.loss_fn(logits, y)
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()
        self.model.cpu()
        clear_cuda_cache()
        running_loss /= (epochs * len(self.train_set))

        protos = defaultdict(list)
        for label in range(self.hyper_params.n_protos):
            Xlbl = self.train_set.tensors[0][self.train_set.tensors[1] == label]
            protos[label] = unwrap(self.model).encoder(Xlbl).detach().data

        self._update_protos(protos)
        return running_loss

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if test_set is not None and self.model is not None:
            model = ArgMaxModule(self.model)
            return evaluator.evaluate(self._last_round, model, test_set, device=self.device)
        return {}

    def finalize(self) -> None:
        self._load_from_cache()
        self.fit(self.hyper_params.fine_tuning_epochs)
        metrics = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
        if metrics:
            self.notify(event="client_evaluation", round=-1,
                        client_id=self.index, phase="post-fit", evals=metrics)
        self._save_to_cache()


class FedNHServer(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Collection[Client],
                 weighted: bool = True,
                 n_protos: int = 10,
                 rho: float = 0.1):
        embedding_size = get_activation_size(model, clients[0].train_set.tensors[0][0])
        super().__init__(model=ProtoNet(model, n_protos, embedding_size),
                         test_set=test_set,
                         clients=clients,
                         weighted=weighted)
        self.hyper_params.update(n_protos=n_protos, rho=rho)

    @torch.no_grad()
    def aggregate(self, eligible: Collection[Client], client_models: Collection[Module]) -> None:

        # This could be the learning rate for the server (not used in the official implementation)
        # server_lr = self.hyper_params.lr * self.hyper_params.lr_decay ** self.round
        server_lr = 1.0
        weights = [server_lr / len(eligible)] * len(eligible)
        cl_weight = torch.zeros(self.hyper_params.n_protos)

        # To get client.count_by_class is actually illegal in fluke, but irrelevant from an
        # implementation point of view
        for client in eligible:
            cl_weight += client.count_by_class

        clients_protos = []

        # Aggregate models = Federated Averaging
        # Get model parameters and buffers
        model_params = dict(self.model.encoder.named_parameters())
        # Includes running_mean, running_var, etc.
        model_buffers = dict(self.model.encoder.named_buffers())

        # Initialize accumulators for parameters
        avg_params = {key: torch.zeros_like(param.data) for key, param in model_params.items()}
        avg_buffers = {key: torch.zeros_like(buffer.data)
                       for key, buffer in model_buffers.items() if "num_batches_tracked" not in key}

        max_num_batches_tracked = 0  # Track the max num_batches_tracked

        # Compute weighted sum (weights already sum to 1, so no division needed)
        for m, w in zip(client_models, weights):
            clients_protos.append(m.prototypes.data)

            for key, param in m.encoder.named_parameters():
                avg_params[key].add_(param.data, alpha=w)

            for key, buffer in m.encoder.named_buffers():
                if "num_batches_tracked" not in key:
                    avg_buffers[key].add_(buffer.data, alpha=w)
                else:
                    max_num_batches_tracked = max(max_num_batches_tracked, buffer.item())

        for key in model_params.keys():
            model_params[key].data.lerp_(avg_params[key], server_lr)  # Soft update

        for key in model_buffers.keys():
            if "num_batches_tracked" not in key:
                model_buffers[key].data.lerp_(avg_buffers[key], server_lr)

        # Assign max num_batches_tracked
        for key in model_buffers.keys():
            if "num_batches_tracked" in key:
                model_buffers[key].data.fill_(max_num_batches_tracked)

        # Group by label
        label_protos = {i: [protos[i] for protos in clients_protos]
                        for i in range(self.hyper_params.n_protos)}

        # Aggregate prototypes
        prototypes = self.model.prototypes.clone()
        if self.hyper_params.weighted:
            for label, protos in label_protos.items():
                prototypes.data[label, :] = torch.sum(torch.stack(protos) / cl_weight[label], dim=0)
        else:
            sim_weights = []
            for protos in clients_protos:
                sim_weights.append(torch.exp(torch.sum(prototypes.data * protos, dim=1)))
            sim_weights = torch.stack(sim_weights, dim=0).T

            for label, protos in label_protos.items():
                prototypes.data[label, :] = prototypes.data[label, :] + \
                    torch.sum(sim_weights[label].unsqueeze(1) * torch.stack(protos), dim=0)

            prototypes.data /= torch.sum(sim_weights, dim=1).unsqueeze(1)

        # Normalize the prototypes
        # prototypes.data /= torch.norm(prototypes.data, dim=0).clamp(min=1e-12)
        prototypes.data = F.normalize(prototypes.data, dim=1)  # .clamp(min=1e-12)

        self.model.prototypes.data = (1 - self.hyper_params.rho) * prototypes.data + \
            self.hyper_params.rho * self.model.prototypes.data

        # Normalize the prototypes again
        self.model.prototypes.data = F.normalize(
            self.model.prototypes.data, dim=1)  # .clamp(min=1e-12)

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if self.test_set is not None:
            model = ArgMaxModule(self.model)
            return evaluator.evaluate(self.rounds + 1, model, self.test_set, device=self.device)
        return {}


class FedNH(CentralizedFL):

    def get_client_class(self) -> type[Client]:
        return FedNHClient

    def get_server_class(self) -> type[Server]:
        return FedNHServer
