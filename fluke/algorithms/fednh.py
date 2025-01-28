
"""Implementation of the [FedNH23]_ algorithm.

References:
    .. [FedNH23] Yutong Dai, Zeyuan Chen, Junnan Li, Shelby Heinecke, Lichao Sun, Ran Xu.
       Tackling Data Heterogeneity in Federated Learning with Class Prototypes. In AAAI (2023).
       URL: https://arxiv.org/abs/2212.02758
"""
import sys
from collections import OrderedDict, defaultdict
from typing import Any, Iterable

import torch
from torch.nn import CrossEntropyLoss, Module, Parameter
from torch.nn import functional as F

sys.path.append(".")
sys.path.append("..")


from .. import GlobalSettings  # NOQA
from ..client import PFLClient  # NOQA
from ..data import FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..server import Server  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import (STATE_DICT_KEYS_TO_IGNORE,  # NOQA
                           get_activation_size, ArgMaxModule)  # NOQA
from . import PersonalizedFL  # NOQA

__all__ = [
    "ProtoNet",
    "FedNHClient",
    "FedNHServer",
    "FedNH"
]


class ProtoNet(Module):
    """Wrapper network for the encoder model and the prototypes.

    Args:
        encoder (nn.Module): The encoder model.
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


class FedNHClient(PFLClient):

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
                 **kwargs: dict[str, Any]):
        fine_tuning_epochs = fine_tuning_epochs if fine_tuning_epochs > 0 else local_epochs
        embedding_size = get_activation_size(model, train_set.tensors[0][0])
        super().__init__(index=index, model=ProtoNet(model, n_protos, embedding_size),
                         train_set=train_set, test_set=test_set, optimizer_cfg=optimizer_cfg,
                         loss_fn=CrossEntropyLoss(), local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, **kwargs)
        self.hyper_params.update(
            n_protos=n_protos
        )
        self.model = ProtoNet(model, n_protos, embedding_size)
        self.count_by_class = torch.bincount(self.train_set.tensors[1],
                                             minlength=self.train_set.num_labels)
        self._tounload.remove("personalized_model")
        self._unload_model()

    def _update_protos(self, protos: Iterable[torch.Tensor]) -> None:
        prototypes = self.model.prototypes.data
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
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                _, logits = self.model(X)
                loss = self.hyper_params.loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(
                    lambda p: p.requires_grad, self.model.parameters()
                ), max_norm=10)
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()
        self.model.to("cpu")
        clear_cache()
        running_loss /= (epochs * len(self.train_set))

        protos = defaultdict(list)
        for label in range(self.hyper_params.n_protos):
            Xlbl = self.train_set.tensors[0][self.train_set.tensors[1] == label]
            protos[label] = self.model.encoder(Xlbl).detach().data

        self._update_protos(protos)
        return running_loss

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if test_set is not None and self.model is not None:
            model = ArgMaxModule(self.model)
            return evaluator.evaluate(self._last_round, model, test_set, device=self.device)
        return {}

    def finalize(self) -> None:
        self._load_model()
        self.fit(self.hyper_params.fine_tuning_epochs)
        metrics = self.evaluate(GlobalSettings().get_evaluator(), self.test_set)
        if metrics:
            self._notify_evaluation(-1, "post-fit", metrics)
        self._unload_model()


class FedNHServer(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Iterable[PFLClient],
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
    def aggregate(self, eligible: Iterable[PFLClient], client_models: Iterable[Module]) -> None:
        # Recieve models from clients, i.e., the prototypes
        clients_protos = [cmodel.prototypes.data for cmodel in client_models]

        # Group by label
        label_protos = {i: [protos[i] for protos in clients_protos]
                        for i in range(self.hyper_params.n_protos)}

        # This could be the learning rate for the server (not used in the official implementation)
        # server_lr = self.hyper_params.lr * self.hyper_params.lr_decay ** self.round
        server_lr = 1.0
        weight = server_lr / len(client_models)
        cl_weight = torch.zeros(self.hyper_params.n_protos)

        # To get client.count_by_class is actually illegal in fluke, but irrelevant from an
        # implementation point of view
        for client in eligible:
            cl_weight += client.count_by_class

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
        # Aggregate models = Federated Averaging
        avg_model_sd = OrderedDict()
        clients_sd = [client.encoder.state_dict() for client in client_models]
        with torch.no_grad():
            for key in self.model.encoder.state_dict().keys():
                if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                    avg_model_sd[key] = self.model.encoder.state_dict()[key].clone()
                    continue

                if key.endswith("num_batches_tracked"):
                    mean_nbt = torch.mean(torch.Tensor([c[key] for c in clients_sd])).long()
                    avg_model_sd[key] = max(avg_model_sd[key], mean_nbt)
                    continue

                for _, client_sd in enumerate(clients_sd):
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weight * client_sd[key].clone()
                    else:
                        avg_model_sd[key] += weight * client_sd[key].clone()
            self.model.encoder.load_state_dict(avg_model_sd)

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if self.test_set is not None:
            model = ArgMaxModule(self.model)
            return evaluator.evaluate(self.rounds + 1, model, self.test_set, device=self.device)
        return {}


class FedNH(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedNHClient

    def get_server_class(self) -> Server:
        return FedNHServer
