from collections import OrderedDict
import torch
from torch.nn import Module, Parameter, CrossEntropyLoss
from typing import Sequence, Callable
from collections import defaultdict
import sys

sys.path.append(".")
sys.path.append("..")

from ..evaluation import ClassificationEval  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import STATE_DICT_KEYS_TO_IGNORE  # NOQA
from ..data import FastDataLoader  # NOQA
from ..client import PFLClient  # NOQA
from ..server import Server  # NOQA
from . import PersonalizedFL  # NOQA


class ProtoNet(Module):

    def __init__(self, encoder: Module, n_protos: int, normalize: bool = False):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self._normalize = normalize
        self.prototypes = Parameter(torch.rand((n_protos, encoder.output_size)),
                                    requires_grad=False)
        self.prototypes.data = torch.nn.init.orthogonal_(torch.rand(n_protos,
                                                                    encoder.output_size))
        self.temperature = Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        if self._normalize:
            embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            embeddings = torch.div(embeddings, embeddings_norm)
            prototype_norm = torch.norm(self.prototypes, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototypes = torch.div(self.prototypes, prototype_norm)
            logits = torch.matmul(embeddings, normalized_prototypes.T)
        else:
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


class FedNHClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,  # Not used
                 local_epochs: int,
                 n_protos: int,
                 proto_norm: bool = True):
        super().__init__(index,
                         ProtoNet(model, n_protos, proto_norm),
                         train_set,
                         test_set,
                         optimizer_cfg,
                         CrossEntropyLoss(),
                         local_epochs)
        self.hyper_params.update(
            n_protos=n_protos,
            proto_norm=proto_norm
        )
        self.model = self.personalized_model

    def _update_protos(self, protos: Sequence[torch.Tensor]) -> None:
        for label, prts in protos.items():
            if prts.shape[0] > 0:
                self.model.prototypes.data[label] = torch.sum(prts, dim=0) / prts.shape[0]
                self.model.prototypes.data[label] /= torch.norm(
                    self.model.prototypes.data[label]).clamp(min=1e-12)
            else:
                self.model.prototypes.data[label] = torch.zeros_like(
                    self.model.prototypes.data[label])

    def fit(self, override_local_epochs: int = 0) -> None:
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)
        self.receive_model()
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                _, logits = self.model(X)
                loss = self.hyper_params.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        self.model.to("cpu")
        clear_cache()

        protos = defaultdict(list)
        for label in range(self.hyper_params.n_protos):
            Xlbl = self.train_set.tensors[0][self.train_set.tensors[1] == label]
            protos[label] = self.model(Xlbl)[0].detach().data

        self._update_protos(protos)
        self.send_model()

    def evaluate(self) -> dict[str, float]:
        if self.test_set is not None and self.model is not None:
            model = ArgMaxModule(self.model)
            return ClassificationEval(None,
                                      self.hyper_params.n_protos,
                                      self.device).evaluate(model,
                                                            self.test_set)
        return {}

    def finalize(self) -> None:
        self.fit()


class FedNHServer(Server):

    def __init__(self,
                 model: Module,
                 test_data: FastDataLoader,
                 clients: Sequence[PFLClient],
                 eval_every: int = 1,
                 weighted: bool = True,
                 n_protos: int = 10,
                 rho: float = 0.1,
                 proto_norm: bool = True):
        super().__init__(ProtoNet(model, n_protos, proto_norm),
                         test_data,
                         clients,
                         eval_every,
                         weighted)
        self.hyper_params.update(n_protos=n_protos, rho=rho, proto_norm=proto_norm)

    @torch.no_grad()
    def aggregate(self, eligible: Sequence[PFLClient]) -> None:
        # Recieve models from clients, i.e., the prototypes
        clients_models = self.get_client_models(eligible, state_dict=False)
        clients_protos = [cmodel.prototypes.data for cmodel in clients_models]

        # Group by label
        label_protos = {i: [protos[i] for protos in clients_protos]
                        for i in range(self.hyper_params.n_protos)}

        # This could be the learning rate for the server (not used in the official implementation)
        # server_lr = self.hyper_params.lr * self.hyper_params.lr_decay ** self.round
        server_lr = 1.0
        weight = server_lr / len(clients_models)

        # Aggregate prototypes
        prototypes = self.model.prototypes.clone()
        if self.hyper_params.weighted:
            for label, protos in label_protos.items():
                prototypes.data[label, :] = torch.sum(
                    weight * torch.stack(protos), dim=0)
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
        prototypes.data /= torch.norm(prototypes.data, dim=0).clamp(min=1e-12)

        self.model.prototypes.data = (1 - self.hyper_params.rho) * prototypes.data + \
            self.hyper_params.rho * self.model.prototypes.data

        # Normalize the prototypes again
        self.model.prototypes.data /= torch.norm(self.model.prototypes.data,
                                                 dim=0).clamp(min=1e-12)

        # Aggregate models = Federated Averaging
        avg_model_sd = OrderedDict()
        clients_sd = [client.encoder.state_dict() for client in clients_models]
        with torch.no_grad():
            for key in self.model.encoder.state_dict().keys():
                if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                    avg_model_sd[key] = self.model.encoder.state_dict()[key].clone()
                    continue
                for _, client_sd in enumerate(clients_sd):
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weight * client_sd[key].clone()
                    else:
                        avg_model_sd[key] += weight * client_sd[key].clone()
            self.model.encoder.load_state_dict(avg_model_sd)

    def evaluate(self) -> dict[str, float]:
        if self.test_data is not None:
            model = ArgMaxModule(self.model)
            return ClassificationEval(None,
                                      self.hyper_params.n_protos,
                                      self.device).evaluate(model,
                                                            self.test_data)
        return {}


class FedNH(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedNHClient

    def get_server_class(self) -> Server:
        return FedNHServer
