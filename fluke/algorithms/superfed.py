from torch.nn.modules import Module
from typing import Any, Callable
import numpy as np
from copy import deepcopy
import sys
sys.path.append(".")
sys.path.append("..")

from ..algorithms import PersonalizedFL  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import (get_global_model_dict,  # NOQA
                           get_local_model_dict,  # NOQA
                           mix_networks,  # NOQA
                           set_lambda_model)  # NOQA
from ..data import FastDataLoader  # NOQA
from ..client import PFLClient  # NOQA


class SuPerFedClient(PFLClient):

    def __init__(self,
                 index: int,
                 model: Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable[..., Any],
                 local_epochs: int = 3,
                 mode: str = "global",
                 start_mix: int = 10,
                 mu: float = 0.1,
                 nu: float = 0.1):
        assert mode in ["mm", "lm"]

        super().__init__(index, model, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
        self.hyper_params.update(
            mode=mode,
            start_mix=start_mix,
            mu=mu,
            nu=nu
        )

        self.internal_model = None
        self.mixed = False

    def fit(self, override_local_epochs: int = 0) -> dict:
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()

        if self.hyper_params.mu > 0:
            prev_global_model = deepcopy(self.model)
            for param in prev_global_model.parameters():
                param.requires_grad = False

        if self.server.rounds >= self.hyper_params.start_mix:
            if not self.mixed:
                self.internal_model = mix_networks(
                    self.model, self.personalized_model, 0)  # temporary lambda
                # Once the mixing starts, the optimizer and scheduler are re-initialized
                # This is not what the original code does
                self.optimizer, self.scheduler = self.optimizer_cfg(self.internal_model)
                self.mixed = True
            else:
                local_dict = {k + "_local": v for k,
                              v in self.personalized_model.state_dict().items()}
                self.internal_model.load_state_dict({**self.model.state_dict(), **local_dict})
        else:
            self.internal_model = self.model
            self.personalized_model.load_state_dict(self.model.state_dict())

        self.internal_model.train()
        self.internal_model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.internal_model)

        for _ in range(epochs):
            loss = None
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)

                if self.server.rounds >= self.hyper_params.start_mix:
                    if self.hyper_params.mode == "mm":
                        set_lambda_model(self.internal_model, np.random.uniform(0.0, 1.0))
                    elif self.hyper_params.mode == "lm":
                        set_lambda_model(self.internal_model, None, layerwise=True)

                self.optimizer.zero_grad()
                y_hat = self.internal_model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)

                if self.hyper_params.mu > 0:
                    prox_term = 0.0
                    for name, param in self.internal_model.named_parameters():
                        if '_local' not in name:
                            continue
                        internal_param = self.internal_model.get_parameter(
                            name.replace('_local', ''))
                        prev_param = prev_global_model.get_parameter(name.replace('_local', ''))
                        prox_term += (internal_param - prev_param).norm(2)
                    loss += self.hyper_params.mu * prox_term

                if self.server.rounds >= self.hyper_params.start_mix:
                    numerator, norm_1, norm_2 = 0, 0, 0
                    for name, param_l in self.internal_model.named_parameters():
                        if '_local' not in name:
                            continue
                        param_g = self.internal_model.get_parameter(name.replace('_local', ''))
                        numerator += (param_g * param_l).add(1e-6).sum()
                        norm_1 += param_g.pow(2).sum()
                        norm_2 += param_l.pow(2).sum()
                    cos_sim = numerator.pow(2).div(norm_1 * norm_2)
                    loss += self.hyper_params.nu * cos_sim

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        self.internal_model.to("cpu")
        clear_cache()

        if self.server.rounds >= self.hyper_params.start_mix:
            self.personalized_model.load_state_dict(get_local_model_dict(self.internal_model))
            self.model.load_state_dict(get_global_model_dict(self.internal_model))
        else:
            self.model.load_state_dict(self.internal_model.state_dict())
            self.personalized_model.load_state_dict(self.internal_model.state_dict())

        self.send_model()


class SuPerFed(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return SuPerFedClient
