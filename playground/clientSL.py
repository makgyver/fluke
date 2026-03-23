from typing import Any, Generator

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from fluke.client import Client
from fluke.comm import Message
from fluke.config import OptimizerConfigurator
from fluke.data import FastDataLoader
from fluke.utils import clear_cuda_cache
from fluke.utils.model import safe_load_state_dict


class ClientSL(Client):
    def __init__(
        self,
        index: int,
        train_set: FastDataLoader | DataLoader,
        test_set: FastDataLoader | DataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,
        local_epochs: int = 1,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        persistency: bool = True,
        **kwargs,
    ):
        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss_fn,   # gestire quetsa loss_fn inutile (in vanillaSL)
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            persistency=persistency,
            **kwargs,
        )
        self.n_batches = 0
        self.running_loss = 0.0
        self.local_smashed = None

    def receive_client_model(self) -> None:
        msg = self.channel.receive(self.index, "server", msg_type="client_model")

        if self.model is None:
            self.model = msg.payload
        else:
            safe_load_state_dict(self.model, msg.payload.state_dict())

    def receive_gradients(self) -> tuple[torch.Tensor, float]:
        msg = self.channel.receive(self.index, "server", msg_type="gradients")
        grad_cut, loss = msg.payload
        return grad_cut, loss


    def send_client_model(self) -> None: #centralized
        self.channel.send(Message(self.model, "client_model", self.index, inmemory=True),"server")

    def send_smashed_data(self, smashed_data, y) -> None: #centralized
        self.channel.send(Message((smashed_data, y), "client_smashed_data", self.index, inmemory=True),"server")

    def start_training(self, current_round: int) -> Generator:
        self.n_batches = 0
        self.running_loss = 0.0
        self.local_smashed = None
        self._load_from_cache()
        self.receive_client_model()
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        self.notify("start_fit", round=current_round, client_id=self.index, model=self.model)
        return self.forward_to_cut()

    def forward_to_cut(self):
        for X, y in self.train_set:
            X = X.to(self.device)
            self.optimizer.zero_grad()
            self.local_smashed = self.model(X)
            remote_smashed = self.local_smashed.clone().detach().requires_grad_(True)
            self.send_smashed_data(remote_smashed, y)
            yield remote_smashed, y


    def backward(self):
        grad_cut, server_loss = self.receive_gradients()
        self.local_smashed.backward(grad_cut.to(self.local_smashed.device))
        self._clip_grads(self.model)
        self.optimizer.step()
        self.running_loss += server_loss
        self.n_batches += 1

    def end_epoch(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def end_round(self, current_round) -> None:
        self._last_round = current_round

        self.notify(
            "end_fit",
            round=current_round,
            client_id=self.index,
            model=self.model,
            loss=(self.running_loss / max(1, self.n_batches)),
        )

        self.model.cpu()
        clear_cuda_cache()

        self.send_client_model()
        self._check_persistency()
        self._save_to_cache()
        # return running_loss / max(1, n_batches)

    def evaluate(self, evaluator, test_set):
        return {}

    def finalize(self) -> None:
        return None