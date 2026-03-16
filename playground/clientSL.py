import torch

from fluke.client import Client
from fluke.comm import Message
from fluke.utils import clear_cuda_cache
from fluke.utils.model import safe_load_state_dict


class ClientSL(Client):
    def __init__(
        self,
        index,
        train_set,
        test_set,
        optimizer_cfg,
        loss_fn,
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

    def receive_client_model(self) -> None:
        msg = self.channel.receive(self.index, "server", msg_type="client_model")

        if self.model is None:
            self.model = msg.payload
        else:
            safe_load_state_dict(self.model, msg.payload.state_dict())

    def send_client_model(self) -> None: #centralized
        self.channel.send(Message(self.model, "client_model", self.index, inmemory=True),"server")

    def forward_to_cut(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_smashed = self.model(X)
        remote_smashed = local_smashed.clone().detach().requires_grad_(True)
        return local_smashed, remote_smashed

    def local_update(self, current_round: int, server=None) -> float:
        if server is None:
            raise ValueError("ClientSL.local_update richiede ServerSL.") #per ora

        self._load_from_cache()
        self.receive_client_model()

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        self.notify("start_fit", round=current_round, client_id=self.index, model=self.model)

        running_loss = 0.0
        n_batches = 0

        for _ in range(self.hyper_params.local_epochs):
            for X, y in self.train_set:
                X = X.to(self.device)
                y = y.to(server.device)

                # client-side forward
                self.optimizer.zero_grad()
                local_smashed, remote_smashed = self.forward_to_cut(X)

                #chiamata diretta al server
                #forward, backward e update del modello server-side
                grad_cut, server_loss = server.train_on_smashed_data(remote_smashed, y)

                # backward e update del modello client-side
                local_smashed.backward(grad_cut.to(local_smashed.device))
                self._clip_grads(self.model)
                self.optimizer.step()

                running_loss += server_loss
                n_batches += 1

            if self.scheduler is not None:
                self.scheduler.step()

            server.end_round() #anche questa è una chiamata diretta

        self._last_round = current_round

        self.notify(
            "end_fit",
            round=current_round,
            client_id=self.index,
            model=self.model,
            loss=(running_loss / max(1, n_batches)),
        )

        self.model.cpu()
        clear_cuda_cache()

        self.send_client_model()
        self._check_persistency()
        self._save_to_cache()

        return running_loss / max(1, n_batches)

    def evaluate(self, evaluator, test_set):
        return {}

    def finalize(self) -> None:
        return None