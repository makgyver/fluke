from copy import deepcopy

import torch

from fluke.evaluation import Evaluator
from fluke.comm import Message
from fluke.data import FastDataLoader
from fluke.server import Server
from fluke.utils import clear_cuda_cache
from fluke.utils.model import safe_load_state_dict


class ServerSL(Server):

    def __init__(
        self,
        model,                 # modello server-side
        client_model,          # modello client-side  (solo nel caso centralized)
        test_set,
        clients,
        optimizer_cfg,
        loss_fn,
        weighted: bool = False,
        lr: float = 1.0,
        clipping: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            test_set=test_set,
            clients=clients,
            weighted=weighted,
            lr=lr,
            **kwargs,
        )

        self.client_model = deepcopy(client_model) #solo nel caso centralized, nel caso p2p  basterà l'index dell'ultimo client allenato
        self._optimizer_cfg = optimizer_cfg
        self.optimizer = None
        self.scheduler = None

        self.hyper_params.update(
            loss_fn=loss_fn,
            clipping=clipping,
        )

    def send_client_model(self, client_index: int) -> None:
        self.channel.send(
            Message(self.client_model, "client_model", "server", inmemory=True), client_index)

    def receive_client_model(self, client_index: int) -> None: #solo nel caso centralized
        msg = self.channel.receive("server", client_index, msg_type="client_model")
        incoming_model = msg.payload

        if self.client_model is None:
            self.client_model = deepcopy(incoming_model)
        else:
            safe_load_state_dict(self.client_model, incoming_model.state_dict())

    def _clip_grads(self) -> None:
        if self.hyper_params.clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clipping)

    def train_on_smashed_data(
        self,
        smashed: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        smashed = smashed.to(self.device)
        smashed.requires_grad_(True)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        server_output = self.model(smashed)
        loss = self.hyper_params.loss_fn(server_output, y)
        loss.backward()

        grad_cut = smashed.grad.clone().detach().cpu()

        self._clip_grads()
        self.optimizer.step()

        return grad_cut, float(loss.item())

    def end_round(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

        self.model.cpu()
        clear_cuda_cache()

    def aggregate(self, eligible, client_models) -> None: #non serve per SL
        return None

    def evaluate(
            self, evaluator: Evaluator, test_set: FastDataLoader, round: int
    ) -> dict[str, float]:
        # "concateno" le due reti per valutare il modello completo
        if test_set is not None:
            full_model = torch.nn.Sequential(
                self.client_model,
                self.model
            )
            return evaluator.evaluate(round, full_model, test_set, loss_fn=None, device=self.device)
        return {}

    def state_dict(self) -> dict:
        return {
            "server_model": self.model.state_dict() if self.model is not None else None,
            "client_model": self.client_model.state_dict() if self.client_model is not None else None,
        }