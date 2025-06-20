from random import choice
from typing import Generator, Literal

import numpy as np
from torch.nn import Module

from ... import FlukeENV  # NOQA
from ...client import Client  # NOQA
from ...comm import TimedMessage  # NOQA
from ...config import OptimizerConfigurator  # NOQA
from ...data import DataLoader, FastDataLoader  # NOQA
from ...utils.model import safe_load_state_dict, aggregate_models  # NOQA

__all__ = ["AbstractDFLClient", "GossipClient"]


class AbstractDFLClient(Client):
    """Abstract client for decentralized federated learning (DFL).

    Args:
        index (int): The index of the client.
        model (Module): The model to be trained.
        neighbours (list[int]): The indices of the neighbouring clients.
        train_set (FastDataLoader | DataLoader): The training dataset.
        test_set (FastDataLoader | DataLoader): The testing dataset.
        optimizer_cfg (OptimizerConfigurator): The optimizer configuration.
        loss_fn (Module): The loss function.
        local_epochs (int): Number of local training epochs. Defaults to 3.
        fine_tuning_epochs (int): Number of fine-tuning epochs. Defaults to 0.
        clipping (float): Gradient clipping value. Defaults to 0 (no clipping).
        persistency (bool): Whether to persist the model across rounds. Defaults to True.
        activation_rate (float): Probability of the client being active in each round.
            Defaults to 1 (always active).
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        index: int,
        model: Module,
        neighbours: list[int],
        train_set: FastDataLoader | DataLoader,
        test_set: FastDataLoader | DataLoader,
        optimizer_cfg: OptimizerConfigurator,
        loss_fn: Module,
        local_epochs: int = 3,
        fine_tuning_epochs: int = 0,
        clipping: float = 0,
        persistency: bool = True,
        activation_rate: float = 1,
        **kwargs,
    ):
        super().__init__(
            index=index,
            train_set=train_set,
            test_set=test_set,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss_fn,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping,
            persistency=persistency,
            **kwargs,
        )

        self.hyper_params.update(activation_rate=activation_rate)
        self.model = model
        self.neighbours: list[int] = neighbours
        self._num_updates: int = 0
        self._active_history: dict[int, bool] = {}

    def is_active(self, iter: int) -> bool:
        """Check if the client is active in the current iteration.

        Args:
            iter (int): The current iteration number.

        Returns:
            bool: True if the client is active, False otherwise.
        """
        if iter not in self._active_history:
            self._active_history[iter] = np.random.rand() < self.hyper_params.activation_rate
        return self._active_history[iter]

    def send_model(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def receive_model(self) -> Generator:
        raise NotImplementedError()

    def local_update(self, round: int) -> None:
        if self._num_updates == 0 or self.channel.has_messages(self.index, "model"):
            super().local_update(round)
            self._num_updates += 1
        elif self._num_updates > 0:
            self.send_model()

    def finalize(self) -> None:
        self._load_from_cache()
        evaluator = FlukeENV().get_evaluator()

        if FlukeENV().get_eval_cfg().pre_fit:
            metrics = self.evaluate(evaluator, self.test_set)
            if metrics:
                self.notify(
                    "client_evaluation",
                    round=-1,
                    client_id=self.index,
                    phase="pre-fit",
                    evals=metrics,
                )

        if FlukeENV().get_eval_cfg().post_fit:
            self.fit()
            metrics = self.evaluate(evaluator, self.test_set)
            if metrics:
                self.notify(
                    "client_evaluation",
                    round=-1,
                    client_id=self.index,
                    phase="post-fit",
                    evals=metrics,
                )

        self._save_to_cache()


class GossipClient(AbstractDFLClient):
    """A client for decentralized federated learning using gossip protocol.

    In the gossip protocol, clients send their model to a randomly chosen neighbour. Upon
    receiving models from neighbours, the client applies a specified policy to update its model.
    Possible policies include:
    - "random": Selects a random model from the received messages.
    - "aggregate": Aggregates all received models using the average.
    - "last": Uses the last received model based on the timestamp.
    - "best": Selects the model with the highest accuracy on the local test set.

    In case of ties, the last model processed in the order of receipt is chosen.

    Args:
        *args: Positional arguments passed to the parent class.
        policy (str): The policy to apply when receiving models from neighbours. Must be one of
            "random", "aggregate", "last", or "best". Defaults to "random".
        **kwargs: Keyword arguments passed to the parent class.

    Raises:
        AssertionError: If the provided policy is not one of the allowed values.
    """

    def __init__(
        self, *args, policy: str = Literal["random", "aggregate", "last", "best"], **kwargs
    ):
        assert policy in ["random", "aggregate", "last", "best"], f"Invalid policy {policy}."
        super().__init__(*args, **kwargs)
        self.hyper_params.update(policy=policy)

    def send_model(self) -> None:
        recipient = choice(self.neighbours)
        self.channel.send(
            TimedMessage(
                self.model, "model", self.index, inmemory=True, timestamp=self._num_updates + 1
            ),
            recipient,
        )

    def _apply_policy(self, messages: list[TimedMessage]) -> tuple[Module, int]:

        if self.hyper_params.policy == "random":
            msg = choice(messages)
            return msg.payload, msg.timestamp
        elif self.hyper_params.policy == "aggregate":
            return (
                aggregate_models(
                    self.model,
                    (msg.payload for msg in messages),
                    np.ones(len(messages)) / len(messages),
                    eta=1,
                    inplace=False,
                ),
                max(msg.timestamp for msg in messages),
            )
        elif self.hyper_params.policy == "last":
            last = 0
            last_msg = None
            for msg in messages:
                if msg.timestamp >= last:
                    last = msg.timestamp
                    last_msg = msg
            return last_msg.payload, last
        elif self.hyper_params.policy == "best":
            best_acc = -1
            best_msg = None
            for msg in messages:
                acc = self.evaluate(msg.payload, self.test_set)["accuracy"]
                if acc >= best_acc:
                    best_acc = acc
                    best_msg = msg

                del msg.payload  # Free memory
            return best_msg.payload, best_msg.timestamp

    def receive_model(self) -> None:
        if self._num_updates == 0:
            return
        messages = self.channel.receive_all(self.index, msg_type="model")
        selected_model, updates = self._apply_policy(messages)

        self._num_updates = updates
        safe_load_state_dict(self.model, selected_model.state_dict())
