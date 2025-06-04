import sys
from typing import Any

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from ..client import Client  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..utils.model import ModOpt  # NOQA
from ..utils.model import optimizer_to, safe_load_state_dict  # NOQA
from .utils import ModelBuilder  # NOQA

__all__ = ["ParallelClient"]


class ParallelClient(Client):
    """A client that can be used in a parallel setting."""

    def __init__(self, builder: ModelBuilder, *args: Any, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._builder = builder

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the client as a dictionary. This method is used to serialize the client
        when using multiprocessing.

        Returns:
            dict: The state of the client.
        """
        state = self.state_dict()
        state["optimizer_cfg"] = self._optimizer_cfg
        state["device"] = str(self.device)
        state["train_set"] = self.train_set
        state["test_set"] = self.test_set
        state["hyper_params"] = self.hyper_params
        state["builder"] = self._builder
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the client from a dictionary. This method is used to deserialize the
        client when using multiprocessing.

        Args:
            state (dict): The state of the client.
        """
        self._modopt = ModOpt()
        self._builder = state["builder"]
        if self.model is None:
            self.model = self._builder.build()
        self._optimizer_cfg = state["optimizer_cfg"]
        if state["modopt"]["model"] is not None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)
            self._modopt.load_state_dict(state["modopt"])

        self._index = state["index"]
        self._last_round = state["last_round"]
        self.device = torch.device(state["device"])
        self.train_set = state["train_set"]
        self.test_set = state["test_set"]
        self.hyper_params = state["hyper_params"]

    def local_update(
        self,
        current_round: int,
        current_model_sd: dict,
        device: str,
        prefit: bool,
        postfit: bool,
        evaluator: Evaluator,
    ) -> tuple[Module, float, int]:
        self.device = device

        self._last_round = current_round
        safe_load_state_dict(self.model, current_model_sd)

        eval_results = {}
        if evaluator is not None and prefit:
            metrics = self.evaluate(evaluator, self.test_set)
            if metrics:
                eval_results["pre-fit"] = metrics

        loss = self.fit()
        optimizer_to(self.optimizer, "cpu")

        if evaluator is not None and postfit:
            metrics = self.evaluate(evaluator, self.test_set)
            if metrics:
                eval_results["post-fit"] = metrics

        return self._modopt, loss, self.index, eval_results

    def send_model(self):
        raise NotImplementedError(
            "ParallelClient does not support sending model to server using the channel."
        )

    def receive_model(self, model: Module) -> None:
        raise NotImplementedError(
            "ParallelClient does not support receiving model from server using the channel."
        )
