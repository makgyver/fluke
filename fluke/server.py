"""
The module ``fluke.server`` provides the base classes for the servers in ``fluke``.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Union

import numpy as np
import torch
from rich.progress import track
from torch import device
from torch.nn import Module

from . import DDict, GlobalSettings, ObserverSubject  # NOQA
from .comm import Channel, Message  # NOQA
from .data import FastDataLoader  # NOQA
from .evaluation import Evaluator  # NOQA
from .utils.model import STATE_DICT_KEYS_TO_IGNORE  # NOQA

if TYPE_CHECKING:
    from .client import Client


__all__ = [
    "Server"
]

torch.serialization.add_safe_globals([set])


class Server(ObserverSubject):
    """Basic Server for Federated Learning.
    This class is the base class for all servers in ``fluke``. It implements the basic
    functionalities of a federated learning server. The default behaviour of this server is based
    on the Federated Averaging algorithm. The server is responsible for coordinating the learning
    process, selecting the clients for each round, sending the global model to the clients, and
    aggregating the models received from the clients at the end of the round. The server also
    evaluates the model server-side (if the test data is provided).

    Attributes:
        hyper_params (DDict):
          The hyper-parameters of the server. The default hyper-parameters are:

          - weighted: A boolean indicating if the clients should be weighted by the number of
            samples when aggregating the models.

          When a new server class inherits from this class, it must add all its hyper-parameters
          to this dictionary.
        device (torch.device): The device where the server runs.
        model (torch.nn.Module): The federated model to be trained.
        clients (Iterable[Client]): The clients that will participate in the federated learning
          process.
        rounds (int): The number of rounds that have been executed.
        test_set (FastDataLoader): The test data to evaluate the model. If None, the model
          will not be evaluated server-side.
        evaluator (Evaluator): The evaluator to compute the evaluation metrics.

    Args:
        model (torch.nn.Module): The federated model to be trained.
        test_set (FastDataLoader): The test data to evaluate the model.
        clients (Iterable[Client]): The clients that will participate in the federated learning
          process.
        evaluator (Evaluator): The evaluator to compute the evaluation metrics.
        eval_every (int): The number of rounds between evaluations. Defaults to 1.
        weighted (bool): A boolean indicating if the clients should be weighted by the
          number of samples when aggregating the models. Defaults to False.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False):
        super().__init__()
        self.hyper_params = DDict(
            weighted=weighted
        )
        self.device: device = GlobalSettings().get_device()
        self.model: Module = model
        self.clients: Iterable[Client] = clients
        self._channel: Channel = Channel()
        self.n_clients: int = len(clients)
        self.rounds: int = 0
        self.test_set: FastDataLoader = test_set
        self._participants: set[int] = set()

        for client in self.clients:
            client.set_server(self)

    @property
    def channel(self) -> Channel:
        """The channel to communicate with the clients.

        Important:
            Always use this channel to exchange data/information with the clients.
            The server should directly call the clients' methods only to trigger specific actions.

        Returns:
            Channel: The channel to communicate with the clients.
        """
        return self._channel

    @property
    def has_test(self) -> bool:
        """Return whether the server can evaluate the model.

        Returns:
            bool: True if the server can evaluate the model, False otherwise.
        """
        return self.test_set is not None

    @property
    def has_model(self) -> bool:
        """Return whether the server owns a global model.

        Returns:
            bool: True if the server owns a global model, False otherwise.
        """
        return self.model is not None

    def broadcast_model(self, eligible: Iterable[Client]) -> None:
        """Broadcast the global model to the clients.

        Args:
            eligible (Iterable[Client]): The clients that will receive the global model.
        """
        self.channel.broadcast(Message(self.model, "model", self), eligible)

    def fit(self,
            n_rounds: int = 10,
            eligible_perc: float = 0.1,
            finalize: bool = True,
            **kwargs: dict[str, Any]) -> None:
        """Run the federated learning algorithm.
        The default behaviour of this method is to run the Federated Averaging algorithm. The server
        selects a percentage of the clients to participate in each round, sends the global model to
        the clients, which compute the local updates and send them back to the server. The server
        aggregates the models of the clients and repeats the process for a number of rounds.
        During the process, the server evaluates the global model and the local model every
        ``eval_every`` rounds.

        Args:
            n_rounds (int, optional): The number of rounds to run. Defaults to 10.
            eligible_perc (float, optional): The percentage of clients that will be selected for
              each round. Defaults to 0.1.
            finalize (bool, optional): If True, the server will finalize the federated learning
                process. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        with GlobalSettings().get_live_renderer():
            progress_fl = GlobalSettings().get_progress_bar("FL")
            progress_client = GlobalSettings().get_progress_bar("clients")
            client_x_round = int(self.n_clients * eligible_perc)
            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            task_local = progress_client.add_task("[green]Local Training", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            for round in range(self.rounds, total_rounds):
                self._notify_start_round(round + 1, self.model)
                eligible = self.get_eligible_clients(eligible_perc)
                self._notify_selected_clients(round + 1, eligible)
                self.broadcast_model(eligible)

                for c, client in enumerate(eligible):
                    client.local_update(round + 1)
                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)

                self.aggregate(eligible)
                self._compute_evaluation(round, eligible)
                self._notify_end_round(round + 1)
                self.rounds += 1
            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

        if finalize:
            self.finalize()

    def _compute_evaluation(self, round: int, eligible: Iterable[Client]) -> None:
        evaluator = GlobalSettings().get_evaluator()

        if GlobalSettings().get_eval_cfg().locals:
            client_evals = {client.index: client.evaluate(evaluator, self.test_set)
                            for client in eligible}
            self._notify_evaluation(round + 1, "locals", client_evals)

        if GlobalSettings().get_eval_cfg().server:
            evals = self.evaluate(evaluator, self.test_set)
            self._notify_evaluation(round + 1, "global", evals)

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        """Evaluate the global federated model on the ``test_set``.
        If the test set is not set, the method returns an empty dictionary.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
                the results.
        """
        if test_set is not None:
            return evaluator.evaluate(self.rounds + 1, self.model, test_set)
        return {}

    def finalize(self) -> None:
        """Finalize the federated learning process.
        The finalize method is called at the end of the federated learning process. The client-side
        evaluation is only done if the client has participated in at least one round.
        """
        client_to_eval = [client for client in self.clients if client.index in self._participants]
        self.broadcast_model(client_to_eval)
        for client in track(client_to_eval, "Finalizing federation...", transient=True):
            client.finalize()
        # self._compute_evaluation(self.rounds, client_to_eval)
        self._notify_finalize()

    def get_eligible_clients(self, eligible_perc: float) -> Iterable[Client]:
        """Get the clients that will participate in the current round.

        Args:
            eligible_perc (float): The percentage of clients that will be selected.

        Returns:
            Iterable[Client]: The clients that will participate in the current round.
        """
        if eligible_perc == 1:
            if not self._participants:
                self._participants = set(range(self.n_clients))
            return self.clients
        n = int(self.n_clients * eligible_perc)
        selected = np.random.choice(self.clients, n, replace=False)
        self._participants.update([c.index for c in selected])
        return selected

    def get_client_models(self, eligible: Iterable[Client], state_dict: bool = True) -> list[Any]:
        """Retrieve the models of the clients.
        This method assumes that the clients have already sent their models to the server.

        Args:
            eligible (Iterable[Client]): The clients that will participate in the aggregation.
            state_dict (bool, optional): If True, the method returns the state_dict of the models.
              Otherwise, it returns the models. Defaults to True.

        Returns:
            list[torch.nn.Module]: The models of the clients.
        """
        client_models = [self.channel.receive(self, client, "model").payload
                         for client in eligible]
        if state_dict:
            return [m.state_dict() for m in client_models]
        return client_models

    def _get_client_weights(self, eligible: Iterable[Client]):
        """Get the weights of the clients for the aggregation.
        The weights are calculated based on the number of samples of each client.
        If the hyperparameter ``weighted`` is True, the clients are weighted by their number of
        samples. Otherwise, all clients have the same weight.

        Caution:
            The computation of the weights do not adhere to the "best-practices" of ``fluke``
            because the server should not have direct access to the number of samples of the
            clients. Thus, the computation of the weights should be done communicating with the
            clients through the channel, but for simplicity, we are not following this practice
            here. However, the communication overhead is negligible and does not affect the logged
            performance.

        Args:
            eligible (Iterable[Client]): The clients that will participate in the aggregation.

        Returns:
            list[float]: The weights of the clients.
        """
        if "weighted" in self.hyper_params.keys() and self.hyper_params.weighted:
            num_ex = [client.n_examples for client in eligible]
            tot_ex = sum(num_ex)
            return [num_ex[i] / tot_ex for i in range(len(eligible))]
        else:
            return [1. / len(eligible)] * len(eligible)

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        r"""Aggregate the models of the clients.
        The aggregation is done by averaging the models of the clients. If the hyperparameter
        ``weighted`` is True, the clients are weighted by their number of samples.
        The method directly updates the model of the server. Formally, let :math:`\theta` be the
        model of the server, :math:`\theta_i` the model of client :math:`i`, and :math:`w_i` the
        weight of client :math:`i` such that :math:`\sum_{i=1}^{N} w_i = 1`. The aggregation is
        done as follows [FedAVG]_:

        .. math::
            \\theta = \\sum_{i=1}^{N} w_i \\theta_i

        Args:
            eligible (Iterable[Client]): The clients that will participate in the aggregation.

        References:
            .. [FedAVG] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
               "Communication-Efficient Learning of Deep Networks from Decentralized Data".
               In AISTATS (2017).
        """
        avg_model_sd = OrderedDict()
        clients_sd = self.get_client_models(eligible)
        weights = self._get_client_weights(eligible)
        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue
            for i, client_sd in enumerate(clients_sd):
                if key not in avg_model_sd:
                    avg_model_sd[key] = weights[i] * client_sd[key]
                else:
                    avg_model_sd[key] = avg_model_sd[key] + weights[i] * client_sd[key]

        self.model.load_state_dict(avg_model_sd)

    def _notify_start_round(self, round: int, global_model: Any) -> None:
        """Notify the observers that a new round has started.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
        """
        for observer in self._observers:
            observer.start_round(round, global_model)

    def _notify_evaluation(self,
                           round: int,
                           type: str,
                           evals: Union[dict[str, float], dict[int, dict[str, float]]]) -> None:
        """Notify the observers that a round has ended.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
            data (FastDataLoader): The test data.
            client_evals (Iterable[Any]): The evaluation metrics of the clients.
        """
        for observer in self._observers:
            observer.server_evaluation(round, type, evals)

    def _notify_end_round(self, round: int) -> None:
        """Notify the observers that a round has ended.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
            data (FastDataLoader): The test data.
            client_evals (Iterable[Any]): The evaluation metrics of the clients.
        """
        for observer in self._observers:
            observer.end_round(round)

    def _notify_selected_clients(self, round: int, clients: Iterable[Any]) -> None:
        """Notify the observers that the clients have been selected for the current round.

        Args:
            round (int): The round number.
            clients (Iterable[Any]): The clients selected for the current round.
        """
        for observer in self._observers:
            observer.selected_clients(round, clients)

    def _notify_finalize(self) -> None:
        """Notify the observers that the federated learning process has ended.

        Args:
            evals (dict[str, float]): The evaluation metrics of the global model.
            client_evals (Iterable[Any]): The evaluation metrics of the clients.
        """
        for observer in self._observers:
            observer.finished(self.rounds + 1)

    def __str__(self) -> str:
        hpstr = ", ".join([f"{h}={str(v)}" for h, v in self.hyper_params.items()])
        return f"{self.__class__.__name__}({hpstr})"

    def __repr__(self) -> str:
        return str(self)

    def state_dict(self) -> dict:
        """Return the server's state as a dictionary.

        Returns:
            dict: The server's state.
        """
        return {
            "model": self.model.state_dict(),
            "rounds": self.rounds,
            "participants": self._participants
        }

    def save(self, path: str) -> None:
        """Save the server's state to file.

        Args:
            path (str): The path to save the server.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the server's state from file.

        Args:
            path (str): The path to load the server's state.
        """
        state = torch.load(path, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.rounds = state["rounds"]
        self._participants = set(state["participants"])
