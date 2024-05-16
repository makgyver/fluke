"""
The module ``fluke.server`` provides the base classes for the servers in ``fluke``.
"""
from __future__ import annotations
from rich.progress import track
import numpy as np
from typing import Any, Sequence
from collections import OrderedDict
import torch
from torch import device
from torch.nn import Module

from .evaluation import ClassificationEval  # NOQA
from .comm import Channel, Message  # NOQA
from .data import FastDataLoader  # NOQA
from .utils.model import STATE_DICT_KEYS_TO_IGNORE  # NOQA
from . import GlobalSettings, ObserverSubject, DDict  # NOQA

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .client import Client


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
        clients (Sequence[Client]): The clients that will participate in the federated learning
          process.
        channel (Channel): The channel to communicate with the clients.
        rounds (int): The number of rounds that have been executed.
        test_data (FastDataLoader): The test data to evaluate the model. If None, the model
          will not be evaluated server-side.

    Args:
        model (torch.nn.Module): The federated model to be trained.
        test_data (FastDataLoader): The test data to evaluate the model.
        clients (Sequence[Client]): The clients that will participate in the federated learning
          process.
        eval_every (int): The number of rounds between evaluations. Defaults to 1.
        weighted (bool): A boolean indicating if the clients should be weighted by the
          number of samples when aggregating the models. Defaults to False.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 test_data: FastDataLoader,
                 clients: Sequence[Client],
                 eval_every: int = 1,
                 weighted: bool = False):
        super().__init__()
        self.hyper_params = DDict(
            weighted=weighted
        )
        self.device: device = GlobalSettings().get_device()
        self.model: Module = model
        self.clients: Sequence[Client] = clients
        self._channel: Channel = Channel()
        self.n_clients: int = len(clients)
        self.rounds: int = 0
        self.test_data: FastDataLoader = test_data
        self._eval_every: int = eval_every
        self._participants: set[int] = set()

        for client in self.clients:
            client.set_server(self)

    @property
    def channel(self) -> Channel:
        """The channel to communicate with the clients.

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
        return self.test_data is not None

    @property
    def has_model(self) -> bool:
        """Return whether the server owns a global model.

        Returns:
            bool: True if the server owns a global model, False otherwise.
        """
        return self.model is not None

    def broadcast_model(self, eligible: Sequence[Client]) -> None:
        """Broadcast the global model to the clients.

        Args:
            eligible (Sequence[Client]): The clients that will receive the global model.
        """
        self._channel.broadcast(Message(self.model, "model", self), eligible)

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1) -> None:
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
                    client.fit()
                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)
                self.aggregate(eligible)

                client_evals, evals = [], {}
                if (round + 1) % self._eval_every == 0:
                    for client in eligible:
                        client_eval = client.evaluate()
                        if client_eval:
                            client_evals.append(client_eval)
                    evals = self.evaluate()

                self._notify_end_round(round + 1, evals, client_evals)
                self.rounds += 1
            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

        self.finalize()

    def evaluate(self) -> dict[str, float]:
        """Evaluate the global federated model on the ``test_set``.
        If the test set is not set, the method returns an empty dictionary.

        Warning:
            To date, only classification tasks are supported.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
                the results.
        """
        if self.test_data is not None:
            return ClassificationEval(self.clients[0].hyper_params.loss_fn,
                                      #   self.model.output_size,
                                      self.test_data.num_labels,
                                      device=self.device).evaluate(self.model,
                                                                   self.test_data)
        return {}

    def finalize(self) -> None:
        """Finalize the federated learning process.
        The finalize method is called at the end of the federated learning process. The client-side
        evaluation is only done if the client has participated in at least one round.
        """
        client_evals = []
        client_to_eval = [client for client in self.clients if client.index in self._participants]
        self.broadcast_model(client_to_eval)
        for client in track(client_to_eval, "Finalizing federation..."):
            client.finalize()
            client_eval = client.evaluate()
            if client_eval:
                client_evals.append(client_eval)
        self._notify_finalize(client_evals)

    def get_eligible_clients(self, eligible_perc: float) -> Sequence[Client]:
        """Get the clients that will participate in the current round.

        Args:
            eligible_perc (float): The percentage of clients that will be selected.

        Returns:
            Sequence[Client]: The clients that will participate in the current round.
        """
        if eligible_perc == 1:
            if not self._participants:
                self._participants = set(range(self.n_clients))
            return self.clients
        n = int(self.n_clients * eligible_perc)
        selected = np.random.choice(self.clients, n)
        self._participants.update([c.index for c in selected])
        return selected

    def get_client_models(self, eligible: Sequence[Client], state_dict: bool = True) -> list[Any]:
        """Retrieve the models of the clients.
        This method assumes that the clients have already sent their models to the server.

        Args:
            eligible (Sequence[Client]): The clients that will participate in the aggregation.
            state_dict (bool, optional): If True, the method returns the state_dict of the models.
              Otherwise, it returns the models. Defaults to True.

        Returns:
            list[torch.nn.Module]: The models of the clients.
        """
        client_models = [self._channel.receive(self, client, "model").payload
                         for client in eligible]
        if state_dict:
            return [m.state_dict() for m in client_models]
        return client_models

    def _get_client_weights(self, eligible: Sequence[Client]):
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
            eligible (Sequence[Client]): The clients that will participate in the aggregation.

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
    def aggregate(self, eligible: Sequence[Client]) -> None:
        """Aggregate the models of the clients.
        The aggregation is done by averaging the models of the clients. If the hyperparameter
        ``weighted`` is True, the clients are weighted by their number of samples.
        The method directly updates the model of the server.

        Args:
            eligible (Sequence[Client]): The clients that will participate in the aggregation.
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
                    avg_model_sd[key] += weights[i] * client_sd[key]
        self.model.load_state_dict(avg_model_sd)

    def _notify_start_round(self, round: int, global_model: Any) -> None:
        """Notify the observers that a new round has started.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
        """
        for observer in self._observers:
            observer.start_round(round, global_model)

    def _notify_end_round(self,
                          round: int,
                          evals: dict[str, float],
                          client_evals: Sequence[Any]) -> None:
        """Notify the observers that a round has ended.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
            data (FastDataLoader): The test data.
            client_evals (Sequence[Any]): The evaluation metrics of the clients.
        """
        for observer in self._observers:
            observer.end_round(round, evals, client_evals)

    def _notify_selected_clients(self, round: int, clients: Sequence[Any]) -> None:
        """Notify the observers that the clients have been selected for the current round.

        Args:
            round (int): The round number.
            clients (Sequence[Any]): The clients selected for the current round.
        """
        for observer in self._observers:
            observer.selected_clients(round, clients)

    def _notify_error(self, error: str) -> None:
        """Notify the observers that an error has occurred.

        Args:
            error (str): The error message.
        """
        for observer in self._observers:
            observer.error(error)

    def _notify_finalize(self, client_evals: Sequence[Any]) -> None:
        """Notify the observers that the federated learning process has ended.

        Args:
            client_evals (Sequence[Any]): The evaluation metrics of the clients.
        """
        for observer in self._observers:
            observer.finished(client_evals)

    def __str__(self) -> str:
        hpstr = ",".join([f"{h}={str(v)}" for h, v in self.hyper_params.items()])
        return f"{self.__class__.__name__}({hpstr})"

    def __repr__(self) -> str:
        return self.__str__()
