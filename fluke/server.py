"""
The module :mod:`fluke.server` provides the base classes for the servers in :mod:`fluke`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator, Iterable, Union

import numpy as np
import torch
from rich.progress import track

from . import DDict, FlukeENV, ObserverSubject  # NOQA
from .comm import Channel, Message  # NOQA
from .data import FastDataLoader  # NOQA
from .evaluation import Evaluator  # NOQA
from .utils.model import aggregate_models  # NOQA

if TYPE_CHECKING:
    from .client import Client


__all__ = [
    "Server",
    "EarlyStopping"
]

torch.serialization.add_safe_globals([set])


class EarlyStopping(Exception):
    """Exception raised when the fedearted training process is stopped early.

    This exception is used to signal that the training process should be stopped early.
    It is used to stop the :meth:`Server.fit` method.
    """

    def __init__(self, round: int) -> None:
        msg = f"Early stopping at round {round}"
        super().__init__(msg)
        self.message = f"Early stopping at round {round}"
        self.round = round

    def __str__(self) -> str:
        return self.message


class Server(ObserverSubject):
    """Basic Server for Federated Learning.

    This class is the base class for all servers in : mod: `fluke`. It implements the basic
    functionalities of a federated learning server. The default behaviour of this server is based
    on the Federated Averaging algorithm. The server is responsible for coordinating the learning
    process, selecting the clients for each round, sending the global model to the clients, and
    aggregating the models received from the clients at the end of the round. The server also
    evaluates the model server-side(if the test data is provided).

    Attributes:
        hyper_params(DDict):
          The hyper-parameters of the server. The default hyper-parameters are:

          - weighted: A boolean indicating if the clients should be weighted by the number of
            samples when aggregating the models.

          When a new server class inherits from this class, it must add all its hyper-parameters
          to this dictionary.
        device(torch.device): The device where the server runs.
        model(torch.nn.Module): The federated model to be trained.
        clients(Iterable[Client]): The clients that will participate in the federated learning
          process.
        n_clients(int): The number of clients that will participate in the federated learning
        rounds(int): The number of rounds that have been executed.
        test_set(FastDataLoader): The test data to evaluate the model. If None, the model
          will not be evaluated server-side.

    Args:
        model(torch.nn.Module): The federated model to be trained.
        test_set(FastDataLoader): The test data to evaluate the model.
        clients(Iterable[Client]): The clients that will participate in the federated learning
          process.
        weighted(bool): A boolean indicating if the clients should be weighted by the
          number of samples when aggregating the models. Defaults to False.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 test_set: FastDataLoader,
                 clients: Iterable[Client],
                 weighted: bool = False,
                 lr: float = 1.0,
                 **kwargs: dict[str, Any]):
        super().__init__()
        self.hyper_params = DDict(
            weighted=weighted,
            lr=lr
        )
        self.device: torch.device = FlukeENV().get_device()
        self.model: torch.nn.Module = model
        self.clients: Iterable[Client] = clients
        self.n_clients: int = len(clients)
        self.rounds: int = 0
        self.test_set: FastDataLoader = test_set

        self._channel: Channel = Channel()
        self._participants: set[int] = set()

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
        self.channel.broadcast(Message(self.model, "model", self, inmemory=None), eligible)

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

        Args:
            n_rounds (int, optional): The number of rounds to run. Defaults to 10.
            eligible_perc (float, optional): The percentage of clients that will be selected for
              each round. Defaults to 0.1.
            finalize (bool, optional): If True, the server will finalize the federated learning
                process. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        with FlukeENV().get_live_renderer():
            progress_fl = FlukeENV().get_progress_bar("FL")
            progress_client = FlukeENV().get_progress_bar("clients")
            client_x_round = int(self.n_clients * eligible_perc)
            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            task_local = progress_client.add_task("[green]Local Training", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            for round in range(self.rounds, total_rounds):
                try:
                    self._notify_start_round(round + 1, self.model)
                    eligible = self.get_eligible_clients(eligible_perc)
                    self._notify_selected_clients(round + 1, eligible)
                    self.broadcast_model(eligible)

                    for c, client in enumerate(eligible):
                        client.local_update(round + 1)
                        self._participants.update([client.index])
                        progress_client.update(task_id=task_local, completed=c+1)
                        progress_fl.update(task_id=task_rounds, advance=1)

                    client_models = self.receive_client_models(eligible, state_dict=False)
                    self.aggregate(eligible, client_models)
                    self._compute_evaluation(round, eligible)
                    self._notify_end_round(round + 1)
                    self.rounds += 1

                except KeyboardInterrupt:
                    self._notify_interrupt()
                    break

                except EarlyStopping:
                    self._notify_early_stop()
                    break

            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

        if finalize:
            self.finalize()

    def _compute_evaluation(self, round: int, eligible: Iterable[Client]) -> None:
        evaluator = FlukeENV().get_evaluator()

        if FlukeENV().get_eval_cfg().locals:
            client_evals = {client.index: client.evaluate(evaluator, self.test_set)
                            for client in eligible}
            self._notify_evaluation(round + 1, "locals", client_evals)

        if FlukeENV().get_eval_cfg().server:
            evals = self.evaluate(evaluator, self.test_set)
            self._notify_evaluation(round + 1, "global", evals)

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        """Evaluate the global federated model on the :attr:`test_set`.
        If the test set is not set, the method returns an empty dictionary.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
                the results.
        """
        if test_set is not None:
            return evaluator.evaluate(self.rounds + 1, self.model, test_set, device=self.device)
        return {}

    def finalize(self) -> None:
        """Finalize the federated learning process.
        The finalize method is called at the end of the federated learning process. The client-side
        evaluation is only done if the client has participated in at least one round.
        """
        if self.rounds == 0:
            return
        client_to_eval = [client for client in self.clients if client.index in self._participants]
        self.broadcast_model(client_to_eval)
        for client in track(client_to_eval, "Finalizing federation...", transient=True):
            client.finalize()
        # self._compute_evaluation(self.rounds, client_to_eval)
        self._notify_finalize()

    def get_eligible_clients(self, eligible_perc: float) -> Iterable[Client]:
        """Get the clients that will participate in the current round.
        Clients are selected randomly based on the percentage of eligible clients.

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
        return selected

    def receive_client_models(self,
                              eligible: Iterable[Client],
                              state_dict: bool = True) -> Generator[torch.nn.Module, None, None]:
        """Retrieve the models of the clients.
        This method assumes that the clients have already sent their models to the server.
        The models are received through the channel in the same order as the clients in
        ``eligible``.

        Caution:
            The method returns a generator of the models of the clients to avoid to clutter the
            memory with all the models. This means that this method is expected to be called only
            once per round. If the models are needed multiple times, the generator should be
            converted to a list, tuple, or any other iterable.

        Args:
            eligible (Iterable[Client]): The clients that will participate in the aggregation.
            state_dict (bool, optional): If True, the method returns the state_dict of the models.
              Otherwise, it returns the models. Defaults to True.

        Returns:
            Generator[torch.nn.Module]: The models of the clients.
        """
        for client in eligible:
            client_model = self.channel.receive(self, client, "model").payload
            if state_dict:
                client_model = client_model.state_dict()
            yield client_model

    def _get_client_weights(self, eligible: Iterable[Client]) -> list[float]:
        """Get the weights of the clients for the aggregation.
        The weights are calculated based on the number of samples of each client.
        If the hyperparameter ``weighted`` is True, the clients are weighted by their number of
        samples. Otherwise, all clients have the same weight.

        Caution:
            The computation of the weights do not adhere to the "best-practices" of :mod:`fluke`
            because the server should not have direct access to the number of samples of the
            clients. Thus, the computation of the weights should be done communicating with the
            clients through the channel but, for simplicity, we are not following this practice
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
    def aggregate(self,
                  eligible: Iterable[Client],
                  client_models: Iterable[torch.nn.Module]) -> None:
        r"""Aggregate the models of the clients.
        The aggregation is done by averaging the models of the clients. If the hyperparameter
        ``weighted`` is ``True``, the clients are weighted by their number of samples.
        The method directly updates the model of the server. Formally, let :math:`\theta` be the
        model of the server, :math:`\theta_i` the model of client :math:`i`, and :math:`w_i` the
        weight of client :math:`i` such that :math:`\sum_{i=1}^{N} w_i = 1`. The aggregation is
        done as follows [FedAVG]_:

        .. math::
            \theta = \sum_{i=1}^{N} w_i \theta_i

        Note:
            In case of networks with batch normalization layers, the running statistics of the
            batch normalization layers are also aggregated. For all statistics but
            ``num_batches_tracked`` are aggregated the mean is computed, while for the
            ``num_batches_tracked`` parameter, the maximum between the clients' values is taken.

        See also:
            :func:`fluke.utils.model.aggregate_models`

        Args:
            eligible (Iterable[Client]): The clients that will participate in the aggregation.
            client_models (Iterable[torch.nn.Module]): The models of the clients.

        References:
            .. [FedAVG] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
               "Communication-Efficient Learning of Deep Networks from Decentralized Data".
               In AISTATS (2017).
        """
        # agg_model_sd = OrderedDict()
        weights = self._get_client_weights(eligible)
        aggregate_models(self.model, client_models, weights, self.hyper_params.lr, inplace=True)

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

    def _notify_interrupt(self) -> None:
        """Notify the observers that the federated learning process has been interrupted by
        the user.
        """
        for observer in self._observers:
            observer.interrupted()

    def _notify_early_stop(self) -> None:
        """Notify the observers that the federated learning process has been stopped early.
        """
        for observer in self._observers:
            observer.early_stop(self.rounds + 1)

    def _notify_track_item(self, item: str, value: Any) -> None:
        """Notify the observers that an item has been tracked.

        Args:
            round (int): The round number.
            item (str): The item to track.
            value (Any): The value of the item.
        """
        for observer in self._observers:
            observer.track_item(self.rounds + 1, item, value)

    def __str__(self, indent: int = 0) -> str:
        clsname = self.__class__.__name__
        indentstr = " " * (indent + len(clsname) + 1)
        hpstr = f",\n{indentstr}".join([f"{h}={str(v)}" for h, v in self.hyper_params.items()])
        return f"{clsname}(\n{indentstr}{hpstr})"

    def __repr__(self, indent: int = 0) -> str:
        return self.__str__(indent=indent)

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
