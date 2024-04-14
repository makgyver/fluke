from __future__ import annotations

import numpy as np
from typing import Any, Dict, Sequence
from collections import OrderedDict

import torch
from torch import device
from torch.nn import Module

from fl_bench.evaluation import ClassificationEval

from .utils import DDict
from .comm import Channel, Message
from .data import FastTensorDataLoader
from . import GlobalSettings, ObserverSubject

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .client import Client


class Server(ObserverSubject):
    """Standard Server for Federated Learning.

    This class is the base class for all servers in `FL-bench`. It implements the basic
    functionalities of a federated learning server. The default behaviour of this server is based
    on the Federated Averaging algorithm. The server is responsible for coordinating the learning
    process, selecting the clients for each round, sending the global model to the clients, and
    aggregating the models of the clients. The server also evaluates the model server-side (if the
    test data is provided) and sends the final model to the clients.

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
        test_data (FastTensorDataLoader): The test data to evaluate the model. If None, the model
          will not be evaluated server-side.

    Args:
        model (torch.nn.Module): The federated model to be trained.
        test_data (FastTensorDataLoader): The test data to evaluate the model.
        clients (Sequence[Client]): The clients that will participate in the federated learning
          process.
        eval_every (int, optional): The number of rounds between evaluations. Defaults to 1.
        weighted (bool, optional): A boolean indicating if the clients should be weighted by the
          number of samples when aggregating the models. Defaults to False.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 test_data: FastTensorDataLoader,
                 clients: Sequence[Client],
                 eval_every: int = 1,
                 weighted: bool = False):
        super().__init__()
        self.hyper_params = DDict({
            "weighted": weighted
        })
        self.device: device = GlobalSettings().get_device()
        self.model: Module = model
        self.clients: Sequence[Client] = clients
        self.channel: Channel = Channel()
        self.n_clients: int = len(clients)
        self.rounds: int = 0
        self.test_data: FastTensorDataLoader = test_data
        self._eval_every: int = eval_every

        for client in self.clients:
            client.set_server(self)

    # def _local_train(self, client: Client) -> None:
    #     self.channel.send(Message((client.fit, {}), "__action__", self), client)

    def _broadcast_model(self, eligible: Sequence[Client]) -> None:
        self.channel.broadcast(Message(self.model, "model", self), eligible)

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1) -> None:
        """Run the federated learning algorithm.

        The default behaviour of this method is to run the Federated Averaging algorithm. The server
        selects a percentage of the clients to participate in each round, sends the global model to
        the clients, which compute the local updates and send them back to the server. The server
        aggregates the models of the clients and repeats the process for a number of rounds.
        During the process, the server evaluates the global model and the local model every
        `eval_every` rounds.

        Args:
            n_rounds (int, optional): The number of rounds to run. Defaults to 10.
            eligible_perc (float, optional): The percentage of clients that will be selected for
              each round. Defaults to 0.1.
        """
        with GlobalSettings().get_live_renderer():
            progress_fl = GlobalSettings().get_progress_bar("FL")
            progress_client = GlobalSettings().get_progress_bar("clients")
            client_x_round = int(self.n_clients*eligible_perc)
            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            task_local = progress_client.add_task("[green]Local Training", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            for round in range(self.rounds, total_rounds):
                self._notify_start_round(round + 1, self.model)
                eligible = self._get_eligible_clients(eligible_perc)
                self._notify_selected_clients(round + 1, eligible)
                self._broadcast_model(eligible)
                for c, client in enumerate(eligible):
                    client.fit()
                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)
                self._aggregate(eligible)

                client_evals = []
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

        self._finalize()

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the global federated model on the :`test_set`.

        If the test set is not set, the method returns an empty dictionary.

        Warning:
            To date, only classification tasks are supported.

        Returns:
            Dict[str, float]: The evaluation results. The keys are the metrics and the values are
                the results.
        """
        if self.test_data is not None:
            return ClassificationEval(self.clients[0].hyper_params.loss_fn,
                                      self.model.output_size,
                                      device=GlobalSettings().get_device()).evaluate(self.model,
                                                                                     self.test_data)
        return {}

    def _finalize(self) -> None:
        """Finalize the federated learning process.

        The finalize method is called at the end of the federated learning process. It is used to
        send the final model to the clients and to notify the observers that the process has ended.
        """
        self._broadcast_model(self.clients)
        for client in self.clients:
            client._receive_model()
        client_evals = [client.evaluate() for client in self.clients]
        self._notify_finalize(client_evals if client_evals[0] else None)

    def _get_eligible_clients(self, eligible_perc: float) -> Sequence[Client]:
        """Get the clients that will participate in the current round.

        Args:
            eligible_perc (float): The percentage of clients that will be selected.

        Returns:
            Sequence[Client]: The clients that will participate in the current round.
        """
        if eligible_perc == 1:
            return self.clients
        n = int(self.n_clients * eligible_perc)
        return np.random.choice(self.clients, n)

    def _get_client_models(self, eligible: Sequence[Client], state_dict: bool = True):
        """Retrieve the models of the clients.

        This method assumes that the clients have already sent their models to the server.

        Args:
            eligible (Sequence[Client]): The clients that will participate in the aggregation.
            state_dict (bool, optional): If True, the method returns the state_dict of the models.
              Otherwise, it returns the models. Defaults to True.

        Returns:
            List[torch.nn.Module]: The models of the clients.
        """
        client_models = [self.channel.receive(self, client, "model").payload
                         for client in eligible]
        if state_dict:
            return [m.state_dict() for m in client_models]
        return client_models

    def _get_client_weights(self, eligible: Sequence[Client]):
        """Get the weights of the clients for the aggregation.

        The weights are calculated based on the number of samples of each client.
        If the hyperparameter `weighted` is True, the clients are weighted by their number of
        samples. Otherwise, all clients have the same weight.

        Note:
            The computation of the weights do not adhere to the "best-practices" of FL-bench
            because the server should not have direct access to the number of samples of the
            clients. Thus, the computation of the weights should be done communicating with the
            clients through the channel, but for simplicity, we are not following this practice
            here. However, the communication overhead is minimal and does not affect the logged
            performance.

        Args:
            eligible (Sequence[Client]): The clients that will participate in the aggregation.

        Returns:
            List[float]: The weights of the clients.
        """
        if "weighted" in self.hyper_params.keys() and self.hyper_params.weighted:
            num_ex = [client.n_examples for client in eligible]
            tot_ex = sum(num_ex)
            return [num_ex[i] / tot_ex for i in range(len(eligible))]
        else:
            return [1. / len(eligible)] * len(eligible)

    def _aggregate(self, eligible: Sequence[Client]) -> None:
        """Aggregate the models of the clients.

        The aggregation is done by averaging the models of the clients. If the hyperparameter
        `weighted` is True, the clients are weighted by their number of samples.
        The method directly updates the model of the server.

        Args:
            eligible (Sequence[Client]): The clients that will participate in the aggregation.
        """
        avg_model_sd = OrderedDict()
        clients_sd = self._get_client_models(eligible)
        weights = self._get_client_weights(eligible)
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                if "num_batches_tracked" in key:
                    # avg_model_sd[key] = clients_sd[0][key].clone()
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
                          evals: Dict[str, float],
                          client_evals: Sequence[Any]) -> None:
        """Notify the observers that a round has ended.

        Args:
            round (int): The round number.
            global_model (Any): The current global model.
            data (FastTensorDataLoader): The test data.
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

    def _notify_send(self, msg: Message) -> None:
        """Notify the observers that a message has been sent.

        Args:
            msg (Message): The message sent.
        """
        for observer in self._observers:
            observer.send(msg)

    def _notify_receive(self, msg: Message) -> None:
        """Notify the observers that a message has been received.

        Args:
            msg (Message): The message received.
        """
        for observer in self._observers:
            observer.receive(msg)

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
