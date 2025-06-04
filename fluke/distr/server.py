import sys
import time
from copy import deepcopy
from queue import SimpleQueue
from typing import Any, Iterable

import torch.multiprocessing as mp

sys.path.append(".")
sys.path.append("..")

from .. import FlukeENV  # NOQA
from ..config import ConfigurationError  # NOQA
from ..data import FastDataLoader  # NOQA
from ..evaluation import Evaluator  # NOQA
from ..server import EarlyStopping, Server  # NOQA
from .client import ParallelClient  # NOQA

__all__ = ["ParallelServer"]


def _train_worker(
    client: ParallelClient,
    model: dict,
    round: int,
    device: str,
    prefit: bool,
    postfit: bool,
    evaluator: Evaluator,
    result_dict: dict,
) -> None:
    result_dict.put(
        client.local_update(
            current_round=round,
            current_model_sd=model,
            device=device,
            prefit=prefit,
            postfit=postfit,
            evaluator=evaluator,
        )
    )


class ParallelServer(Server):
    """Server for parallel federated learning using multiple GPUs.

    This class is a subclass of :class:`Server` that allows the use of multiple GPUs - one
    for each client. It uses multiprocessing to handle the training of clients in parallel.
    It is designed to work with the `torch.multiprocessing` module, which allows for
    parallel execution of client training processes across multiple GPUs.
    """

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)

        if len(FlukeENV().get_device_ids()) <= 1:
            raise ConfigurationError(
                "ParallelServer requires at least 2 GPUs."
                + "Please set a list of devices in the config file."
            )
        self.device = FlukeENV().get_device()
        mp.set_start_method("spawn")
        self.manager = mp.Manager()
        self.result_dict = self.manager.dict()
        self.processes = []
        self.running_processes = []

    def _clients_local_fit(self, eligible: Iterable[ParallelClient]) -> dict[int, dict]:
        console = FlukeENV().get_progress_bar("clients").console
        trainer_queue = SimpleQueue()
        for client in eligible:
            trainer_queue.put(client)

        self.available_gpus = [f"cuda:{i}" for i in FlukeENV().get_device_ids()]
        result_queue = self.manager.Queue()
        current_model = deepcopy(self.model.state_dict())
        current_round = self.rounds + 1
        result_dict = {}
        while not trainer_queue.empty() or self.running_processes:
            while self.available_gpus and not trainer_queue.empty():
                device = self.available_gpus.pop(0)
                client = trainer_queue.get()
                p = mp.Process(
                    target=_train_worker,
                    args=(
                        client,
                        current_model,
                        current_round,
                        device,
                        FlukeENV().get_eval_cfg().pre_fit,
                        FlukeENV().get_eval_cfg().post_fit,
                        FlukeENV().get_evaluator(),
                        result_queue,
                    ),
                )
                p.start()
                self.running_processes.append((p, device))

            # Check if any process finished
            for p, gpu_id in self.running_processes.copy():
                if not p.is_alive():
                    p.join()
                    self.running_processes.remove((p, gpu_id))
                    self.available_gpus.append(gpu_id)

            # Process results if any
            while not result_queue.empty():
                model, loss, idx, results = result_queue.get()
                result_dict[idx] = (model, results)
                console.log(f"Client {idx} local update complete [loss={loss}]")
                # console.log(f"Client {idx} evaluation \n{results}")

            time.sleep(1)  # Avoid busy waiting

        return result_dict

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        if test_set is not None:
            return evaluator.evaluate(self.rounds + 1, self.model, test_set, device=self.device[0])
        return {}

    def fit(
        self, n_rounds: int = 10, eligible_perc: float = 0.1, finalize: bool = True, **kwargs
    ) -> None:

        total_rounds = self.rounds + n_rounds
        for rnd in range(self.rounds, total_rounds):
            try:
                self.notify(event="start_round", round=rnd + 1, global_model=self.model)
                eligible = self.get_eligible_clients(eligible_perc)
                self.notify(event="selected_clients", round=rnd + 1, clients=eligible)
                client_modopts = self._clients_local_fit(eligible)
                for client in eligible:
                    self._participants.add(client.index)
                    c_modopt, c_results = client_modopts[client.index]
                    self.clients[client.index]._modopt = c_modopt

                    for k, evals in c_results:
                        self._notify(
                            "client_evaluation",
                            round=rnd + 1,
                            client_id=client.index,
                            phase=k,
                            results=evals,
                        )

                client_models = [self.clients[c.index].model for c in eligible]
                self.aggregate(eligible, client_models)
                self._compute_evaluation(rnd, eligible)
                self.notify(event="end_round", round=rnd + 1)
                self.rounds += 1

            except KeyboardInterrupt:
                self.notify(event="interrupted")
                break

            except EarlyStopping:
                self.notify(event="early_stop")
                break

        if finalize:
            self.finalize()
