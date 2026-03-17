from typing import Sequence, Any

from fluke import FlukeENV, DDict
from fluke.algorithms import CentralizedFL
from fluke.client import Client
from fluke.config import OptimizerConfigurator
from fluke.data import FastDataLoader
from fluke.server import EarlyStopping, Server
from fluke.utils import get_loss
from playground.clientSL import ClientSL
from playground.serverSL import ServerSL


class VanillaSL(CentralizedFL):

    def get_client_class(self):
        return ClientSL

    def get_server_class(self):
        return ServerSL

    def init_server(self, model:Any, data: FastDataLoader, config:DDict) -> Server:
        """
        model = {"client": client_model, "server": server_model}, dovrei cambiarlo con un DDict
        """
        if not isinstance(model, dict):
            raise TypeError(
                "Per VanillaSL, hyper_params.model deve essere un dict "
                "con chiavi 'client' e 'server'."
            )
        if "client" not in model or "server" not in model:
            raise ValueError("Le chiavi richieste sono 'client' e 'server'.")

        self._fix_opt_cfg(config.optimizer)
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=config.optimizer,
            scheduler_cfg=config.scheduler,
        )
        loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss()

        server = self.get_server_class()(
            model=model["server"],
            client_model=model["client"],
            test_set=data,
            clients=self.clients,
            optimizer_cfg=optimizer_cfg,
            loss_fn=loss,
            **config.exclude("optimizer", "loss", "scheduler"),
        )

        if FlukeENV().get_save_options()[0] is not None:
            server.attach(self)

        return server

    def run(self, n_rounds: int, eligible_perc: float, finalize: bool = True, **kwargs) -> None:
        with FlukeENV().get_live_renderer():
            progress_fl = FlukeENV().get_progress_bar("FL")
            progress_client = FlukeENV().get_progress_bar("clients")
            client_x_round = int(self.n_clients * eligible_perc)
            task_rounds = progress_fl.add_task("[red]SL Rounds", total=n_rounds * client_x_round)
            task_local = progress_client.add_task("[green]Client Updates", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            self._round_zero()
            for rnd in range(self.rounds, total_rounds):
                try:
                    # server.model non è il global_model.
                    # Comunque non capisco a cosa serve questo parametro.
                    # Eventualmente dovrebbe essere:
                    # global_model = torch.nn.Sequential(self.server.client_model, self.server.model)
                    self.notify(event="start_round", round=rnd + 1, global_model=self.server.model)

                    eligible = self.server.get_eligible_clients(eligible_perc)

                    self.notify(event="selected_clients", round=rnd + 1, clients=eligible)

                    for c, client in enumerate(eligible):
                        # passa al client il modello client-side corrente
                        self.server.send_client_model(client.index)

                        # local_update si occupa di fare il training vero e proprio,
                        # fa call dirette al server quindi andrà cambiata
                        client_loss = client.local_update(rnd + 1, server=self.server)

                        # recupera dal client il modello client-side aggiornata,
                        # che verrà poi passato al client successivo
                        self.server.receive_client_model(client.index)

                        progress_client.update(task_id=task_local, completed=c + 1)
                        progress_fl.update(task_id=task_rounds, advance=1)

                    self._compute_evaluation_full_model(rnd)
                    self.notify(event="end_round", round=rnd + 1)
                    self.rounds += 1

                    path, freq, g_only = FlukeENV().get_save_options()
                    if freq > 0 and rnd % freq == 0:
                        self.save(path, g_only, rnd)

                except KeyboardInterrupt:
                    self.notify(event="interrupted")
                    break

                except EarlyStopping:
                    self.notify(event="early_stop", round=self.rounds + 1)
                    break

            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

        if finalize:
            self.finalize()

        self.notify(event="finished", round=self.rounds + 1)


    def finalize(self) -> None:
        if self.rounds > 0:
            self._compute_evaluation_full_model(self.rounds - 1)

        FlukeENV().close_cache()

    def _compute_evaluation(self, round: int, eligible: Sequence[Client]) -> None:
        return None

    def _compute_evaluation_full_model(self, round: int) -> None:
        evaluator = FlukeENV().get_evaluator()
        if FlukeENV().get_eval_cfg().server:
            evals = self.server.evaluate_full_model(evaluator, round=round + 1)
            self.notify(event="server_evaluation", round=round + 1, eval_type="global", evals=evals)
