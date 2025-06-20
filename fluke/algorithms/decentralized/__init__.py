from __future__ import annotations

import json
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Collection, Optional, Union

import networkx
import numpy as np
from rich.progress import track
from torch.nn import Module
from torch.utils.data import DataLoader

from ... import DDict, FlukeENV, ObserverSubject  # NOQA
from ...comm import Channel, ChannelObserver  # NOQA
from ...config import OptimizerConfigurator  # NOQA
from ...data import DataSplitter, FastDataLoader  # NOQA
from ...server import EarlyStopping  # NOQA
from ...utils import (
    ClientObserver,
    FederationObserver,
    get_loss,
    get_model,
)  # NOQA
from .client import AbstractDFLClient, GossipClient  # NOQA

__all__ = ["DecentralizedFL", "GossipDFL", "Topology", "client"]


class Topology:

    @staticmethod
    def fully_connected(num_nodes: int) -> Topology:
        """Create a fully connected (complete) graph with num_nodes nodes.

        Args:
            num_nodes (int): The number of nodes in the graph.

        Returns:
            Topology: A fully connected graph.
        """
        return Topology(networkx.complete_graph(num_nodes))

    @staticmethod
    def ring(num_nodes: int) -> Topology:
        """Create a ring topology graph with num_nodes nodes.

        Args:
            num_nodes (int): The number of nodes in the graph.

        Returns:
            Topology: A ring topology graph.
        """
        graph = networkx.DiGraph()
        for i in range(num_nodes):
            graph.add_edge(i, (i + 1) % num_nodes)
        return Topology(graph)

    @staticmethod
    def random(num_nodes: int, p: float = 0.5) -> Topology:
        """Create a random d-regular graph with num_nodes nodes and edge probability p.

        Args:
            num_nodes (int): The number of nodes in the graph.
            p (float): The probability of creating an edge between any two nodes.

        Returns:
            Topology: A random d-regular graph.
        """
        return Topology(networkx.random_regular_graph(d=int(p * num_nodes), n=num_nodes))

    def __init__(self, graph: networkx.Graph):
        self.graph: networkx.Graph = graph

    def __getitem__(self, node: int) -> list[int]:
        """Get the neighbors of a node in the topology graph.

        Args:
            node (int): The node for which to get neighbors.

        Returns:
            list[int]: A list of neighboring nodes.
        """
        return list(self.graph.neighbors(node))

    @property
    def num_nodes(self) -> int:
        """Get the number of nodes in the topology graph.

        Returns:
            int: The number of nodes.
        """
        return self.graph.number_of_nodes()

    def add_node(self, node: int) -> None:
        """Add a node to the topology graph.

        Args:
            node (int): The node to add.
        """
        self.graph.add_node(node)

    def add_edge(self, node1: int, node2: int, weight: float = None) -> None:
        """Add an edge between two nodes in the topology graph.

        Args:
            node1 (int): The first node.
            node2 (int): The second node.
            weight (float, optional): The weight of the edge. Defaults to None.
        """
        if weight is not None:
            self.graph.add_edge(node1, node2, weight=weight)
        else:
            self.graph.add_edge(node1, node2)

    def has_node(self, node: int) -> bool:
        """Check if a node exists in the topology graph.

        Args:
            node (int): The node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        return self.graph.has_node(node)

    def has_edge(self, node1: int, node2: int) -> bool:
        """Check if an edge exists between two nodes in the topology graph.

        Args:
            node1 (int): The first node.
            node2 (int): The second node.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        return self.graph.has_edge(node1, node2)

    def nodes(self) -> list[int]:
        """Get a list of all nodes in the topology graph.

        Returns:
            list[int]: A list of all nodes.
        """
        return list(self.graph.nodes)

    def edges(self) -> list[tuple[int, int]]:
        """Get a list of all edges in the topology graph.

        Returns:
            list[tuple[int, int]]: A list of all edges.
        """
        return list(self.graph.edges)

    def draw(self, **kwargs) -> None:
        """Draw the topology graph using networkx's draw function.

        Args:
            **kwargs: Additional keyword arguments passed to the networkx draw function.
        """
        import matplotlib.pyplot as plt

        pos = networkx.kamada_kawai_layout(self.graph)

        plt.figure(figsize=(8, 6))
        networkx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="lightcoral",
            node_size=300,
            edge_color="gray",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
        )
        plt.title("Topology")
        plt.show()


class DecentralizedFL(ObserverSubject):

    def __init__(
        self,
        n_clients: int,
        data_splitter: DataSplitter,
        hyper_params: DDict | dict[str, Any],
        topology: Topology = None,
        clients: Optional[list[AbstractDFLClient]] = None,
        test_data: Optional[FastDataLoader | DataLoader] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._id: str = str(uuid.uuid4().hex)
        FlukeENV().open_cache(self._id)
        self.channel: Channel = Channel()

        if clients is not None:
            self.clients = clients
            self.n_clients = len(clients)
            if self.n_clients != n_clients:
                warnings.warn(
                    f"Number of clients provided ({self.n_clients}) is different from"
                    + f"the number of clients expected ({n_clients}). Overwriting "
                    + f"the number of clients to {self.n_clients}."
                )

            self.topology: Topology = (
                topology if topology is not None else Topology.fully_connected(n_clients)
            )
            assert (
                self.topology.num_nodes == self.n_clients
            ), "Topology must have the same number of nodes as clients."

        else:
            if isinstance(hyper_params, dict):
                hyper_params = DDict(hyper_params)

            if test_data is not None:
                warnings.warn(
                    "The provided `test_data` will be ignored and replaced by the one"
                    + " generated by the `data_splitter`."
                )
            self.hyper_params = hyper_params
            self.n_clients = n_clients
            (clients_tr_data, clients_te_data), test_data = data_splitter.assign(
                n_clients, hyper_params.client.batch_size
            )

            self.topology: Topology = (
                topology if topology is not None else Topology.fully_connected(n_clients)
            )

            # Federated model
            model = (
                get_model(
                    mname=hyper_params.model,
                    **hyper_params.net_args if "net_args" in hyper_params else {},
                )
                if isinstance(hyper_params.model, str)
                else hyper_params.model
            )

            self.clients = self.init_clients(
                model, self.topology, clients_tr_data, clients_te_data, hyper_params.client
            )

        self.test_data: Optional[FastDataLoader | DataLoader] = test_data
        self.rounds: int = 0
        self._participants: dict[int, set[int]] = defaultdict(set)

        for client in self.clients:
            client.set_channel(self.channel)

    @property
    def id(self) -> str:
        """Get the unique identifier of this instance of the algorithm.

        Returns:
            str: Unique identifier of the instance of the algorithm.
        """
        return str(self._id)

    def get_client_class(self) -> type[AbstractDFLClient]:
        """Get the class of the client to be used in the decentralized FL algorithm.

        Returns:
            type[AbstractDFLClient]: The class of the client.
        """
        return AbstractDFLClient

    def init_clients(
        self,
        model: Module,
        topology: Topology,
        clients_tr_data: list[FastDataLoader],
        clients_te_data: list[FastDataLoader],
        config: DDict | dict[str, Any],
    ) -> list[AbstractDFLClient]:
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=config.optimizer, scheduler_cfg=config.scheduler
        )
        loss = get_loss(config.loss) if isinstance(config.loss, str) else config.loss()
        clients = [
            self.get_client_class()(
                index=i,
                model=deepcopy(model),
                neighbours=topology[i],
                train_set=clients_tr_data[i],
                test_set=clients_te_data[i],
                optimizer_cfg=optimizer_cfg,
                loss_fn=deepcopy(loss),
                **config.exclude("optimizer", "loss", "batch_size", "scheduler"),
            )
            for i in range(self.n_clients)
        ]
        return clients

    def set_callbacks(self, callbacks: Union[callable, Collection[Callable]]) -> None:
        """Set the callbacks for the server the clients and the channel.

        The callbacks are expected to be instances of the :class:`fluke.server.ServerObserver`,
        :class:`fluke.client.ClientObserver` or :class:`fluke.comm.ChannelObserver` classes.
        Each callback will be attached to the corresponding entity.

        Args:
            callbacks (Union[callable, Collection[callable]]): Callbacks to attach to the algorithm.
        """
        if not isinstance(callbacks, Collection):
            callbacks = [callbacks]

        self.channel.attach([c for c in callbacks if isinstance(c, ChannelObserver)])

        for client in self.clients:
            client.attach([c for c in callbacks if isinstance(c, ClientObserver)])

        self.attach([c for c in callbacks if isinstance(c, FederationObserver)])

    def __str__(self, indent: int = 0) -> str:
        algo_hp = f"\n\tmodel={str(self.hyper_params.model)}("
        if "net_args" in self.hyper_params:
            algo_hp += ", ".join([f"{k}={v}" for k, v in self.hyper_params.net_args.items()])
        algo_hp += ")"
        extra_hp = [
            f"{h}={v}"
            for h, v in self.hyper_params.items()
            if h not in ["client", "model", "net_args"]
        ]
        if extra_hp:
            algo_hp += "\n\t" + ",\n\t".join(extra_hp)
        algo_hp = f"\t{algo_hp}," if algo_hp else ""

        if self.clients is None:
            client_str = "Client?"
        else:
            client_str = (
                self.clients[0]
                .__str__(indent=indent + 4)
                .replace("[0](", f"[0-{self.n_clients - 1}](")
            )

        return f"{self.__class__.__name__}[{self._id}]" + f"({algo_hp}\n\t{client_str}\n)"

    def __repr__(self, indent: int = 0) -> str:
        return self.__str__(indent=indent)

    def save(self, path: str, round: int | None = None, *args, **kwargs) -> str:
        """Save the algorithm state into files in the specified directory.

        Note:
            To avoid overwriting previous saved states, the folder name will be suffixed with the
            unique (randomly generated) identifier of the algorithm.

        Args:
            path (str): Path to the folder where the algorithm state will be saved.
            round (int, optional): Round number. Defaults to ``None``.

        Returns:
            str: Path to the folder where the algorithm state was saved.
        """
        path = f"{path}_{self._id}" if path else self._id
        if not os.path.exists(path):
            os.makedirs(path)

        real_rounds = round if round is not None else self.rounds
        prefix = f"r{str(real_rounds).zfill(4)}_"
        for i, client in enumerate(self.clients):
            client.save(os.path.join(path, f"{prefix}client_{i}.pth"))

        with open(os.path.join(path, "federation.json"), "w") as f:
            f.write(json.dumps({"rounds": self.rounds}, indent=4))

        return path

    def load(self, path: str, round: int | None = None) -> None:
        """Load the algorithm state from the specified folder

        Args:
            path (str): Path to the folder where the algorithm state is saved.
            round (int, optional): Round number. Defaults to ``None``.
        """
        if round is not None:
            prefix = f"r{str(round).zfill(4)}_"
        else:
            prefix = ""
            # search in path for the last round
            max_round = -1
            for f in os.listdir(path):
                if f.startswith("r") and f.endswith("client_0.pth"):
                    round = int(f[1:5])
                    max_round = max(max_round, round)
            if max_round != -1:
                prefix = f"r{str(max_round).zfill(4)}_"

        for i, client in enumerate(self.clients):
            client.load(os.path.join(path, f"{prefix}client_{i}.pth"), client.model)

        with open(os.path.join(path, "federation.json"), "r") as f:
            federation_info = json.load(f)
            self.rounds = federation_info.get("rounds", 0)

    def _compute_evaluation(self, round: int | None = None) -> None:
        evaluator = FlukeENV().get_evaluator()

        if FlukeENV().get_eval_cfg().locals:
            clients = (
                self._participants[round]
                if round is not None
                else set.union(*self._participants.values())
            )
            client_evals = {
                index: self.clients[index].evaluate(evaluator, self.test_data) for index in clients
            }
            if round is None:
                round = self.rounds
            self.notify(
                event="server_evaluation", round=round, eval_type="locals", evals=client_evals
            )

    def finalize(self) -> None:
        if self.rounds > 0:
            self._compute_evaluation()
            client_to_eval = set.union(*self._participants.values())
            for cid in track(client_to_eval, "Finalizing federation...", transient=True):
                self.clients[cid].finalize()

            path, freq, g_only = FlukeENV().get_save_options()
            if freq == -1:
                self.save(path, g_only, self.rounds - 1)

        FlukeENV().close_cache()


class GossipDFL(DecentralizedFL):

    def get_client_class(self) -> type[GossipClient]:
        return GossipClient

    def run(self, n_rounds: int, *args, **kwargs) -> None:
        with FlukeENV().get_live_renderer():
            progress_fl = FlukeENV().get_progress_bar("FL")
            progress_client = FlukeENV().get_progress_bar("clients")
            client_x_round = int(self.n_clients)
            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds * client_x_round)
            task_local = progress_client.add_task("[green]Local Training", total=client_x_round)

            total_rounds = self.rounds + n_rounds
            for rnd in range(self.rounds, total_rounds):
                try:
                    self.notify(event="start_round", round=rnd + 1, global_model=None)

                    perm_idx = np.random.permutation(self.n_clients)
                    for c, cid in enumerate(perm_idx):
                        client = self.clients[cid]
                        if client.is_active(rnd):
                            client.local_update(rnd + 1)
                            self._participants[rnd + 1].add(client.index)
                        progress_client.update(task_id=task_local, completed=c + 1)
                        progress_fl.update(task_id=task_rounds, advance=1)

                    self._compute_evaluation(rnd + 1)
                    self.notify(event="end_round", round=rnd + 1)
                    self.rounds += 1

                    path, freq, _ = FlukeENV().get_save_options()
                    if freq > 0 and (rnd + 1) % freq == 0:
                        self.save(path, rnd + 1)

                except KeyboardInterrupt:
                    self.notify(event="interrupted")
                    break

                except EarlyStopping:
                    self.notify(event="early_stop")
                    break

            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

        self.finalize()
        self.notify(event="finished", round=self.rounds + 1)
