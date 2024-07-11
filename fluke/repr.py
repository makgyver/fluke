"""`fluke` command line interface."""
from .utils.model import AllLayerOutputModel
from rich.pretty import Pretty
from rich.panel import Panel
from rich.progress import track
import rich
from typing import Any
import typer
import torch
from copy import deepcopy
import sys
import os
import uuid
sys.path.append(".")

from . import GlobalSettings  # NOQA
from .utils import (Configuration, get_class_from_qualified_name, get_logger)  # NOQA
from .data import DataSplitter  # NOQA
from .data.datasets import Datasets  # NOQA


app = typer.Typer()

# CONST
CONFIG_FNAME = ""


def _sample_data(X: torch.Tensor,
                 y: torch.Tensor,
                 sample_x_class: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    samp_X, samp_y = None, None
    for c in set(y.tolist()):
        idx = torch.where(y == c)[0]
        idx = idx[torch.randperm(idx.size(0))[:sample_x_class]]
        if samp_X is None:
            samp_X, samp_y = X[idx], y[idx]
        else:
            samp_X = torch.cat((samp_X, X[idx]), dim=0)
            samp_y = torch.cat((samp_y, y[idx]), dim=0)

    return samp_X, samp_y


@app.command()
def analysis(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
             filename: str = typer.Argument(..., help='Filename template to save the results'),
             n_clients: int = typer.Option(0, help='Number of clients to run'),
             sample_class: int = typer.Option(5, help='Number of samples per class')) -> None:

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_container = Datasets.get(**cfg.data.dataset)
    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg.data.distribution.name,
                                 dist_args=cfg.data.distribution.exclude("name"),
                                 **cfg.data.exclude('dataset', 'distribution'))

    fl_algo_class = get_class_from_qualified_name(cfg.method.name)
    fl_algo = fl_algo_class(cfg.protocol.n_clients if n_clients == 0 else n_clients,
                            data_splitter,
                            cfg.method.hyperparameters)

    log = get_logger(cfg.logger.name, name=str(cfg), **cfg.logger.exclude('name'))
    log.init(**cfg)
    fl_algo.set_callbacks(log)
    rich.print(Panel(Pretty(fl_algo), title="FL algorithm"))

    X = fl_algo.server.test_data.tensors[0].clone()
    y = fl_algo.server.test_data.tensors[1].clone()
    X, y = _sample_data(X, y, sample_class)

    # rounds x layer x clients x activations
    repr_clients = [{} for _ in range(cfg.protocol.n_rounds)]
    norm_clients = [{} for _ in range(cfg.protocol.n_rounds)]

    # clients label distribution
    dists = []
    for c in fl_algo.clients:
        yc = c.train_set.tensors[1]
        ycnt = torch.bincount(yc)
        ycnt = {i: int(ycnt[i]) for i in range(ycnt.size(0))}
        dists.append(ycnt)

    exp_id = str(uuid.uuid4())
    if not os.path.exists(f"repr_results/{exp_id}"):
        os.makedirs(f"repr_results/{exp_id}")

    rich.print(f"\nSaving files into the folder 'repr_results/{exp_id}'")

    splitted = filename.split(".")
    noext = "".join(splitted[:-1])
    ext = splitted[-1]

    layers = None
    for i in range(cfg.protocol.n_rounds):
        fl_algo.run(n_rounds=1, eligible_perc=1, finalize=False)

        with torch.no_grad():
            for c in track(fl_algo.clients, description="Computing activations"):
                cmodel = AllLayerOutputModel(deepcopy(c.model))
                output = cmodel(X)

                if layers is None:
                    layers = tuple(cmodel.activations_in.keys())

                for ik, (k, v) in enumerate(cmodel.activations_in.items()):
                    if ik == 0:
                        continue
                    if k not in repr_clients[i]:
                        repr_clients[i][layers[ik - 1]] = []
                        norm_clients[i][layers[ik - 1]] = []
                    repr_clients[i][layers[ik - 1]].append(torch.mm(v, v.t()))
                    norm_clients[i][layers[ik - 1]].append(torch.norm(v, dim=1))
                repr_clients[i][layers[-1]] = output.detach()
                norm_clients[i][layers[-1]] = torch.norm(output.detach(), dim=1)

        fname = f"repr_results/{exp_id}/{noext}_R{i}.{ext}"
        torch.save({
            'repr_clients': repr_clients[i],
            'norm_clients': norm_clients[i],
            'dists': dists
        }, fname)


@ app.callback()
def repr(config: str = typer.Option(CONFIG_FNAME, help="Configuration file")) -> None:
    global CONFIG_FNAME
    CONFIG_FNAME = config


def main() -> Any:
    return app()


if __name__ == "__main__":
    main()
