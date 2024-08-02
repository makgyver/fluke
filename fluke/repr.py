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
import time
sys.path.append(".")

from . import GlobalSettings  # NOQA
from .utils import (Configuration, get_class_from_qualified_name)  # NOQA
from .utils.log import get_logger  # NOQA
from .data import DataSplitter  # NOQA
from .data.datasets import Datasets  # NOQA


app = typer.Typer()

# CONST
CONFIG_FNAME = ""


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # ||X.T Y||_F / (||X.T X||_F * ||Y.T Y||_F)
    # compute the gram matrix
    X_gram = X @ X.T
    Y_gram = Y @ Y.T
    XY_gram = X @ Y.T
    # centralize the gram matrix
    # H = I - 1/n 11^T
    # Kc = H K H
    n = X_gram.shape[0]
    I_n = torch.eye(n, device=X.device)
    H = I_n - torch.ones(n, n, device=X.device) / n
    X_gram = H @ X_gram @ H
    Y_gram = H @ Y_gram @ H
    XY_gram = H @ XY_gram @ H
    # compute the norms
    X_norm = torch.norm(X_gram, p='fro')
    Y_norm = torch.norm(Y_gram, p='fro')
    XY_norm = torch.norm(XY_gram, p='fro') ** 2
    return XY_norm / (X_norm * Y_norm)


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
def extract(alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to run'),
            # filename: str = typer.Argument(..., help='Filename template to save the results'),
            n_clients: int = typer.Option(0, help='Number of clients to run'),
            sample_class: int = typer.Option(5, help='Number of samples per class'),
            use_kernel: bool = typer.Option(False, help='Use kernel CKA instead of linear CKA')) \
        -> None:

    cfg = Configuration(CONFIG_FNAME, alg_cfg)
    if n_clients != 0:
        cfg.protocol.n_clients = n_clients

    GlobalSettings().set_seed(cfg.exp.seed)
    GlobalSettings().set_device(cfg.exp.device)
    data_container = Datasets.get(**cfg.data.dataset)
    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg.data.distribution.name,
                                 dist_args=cfg.data.distribution.exclude("name"),
                                 **cfg.data.exclude('dataset', 'distribution'))

    fl_algo_class = get_class_from_qualified_name(cfg.method.name)
    fl_algo = fl_algo_class(cfg.protocol.n_clients,
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

    # exp_id = str(uuid.uuid4())
    exp_id = ".".join(str(cfg).split(".")[3:]) + "_" + time.strftime("%Y%m%dh%H%M%S")
    if not os.path.exists(f"repr_results/{exp_id}"):
        os.makedirs(f"repr_results/{exp_id}")

    rich.print(f"\nSaving files into the folder 'repr_results/{exp_id}'")

    layers = None
    for i in range(cfg.protocol.n_rounds):
        fl_algo.run(n_rounds=1, eligible_perc=1, finalize=False)

        with torch.no_grad():
            for c in track(fl_algo.clients, description="Computing activations"):
                cmodel = AllLayerOutputModel(deepcopy(c.model))
                output = cmodel(X)

                if layers is None:
                    layers = tuple(cmodel.activations_in.keys())

                for ik, (_, v) in enumerate(cmodel.activations_in.items()):

                    if ik == 0 or len(v.shape) > 2:
                        continue

                    if layers[ik - 1] not in repr_clients[i]:
                        repr_clients[i][layers[ik - 1]] = []
                        norm_clients[i][layers[ik - 1]] = []
                    if use_kernel:
                        repr_clients[i][layers[ik - 1]].append(torch.mm(v, v.t()))
                    else:
                        repr_clients[i][layers[ik - 1]].append(v)
                    norm_clients[i][layers[ik - 1]].append(torch.norm(v, dim=1))

                # last layer
                if not layers[-1] in repr_clients[i]:
                    repr_clients[i][layers[-1]] = []
                    norm_clients[i][layers[-1]] = []
                v = output.detach()
                if use_kernel:
                    repr_clients[i][layers[-1]].append(torch.mm(v, v.t()))
                else:
                    repr_clients[i][layers[-1]].append(v)
                norm_clients[i][layers[-1]].append(torch.norm(v, dim=1))

        fname = f"repr_results/{exp_id}/R{i}.pth"
        torch.save({
            'repr_clients': repr_clients[i],
            'norm_clients': norm_clients[i],
            'dists': dists
        }, fname)


@app.command()
def analyze(folder: str = typer.Argument(..., help='Folder containing \
                                         the results to analyze')) -> None:
    # load the results
    files = os.listdir(folder)
    files = {int(f.split(".")[-2].split("_R")[-1]): f for f in files}

    for k in sorted(list(files.keys())):
        v = files[k]
        rich.print(f"Round: {k+1}")
        data = torch.load(f"{folder}/{v}")
        repr_clients = data['repr_clients']
        # norm_clients = data['norm_clients']
        # dists = data['dists']
        # compute the similarity
        layers_sim = []
        for layer, reprs in repr_clients.items():
            rich.print(f"Layer: {layer}")
            # compute the similarity
            layer_sim = []
            for c1 in range(len(reprs)):
                for c2 in range(c1+1, len(reprs)):
                    layer_sim.append(linear_CKA(reprs[c1], reprs[c2]))
            layers_sim.append(torch.mean(torch.tensor(layer_sim)))
            # rich.print(f"Layer: {layer}")
            rich.print(f"Similarity: {layers_sim[-1]}\n")


@ app.callback()
def repr(config: str = typer.Option(CONFIG_FNAME, help="Configuration file")) -> None:
    global CONFIG_FNAME
    CONFIG_FNAME = config


def main() -> Any:
    return app()


if __name__ == "__main__":
    main()
