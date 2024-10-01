import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import seaborn as sns
import torch
import typer
import yaml
from rich.panel import Panel
from rich.pretty import Pretty
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(".")
sys.path.append("..")

from . import DDict, GlobalSettings  # NOQA
from .algorithms import CentralizedFL  # NOQA
from .data import DataSplitter, FastDataLoader  # NOQA
from .data.datasets import Datasets  # NOQA
from .evaluation import ClassificationEval  # NOQA
from .utils import get_class_from_qualified_name  # NOQA
from .utils.log import Log  # NOQA
from .utils.model import AllLayerOutputModel  # NOQA

app = typer.Typer()


@app.command()
def main(config_exp: str = typer.Argument(..., help='Config file for the experiment to run'),
         config_alg: str = typer.Argument(..., help='Config file for the algorithm to run'),
         plot_type: str = typer.Option("pca", help="Type of 2D projection to use ('pca', 'tsne')"),
         load_path: str = typer.Option(None,
                                       help='Directory containing the checkpoint files')):

    with open(config_exp) as f:
        cfg_exp = DDict(**yaml.safe_load(f))

    with open(config_alg) as f:
        cfg_alg = DDict(**yaml.safe_load(f))

    GlobalSettings().set_seed(cfg_exp.exp.seed)
    GlobalSettings().set_device(cfg_exp.exp.device)
    data_container = Datasets.get(**cfg_exp.data.dataset)
    evaluator = ClassificationEval(eval_every=cfg_exp.eval.eval_every,
                                   n_classes=data_container.num_classes)
    GlobalSettings().set_evaluator(evaluator)
    GlobalSettings().set_eval_cfg(cfg_exp.eval)

    data_splitter = DataSplitter(dataset=data_container,
                                 distribution=cfg_exp.data.distribution.name,
                                 dist_args=cfg_exp.data.distribution.exclude("name"),
                                 **cfg_exp.data.exclude('dataset', 'distribution'))

    fl_algo_class = get_class_from_qualified_name(cfg_alg.name)
    fl_algo: CentralizedFL = fl_algo_class(cfg_exp.protocol.n_clients,
                                           data_splitter,
                                           cfg_alg.hyperparameters)
    # plot_distribution(fl_algo.clients)
    log = Log()
    log.init(**cfg_exp, **cfg_alg)
    fl_algo.set_callbacks(log)

    rich.print(Panel(Pretty(fl_algo), title="FL algorithm"))

    if load_path is not None:
        fl_algo.load(load_path)
    else:
        try:
            fl_algo.run(cfg_exp.protocol.n_rounds, cfg_exp.protocol.eligible_perc)
        except KeyboardInterrupt:
            fl_algo.server.finalize()

    visualize_repr(fl_algo.server.test_set,
                   fl_algo.server.model,
                   type=plot_type,
                   layer_name="_encoder.fc2")


def visualize_repr(data_loader: FastDataLoader,
                   model: torch.nn.Module,
                   type: Literal["pca", "tsne"] = "pca",
                   layer_name: str = None,
                   layer_index: int = None):

    assert layer_name or layer_index, "One between 'layer_name' and 'layer_index' must be valued."
    assert type in ("pca", "tsne"), "Argument 'type' must be 'pca' or 'tsne'."

    if layer_name is None:
        layer_name = list(model.state_dict().keys())[layer_index]

    proxy_model = AllLayerOutputModel(model=model)

    reprs = []
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    for X, y in data_loader:
        out = proxy_model(X)
        loss = criterion(out, y)
        repr = proxy_model.activations_out[layer_name]
        reprs.append(repr)
        losses.append(loss.item())

    labels = data_loader.tensors[1]
    reprs = torch.cat(reprs)[:labels.shape[0]]
    print("Mean loss:", np.mean(losses))

    if type == "pca":
        pca = PCA(n_components=2)
        proj = pca.fit_transform(reprs)

    else:  # tsne
        tsne = TSNE(n_components=2)
        proj = tsne.fit_transform(reprs, labels)

    data = np.vstack((proj.T, labels)).T
    df = pd.DataFrame(data=data, columns=("Dim_1", "Dim_2", "label"))
    sns.FacetGrid(df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    plt.show()


if __name__ == "__main__":
    app()
