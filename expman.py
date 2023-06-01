import sys; sys.path.append("fl_bench")

import os
import glob
import json
import typer

from fl_bench.utils import DatasetsEnum, DistributionEnum
from fl_bench.algorithms import FedAdaboostAlgorithmsEnum


app = typer.Typer()

# CONSTANTS
VALIDATION_SPLIT = 0.1
STANDARDIZE = True
DEVICE = "auto"
BASE_SEED = 98765
TAGS = ["PAMI"]


@app.command()
def make():
    alg_config_files = glob.glob('alg_configs/*.json')
    exp_config_files = glob.glob('exp_configs/*.json')

    with open("launch_exps.sh", "w") as f:
        for alg_config_file in alg_config_files:
            for exp_config_file in exp_config_files:
                alg_cfg = json.load(open(alg_config_file))
                boost = "-boost" if FedAdaboostAlgorithmsEnum.contains(alg_cfg["name"]) else ""
                f.write(f"python fl_bench/main.py --config={exp_config_file} run{boost} {alg_config_file}\n")


@app.command()
def gen(
        # alg_cfg: str = typer.Argument(..., help='Config file for the algorithm to test'),
        dataset: str = typer.Argument(..., help='Dataset to use'),
        n_clients: int = typer.Option(10, help='Number of clients'),
        n_rounds: int = typer.Option(100, help='Number of rounds'),
        elig: float = typer.Option(1.0, help='Fraction of eligible clients'),
        repeats: int = typer.Option(5, help='Number of repetitions')
    ):
    
    # alg_name = alg_cfg["name"]
    # if not FedAdaboostAlgorithmsEnum.contains(alg_name) and not FedAlgorithmsEnum(alg_name):
    #     raise ValueError(f"Unknown algorithm {alg_name}.")

    if not DatasetsEnum.contains(dataset):
        raise ValueError(f"Unknown dataset {dataset}.")

    data_cfgs = {}
    for iidness in DistributionEnum:
        # skip classwise quantity skewed
        if iidness == DistributionEnum.CLASSWISE_QUANTITY_SKEWED:
            continue

        data_cfg = {
            "dataset": dataset,
            "standardize": STANDARDIZE, #CHECK ME
            "distribution": iidness.value,
            "validation_split": VALIDATION_SPLIT,
            "sampling_perc": 1.0
        }

        data_cfgs[iidness.value] = data_cfg

    proto_cfg = {
        "n_clients": n_clients,
    	"n_rounds": n_rounds,
    	"eligible_perc": elig
    }

    seeds = list(range(BASE_SEED, BASE_SEED + repeats))

    exp_cfgs = {}
    for seed in seeds:
        exp_cfg = {
            "seed": seed,
            "device": DEVICE,
            "logger": "wandb",
            "wandb_params": {
                "project": "fl-bench",
                "entity": "mlgroup",
                "tags": TAGS
            }
        }

        exp_cfgs[seed] = exp_cfg
    
    cfg_files = []
    for _, data_cfg in data_cfgs.items():
        for _, exp_cfg in exp_cfgs.items():
            
            data_container = DatasetsEnum(data_cfg["dataset"]).klass()()
            if data_container.num_classes == 2 and data_cfg["distribution"] in {"lblqnt", "dir", "path"}:
                continue

            cfg = {
                "protocol": proto_cfg,
                "data": data_cfg,
                "exp": exp_cfg
            }

            cfg_files.append(cfg)
    
    if not os.path.exists("exp_configs"):
        os.makedirs("exp_configs")

    for cfg_file in cfg_files:
        fname = f"exp_configs/{dataset}_{cfg_file['data']['distribution']}_{cfg_file['exp']['seed']}.json"
        print(f"Saving {fname}")
        json.dump(cfg_file, open(fname, "w"), indent=4)


if __name__ == '__main__':
    app()