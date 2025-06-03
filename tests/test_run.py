from __future__ import annotations

import sys

import pytest

sys.path.append(".")
sys.path.append("..")


import fluke.run  # noqa: E402


def test_federation():
    fluke.run.federation(
        exp_cfg="./tests/configs/exp.yaml",
        alg_cfg="./tests/configs/alg/fedavg.yaml",
        overrides=["exp.seed=98765"],
        resume=None,
    )


def test_clients_only():
    fluke.run.clients_only(
        exp_cfg="./tests/configs/exp.yaml", alg_cfg="./tests/configs/alg/fedavg.yaml", epochs=1
    )


def test_centralized():
    fluke.run.centralized(
        exp_cfg="./tests/configs/exp.yaml", alg_cfg="./tests/configs/alg/fedavg.yaml", epochs=1
    )


def test_sweep():
    fluke.run.sweep(
        exp_cfg="./tests/configs/exp.yaml",
        alg_cfgs=["./tests/configs/alg/fedavg.yaml", "./tests/configs/alg/fedprox.yaml"],
    )


def test_run_others():
    fluke.run.run(True)
    with pytest.raises(Exception):
        fluke.run.version_callback(True)


if __name__ == "__main__":
    test_federation()
    test_clients_only()
    test_centralized()
