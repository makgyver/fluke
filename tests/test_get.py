from __future__ import annotations

import sys

sys.path.append(".")
sys.path.append("..")


import fluke.get  # noqa: E402


def test_get():

    fluke.get.list()
    fluke.get.config(name="exp", outdir="./tmp")
    fluke.get.config(name="fedavg", outdir="./tmp")


if __name__ == "__main__":
    test_get()
