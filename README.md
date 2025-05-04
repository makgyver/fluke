![Coveralls](https://img.shields.io/coverallsCoverage/github/makgyver/fluke?style=for-the-badge&logo=coveralls)
<a href="https://makgyver.github.io/fluke"><img src="https://img.shields.io/github/actions/workflow/status/makgyver/fluke/doc-publish.yml?style=for-the-badge&label=DOCUMENTATION"/></a>
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fluke-fl?style=for-the-badge&logo=python&logoColor=yellow)
![GitHub License](https://img.shields.io/github/license/makgyver/fluke?style=for-the-badge)
[![arXiv](https://img.shields.io/badge/arxiv-2412.15728-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2412.15728)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

# **fluke**: **f**ederated **l**earning **u**tility framewor**k** for **e**xperimentation and research

``fluke`` is a Python package that provides a framework for federated learning research. It is designed to be modular and extensible, allowing researchers to easily implement and test new federated learning algorithms. ``fluke`` provides a set of pre-implemented state-of-the-art federated learning algorithms that can be used as a starting point for research or as a benchmark for comparison.

## Installation

### Pypi

``fluke`` is a Python package that can be installed via pip. To install it, you can run the following command:

```bash
pip install fluke-fl
```

### 🐳 Docker

You can use this library directly inside a Docker container — no installation needed on your local machine.

```bash
docker build -t fluke_container .
```

Then, you can run an interactive session with

```bash
docker run --rm fluke_container fluke [ARGS]
```

where `ARGS` are the arguments you want to pass to the `fluke` command as described in the next section.

## Run a federated algorithm

To run an algorithm in ``fluke`` you need to create two configuration files:
- `EXP_CONFIG`: the experiment configuration file (independent from the algorithm);
- `ALG_CONFIG`: the algorithm configuration file;

Then, you can run the following command:

```bash
fluke federation EXP_CONFIG ALG_CONFIG
```

You can find some examples of these files in the [configs](https://github.com/makgyver/fluke/tree/main/configs) folder of the repository.

### Example
Let say you want to run the classic `FedAvg` algorithm on the `MNIST` dataset. Then, using the configuration files [exp.yaml](https://github.com/makgyver/fluke/blob/main/configs/exp.yaml) and [fedavg.yaml](https://github.com/makgyver/fluke/blob/main/configs/fedavg.yaml), you can run the following command:

```bash
fluke federation path_to_folder/exp.yaml path_to_folder/fedavg.yaml
```

where `path_to_folder` is the path to the folder containing the configuration files.


## Documentation

The documentation for ``fluke`` can be found [here](https://makgyver.github.io/fluke). It contains detailed information about the package, including how to install it, how to run an experiment, and how to implement new algorithms.

## Tutorials

Tutorials on how to use ``fluke`` can be found [here](https://makgyver.github.io/fluke/tutorials.html). In the following, you can find some quick tutorials to get started with ``fluke``:

- Getting started with `fluke` API [![Open in Colab](https://img.shields.io/badge/Open_in_Colab-blue?style=flat-square&logo=google-colab&logoColor=yellow&labelColor=gray)
](https://colab.research.google.com/github/makgyver/fluke/blob/main/tutorials/fluke_quick_api.ipynb)
- Run your algorithm in ``fluke`` [![Open in Colab](https://img.shields.io/badge/Open_in_Colab-blue?style=flat-square&logo=google-colab&logoColor=yellow&labelColor=gray)
](https://colab.research.google.com/github/makgyver/fluke/blob/main/tutorials/fluke_custom_alg.ipynb)
- Use your own model with `fluke` [![Open in Colab](https://img.shields.io/badge/Open_in_Colab-blue?style=flat-square&logo=google-colab&logoColor=yellow&labelColor=gray)
](https://colab.research.google.com/github/makgyver/fluke/blob/main/tutorials/fluke_custom_nn.ipynb)
- Add your dataset and use it with ``fluke`` [![Open in Colab](https://img.shields.io/badge/Open_in_Colab-blue?style=flat-square&logo=google-colab&logoColor=yellow&labelColor=gray)
](https://colab.research.google.com/github/makgyver/fluke/blob/main/tutorials/fluke_custom_dataset.ipynb)
- Add your custom evaluation metric in ``fluke`` [![Open in Colab](https://img.shields.io/badge/Open_in_Colab-blue?style=flat-square&logo=google-colab&logoColor=yellow&labelColor=gray)
](https://colab.research.google.com/github/makgyver/fluke/blob/main/tutorials/fluke_custom_eval.ipynb)

## Contributing

If you have suggestions for how ``fluke`` could be improved, or want to report a bug, open an issue! We'd love all and any contributions.

For more, check out the [Contributing Guide](CONTRIBUTING.md).

## `fluke` @ ECML-PKDD 2024

`fluke` has been presented at the [ECML-PKDD 2024](https://ecmlpkdd2024.org/) conference in the [Workshop on Advancements in Federated Learning](https://wafl2024.di.unito.it). The slides of the presentation are available [here](slides/fluke_ecmlpkdd2024.pdf).

## Citing `fluke`
``fluke`` is a research tool, and we kindly ask you to cite it in your research papers if you use it. You can use the following BibTeX entry:

```bibtex
@misc{polato2024fluke,
      title={fluke: Federated Learning Utility frameworK for Experimentation and research}, 
      author={Mirko Polato},
      year={2024},
      eprint={2412.15728},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.15728}, 
}
```

## Authors and main contributors

- [**Mirko Polato**](https://makgyver.github.io) - *Idealization*, *Design*, *Development*, *Testing*, *Tutorial*, and *Documentation*
- [**Roberto Esposito**](https://github.com/boborbt) - *Testing*
- [**Samuele Fonio**](https://github.com/samuelefonio) - *Testing*, *Tutorial*
- [**Edoardo Oglietti**](https://github.com/00-uno-00) - *Testing*
