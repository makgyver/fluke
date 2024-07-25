![Coveralls](https://img.shields.io/coverallsCoverage/github/makgyver/fluke?style=for-the-badge&logo=coveralls)
<a href="https://makgyver.github.io/fluke"><img src="https://img.shields.io/github/actions/workflow/status/makgyver/fluke/doc-publish.yml?style=for-the-badge&label=DOCUMENTATION"/></a>
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fluke-fl?style=for-the-badge&logo=python&logoColor=yellow)
![GitHub License](https://img.shields.io/github/license/makgyver/fluke?style=for-the-badge)


# **fluke**: **f**ederated **l**earning **u**tility framewor**k** for **e**xperimentation and research

``fluke`` is a Python package that provides a framework for federated learning research. It is designed to be modular and extensible, allowing researchers to easily implement and test new federated learning algorithms. ``fluke`` provides a set of pre-implemented state-of-the-art federated learning algorithms that can be used as a starting point for research or as a benchmark for comparison.

## Installation

``fluke`` is a Python package that can be installed via pip. To install it, you can run the following command:

```bash
pip install fluke-fl
```

## Run a federated algorithm

To run an algorithm in ``fluke`` you need to create two configuration files:
- `EXP_CONFIG`: the experiment configuration file (independent from the algorithm);
- `ALG_CONFIG`: the algorithm configuration file;

Then, you can run the following command:

```bash
fluke --config=EXP_CONFIG federation ALG_CONFIG
```

You can find some examples of these files in the [configs](https://github.com/makgyver/fluke/tree/main/configs) folder of the repository.

### Example
Let say you want to run the classic `FedAvg` algorithm on the `MNIST` dataset. Then, using the configuration files [exp.yaml](https://github.com/makgyver/fluke/blob/main/configs/exp.yaml) and [fedavg.yaml](https://github.com/makgyver/fluke/blob/main/configs/fedavg.yaml), you can run the following command:

```bash
fluke --config=path_to_folder/exp.yaml federation path_to_folder/fedavg.yaml
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

## Contributing

If you have suggestions for how ``fluke`` could be improved, or want to report a bug, open an issue! We'd love all and any contributions.

For more, check out the [Contributing Guide](CONTRIBUTING.md).


## Authors

- [**Mirko Polato**](https://makgyver.github.io) - *Idealization*, *Design*, *Development*, *Testing*, and *Documentation*
- [**Roberto Esposito**]() - *Testing*
- [**Samuele Fonio**]() - *Testing*
