# Installation

`fluke` is available on [PyPI](https://pypi.org/project/fluke-fl/).

To install ``fluke``, you can use `pip`:

```bash
pip install fluke-fl
```

When installing `fluke`, the following dependencies are installed:

- [`numpy`](https://numpy.org/doc/): for numerical operations;
- [`pandas`](https://pandas.pydata.org/pandas-docs/stable/): for handling data;
- [`scikit-learn`](https://scikit-learn.org/stable/): for specific functionalities, like Kernel Density Estimation;
- [`torch`](https://pytorch.org/docs/stable/index.html): for managing tensors and neural networks;
- [`torchmetrics`](https://torchmetrics.readthedocs.io/en/stable/): for metrics computation;
- [`torchvision`](https://pytorch.org/vision/stable/index.html): for datasets and models;
- [`rich`](https://rich.readthedocs.io/en/stable/): for rich console output;
- [`typer`](https://typer.tiangolo.com/): for command-line interface;
- [`wandb`](https://docs.wandb.ai/): for experiment tracking;
- ['datasets'](https://huggingface.co/docs/datasets/): for datasets loading;
- [`psutil`](https://psutil.readthedocs.io/en/latest/): for system monitoring;
- [`tensorboard`](https://www.tensorflow.org/tensorboard): for experiment tracking;
- [`clearml`](https://clear.ml/docs/latest/docs/): for experiment tracking;
- [`matplotlib`](https://matplotlib.org/stable/contents.html): for plotting;
- [`seaborn`](https://seaborn.pydata.org/): for plotting;
- [`opacus`](https://opacus.ai/): for differential privacy;
- [`diskcache`](http://www.grantjenks.com/docs/diskcache/): for caching;
- [`cerberus`](https://docs.cerberus.org/en/stable/): for data validation;
- [`hydra-core`](https://hydra.cc/docs/): for configuration management.


## 🐳 Docker

You can use also use `fluke` directly inside a Docker container — no installation needed on your local machine.
To do this, you can use the provided Dockerfile. First, clone the repository:

```bash
git clone https://github.com/makgyver/fluke.git
cd fluke
```

Then, build the Docker image:

```bash
docker build -t fluke_container .
```

Then, you can run an interactive session with

```bash
docker run --rm fluke_container fluke [ARGS]
```

where `ARGS` are the arguments you want to pass to the `fluke` command as described in the [fluke CLI](./examples/run.md) section.
