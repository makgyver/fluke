# First run

After installing `fluke`, you can run a federated algorithm from the command line. 
To do so, you need two configuration files:

- `EXP_CONFIG_FILE`: the experiment configuration file (independent from the algorithm);
- `ALG_CONFIG_FILE`: the algorithm configuration file;

Then, you can run the following command:

```bash
fluke federation EXP_CONFIG_FILE ALG_CONFIG_FILE
```

```{eval-rst}

.. important::

    The ``fluke`` command provides also the possibility to run the algorithm in a ``centralized``
    fashion as well as ``client-side only`` (no federation).

    .. code-block:: bash

        fluke centralized EXP_CONFIG_FILE ALG_CONFIG_FILE
        fluke clients-only EXP_CONFIG_FILE ALG_CONFIG_FILE
    
    In these settings, the same configuration files as the federated case can be used but some
    of the parameters might be ignored and some others are adjusted to be consistent with the
    selected setting. For example, the number of epochs in the centralized setting is the number of
    total rounds times the participation fraction, while in the clients-only setting it is the
    number of local epochs times the number of rounds times the participation fraction 
    (anyway no less than 100 epochs). Please refer to :ref:`this tutorial <fluke.cli>` for more details.

```

For the purpose of this introductory page, we will use the configuration files provided in the 
[configs folder](https://github.com/makgyver/fluke/tree/main/configs) of the `fluke` repository.

```{eval-rst}

.. tip::
    You can easily get the configuration files provided in the repository using the ``fluke-get`` command.
    For example, to get the configuration files used in this tutorial, you can run the following commands:

    .. code-block:: bash

        fluke-get config exp
        fluke-get config fedavg
    
    These two file will be downloaded in the "./config" folder.
```

In particular, we will use the configuration files [exp.yaml](https://github.com/makgyver/fluke/tree/main/configs/exp.yaml)
and [fedavg.yaml](https://github.com/makgyver/fluke/tree/main/configs/fedavg.yaml) to run the classic [`FedAvg`](#fluke.algorithms.fedavg.FedAVG) algorithm on the [`MNIST`](#fluke.data.datasets.Datasets) dataset.

```bash
fluke federation path_to_configs/exp.yaml path_to_configs/fedavg.yaml
```

where `path_to_configs` is the path to the folder containing the configuration files.

For more details on the configuration files, you can refer to the [Configuration](configuration.md) section.
