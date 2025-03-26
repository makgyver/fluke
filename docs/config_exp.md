(config_exp)=

# Experiment configuration

This page describes the configuration of the experiment. The configuration is a YAML file that contains the settings of the experiment, such as the dataset, the device, the evaluation, the logging, and the federated learning protocol.

```{eval-rst}

.. seealso::

    - For a general overview of the configuration files, see :ref:`configuration`.
    - For a detailed description of the dataset configuration, see :ref:`dataset configuration <config_data>`.
    - For a detailed description of the logging configuration, see :ref:`logging configuration <config_log>`.

```

## General experimental configuration

These configurations are related to how the experiment is run. Specifically the user can define:

- `device`: the device where the training and evaluation will be performed. The supported settings are
    - `cpu`: everything is run on the CPU;
    - `cuda` (or `cuda:N` where `N` is the GPU number): the training and evaluation are run on the specified GPU;
    - `mps`: the training and evaluation are run on the GPU using the Multi-Process Service (MPS);
    - `auto`: the device is automatically selected based on the availability of a GPU;
- `seed`: the seed for the random number generator. This is useful to make the experiment reproducible;
- `inmemory`: whether to use caching to save memory. If `true`, the data is stored in memory, otherwise it is stored on disk.

### Caching

**Activate caching**: `inmemory: false`.

In `fluke`, caching means that only the "active" models are stored in memory, while the others are stored on disk. 
If not differently defined in the algorithm, at each point in time `fluke` keeps in memory only the global model and the models (+ additional data) of the client that is currently training. Using caching can save a lot of memory, especially when the number of clients is high and models are large. However, it can significantly slow down the training process, as the models need to be loaded from disk when needed.


## Evaluation configuration

In federated learning, the evaluation can be performed in different ways. `fluke` offers the followins options:

- `pre_fit: true`: Evaluation of the client model before the client local training starts. In most of the cases, this means testing the just received global model on the local test set;
- `post_fit: true`: Evaluation of the client model after the client local training. This is useful to understand how the client model has improved during the local training;
- `server: true`: Evaluation of the global model on a held out test set (that is kept server side);
- `locals: true`: Evaluation of the client local models on a held out test set.

## Saving configuration

This section allows to specify where to save the models and how often to save them. It is an optional section and can be omitted if the user does not want to save the models.
To save the models, the user must specify the following parameters:

- `path`: the path to the folder where to save the models;
- `save_every`: the frequency of saving the models (in rounds);
- `global_only`: whether to save only the global model.

Models are saved using the following naming convention: 
- for clients: `r{round}_client_{id}.pth` where `{round}` is the round number and `{id}` is the client id. E.g., `r10_client_1.pth` is the model of client 1 at round 10;
- for the server (i.e., global model): `r{round}_server.pth` where `{round}` is the round number. E.g., `r10_server.pth` is the model of the server at round 10.

Besides the models, also a `config.yaml` file is saved in the same folder. This file contains the configuration of the experiment and the algorithm.

## FL protocol configuration

The FL protocol can be specified using the following parameters:

- `eligible_perc`: the percentage of eligible clients in each round, e.g., 0.1 means that 10% of the clients are selected in each round;
- `n_clients`: the total number of clients participating in the federation;
- `n_rounds`: the total number of rounds.


```{eval-rst}

.. toctree::
    :maxdepth: 2
    :hidden:

    config_data
    config_log

```