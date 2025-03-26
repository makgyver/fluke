(fluke.cli)=

# `fluke` CLI

The package provides a command line interface (CLI) to run federated learning experiments. 
The two main commands are `fluke` and `fluke-get`.

## `fluke`

The `fluke` command is used to run federated learning experiments. It requires two configuration files: 
- `EXP_CONFIG_FILE`: the experiment configuration file (independent from the algorithm);
- `ALG_CONFIG_FILE`: the algorithm configuration file.

```{eval-rst}

.. hint::
    For more information on the configuration files, see :ref:`configuration`.
```

The `fluke` command allows to run four different types of experiments:

1. **Centralized learning**: (`centralized`) the server trains a global model on the entire dataset.
2. **Clients only**: (`clients-only`) train each client individually on its own data.
3. **Federated learning**: (`federation`) run a centralized federated learning experiment according to the configurations.
4. **Battery of FL experiments**: (`sweep`) run a battery of experiments according to the configurations.


The structure of the command is the following:

```bash
fluke <experiment_type> <EXP_CONFIG_FILE> <ALG_CONFIG_FILE> [OPTIONS]
```

where: 
- `<EXP_CONFIG_FILE>` is the path to the experiment configuration file;
- `<ALG_CONFIG_FILE>` is the path to the algorithm configuration file. 


### Federated learning

To run a the federated learning algorithm, use the following command:

```bash
fluke federation <EXP_CONFIG_FILE> <ALG_CONFIG_FILE> [--resume=<PATH>] 
```

where `--resume=<PATH>: <PATH>` must be the path to a directory where the state of a previous experiment is saved. If specified, the experiment will resume from the saved state. If not specified, the experiment starts from scratch.

### Battery of FL experiments

Ususally, when running a federated learning experiment, it is useful to run a battery of experiments to evaluate the performance of the algorithm under different configurations. These configurations can be related to the data distribution, the algorithm hyper-parameters, or simply another random seed.

In `fluke` you can run a battery of experiments using the `sweep` command. This command requires two configuration files: the experiment configuration file and the algorithm configuration file:

```bash
fluke sweep <EXP_CONFIG_FILE> <ALG_CONFIG_FILE> [<ALG_CONFIG_FILES>]
```

These experiments are run sequentially.

To specify the different settings, one just need to modify the configuration files in such a way that the parameters to sweep are lists. For example, if you want to sweep over the number of clients, you can set the `n_clients` parameter in the experiment configuration file as a list of integers.

```yaml
n_clients: [10, 20, 30]
```

Another example is to sweep over different data distributions. In this case, you can set the `dataset` parameter in the experiment configuration file as a list of strings.

```yaml
dataset:
    name: mnist
    path: ./data
distribution:
    - name: iid
    - name: dir
      beta: 0.3
```

If instead you want to sweep over different algorithms, one can specifiy a list of algorithm configuration files, e.g.,

```bash
fluke sweep exp.yaml alg1.yaml alg2.yaml alg3.yaml
```


### Centralized learning

To run a centralized learning experiment, use the following command:

```bash
fluke centralized <EXP_CONFIG_FILE> <ALG_CONFIG_FILE> [--epochs=<EPOCHS>]
```

where `<EPOCHS>` is the number of epochs to train the model (default is 0). If not specified (=0), the number of epochs are calculated according to the algorithm configuration. Specifically, `EPOCHS = n_rounds * eligible_perc`.
Since the learning is centralized, some of the parameters in the configuration file are ignored (i.e., the parameters related to the data distribution). The set up of for the learning (e.g., batch size, optimizer, etc.) is taken from the client hyper-parameters in the algorithm configuration file.


### Clients only

Performing the training on the clients only is useful to evaluate the performance of the algorithm on the clients' data without the federation.
To run a the learning only on the clients, use the following command:

```bash
fluke clients-only <EXP_CONFIG_FILE> <ALG_CONFIG_FILE> [--epochs=<EPOCHS>]
```

where `<EPOCHS>` is the number of epochs to train the model (default is 0). If not specified (=0), the number of epochs are calculated according to the algorithm configuration. Specifically, `EPOCHS = n_rounds * eligible_perc * local_epochs`, but in any case no less than 100.
The hyper-parameters related to the server are ignored in this case.

```{eval-rst}

.. important::
    When running in `clients-only` mode, the logging on services like `wandb` happens at the end of training of all clients.
```

## `fluke-get`

The `fluke-get` command is useful to get a configuration file template for the specified algorithm or for the experiment.
It directly downloads the template from the repository and saves it in the `config` directory (if it does not exist, it creates it).

```bash
fluke-get <config file name>
```

where `<config file name>` is the name of the configuration file to download (without the `.yaml` extension).

You can also ask for the list of available algorithms or experiments.

```bash
fluke-get list
```