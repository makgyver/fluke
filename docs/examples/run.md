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

The `fluke` command allows to run three different types of experiments:

1. **Centralized learning**: (`centralized`) the server trains a global model on the entire dataset.
2. **Federated learning**: (`federation`) run a centralized federated learning experiment according to the configurations.
3. **Clients only**: (`clients-only`) train each client individually on its own data.

The structure of the command is the following:

```bash
fluke --config <EXP_CONFIG_FILE> <experiment_type> <ALG_CONFIG_FILE> [OPTIONS]
```

where: 
- `<EXP_CONFIG_FILE>` is the path to the experiment configuration file;
- `<ALG_CONFIG_FILE>` is the path to the algorithm configuration file. 


### Centralized learning

To run a centralized learning experiment, use the following command:

```bash
fluke --config <EXP_CONFIG_FILE> centralized <ALG_CONFIG_FILE> [--epochs=<EPOCHS>]
```

where `<EPOCHS>` is the number of epochs to train the model (default is 0). If not specified (=0), the number of epochs are calculated according to the algorithm configuration. Specifically, `EPOCHS = n_rounds * eligible_perc`.
Since the learning is centralized, some of the parameters in the configuration file are ignored (i.e., the parameters related to the data distribution). The set up of for the learning (e.g., batch size, optimizer, etc.) is taken from the client hyper-parameters in the algorithm configuration file.

### Federated learning

To run a the federated learning algorithm, use the following command:

```bash
fluke --config <EXP_CONFIG_FILE> federation <ALG_CONFIG_FILE>
```

### Clients only

To run a the learning only on the clients, use the following command:

```bash
fluke --config <EXP_CONFIG_FILE> clients-only <ALG_CONFIG_FILE> [--epochs=<EPOCHS>]
```

where `<EPOCHS>` is the number of epochs to train the model (default is 0). If not specified (=0), the number of epochs are calculated according to the algorithm configuration. Specifically, `EPOCHS = n_rounds * eligible_perc * local_epochs`, but in any case no less than 100.
The hyper-parameters related to the server are ignored in this case.

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