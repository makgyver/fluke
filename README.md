# fl-bench
Python module to benchmark federated learning algorithms.

## Running the code

### Install the required python modules:
```bash
pip install -r requirements.txt
```

### Run a federated algorithm
To run an algorithm in FL-Bench you need to create two configuration files:
- `EXP_CONFIG`: the experiment configuration file
- `ALG_CONFIG`: the algorithm configuration file

Some examples of these files can be found in the `configs` folder.

The `EXP_CONFIG` is a json file contains the following fields:
- `protocol`: must contains a dictionary with the overall federated protocol settings, namely:
    - `n_clients`: the number of clients in the federated learning system
    - `n_rounds`: the number of communication rounds
    - `eligible_perc`: the percentage of clients that are elegible for training in each round

- `data`: must contains a dictionary with all settings regarding the data loading process, namely:
    - `dataset`: the dataset used for training the model. This is a dictionary with a single 
      mandatory filed, i.e., `name` which is the name of the dataset. To date the following datasets 
      are supported: `mnist`, `mnistm`, `emnist`, `femnist`, `cifar10`, `cifar100`, `svhn`, 
      `tiny_imagenet`, `shakespeare` and `femnist`. All other fields are treated as parameters for 
      the specific dataset. Please see `fl_bench.data.datasets` for more details.
    - `client_split`: the percentage of the local datasets used as test set
    - `standardize`: boolean value that indicates whether the features have to be standardized or not
    - `distribution`: the data distribution used for the experiment. It is a dictionary with a single
      mandatory field, i.e., `name` which is the name of the distribution. To date the following
      distributions are supported:
        - `iid`: iid
        - `qnt`: quantity skewed
        - `classqnt`: classwise quantity skewed
        - `lblqnt`: label quantity skewed
        - `dir`: label dirichlet skewed
        - `path`: label pathological skewed
        - `covshift`: covariate shift
      All other fields are treated as parameters for the specific distribution. Please see 
      `fl_bench.data` for more details.
    - `sampling_perc`: percentage of the dataset considered for training. 
      If `1.0`, the whole dataset is used for training.
    
- `exp`: must contains the other settings for the experiment:
    - `seed`: the seed used for reproducibility purposes
    - `device`: the device used for training the model. If `auto`, the device is automatically selected
    - `average`: the averaging method using in the evaluation (e.g., "macro", "micro")
        
- `logger`: must contains a dictionary with the settings for the logging:
      - `name`: the logger used for logging the results. It must be one of the following:
          - `local`: the results are logged locally
          - `wandb`: the results are logged on wandb
      - `eval_every`: the number of rounds after which the model(s) is/are evaluated
    In the case of `wandb` as logger, the following fields must be specified:
      - `project`: the name of the project on wandb
      - `entity`: the name of the entity on wandb
      - `tags`: the tags used for logging on wandb


The `ALG_CONFIG` is a json file containing the following fields:
- `name`: the name of the algorithm
- `hyperparameters`: contains the dictinaries for the hyperparameters for clients and server:
    - `model`: the model to train. It must be a valid PyTorch model defined in the `net.py` file

    - `server`: must contains a dictionary with the server hyperparameters (e.g., `{"weighted": true}`)

    - `client`: must contains a dictionary with the client hyperparameters, for example:
        - `batch_size`: the batch size used client-side for training the model
        - `loss`: the loss function used for training the model. It must be a valid PyTorch loss function
        - `local_epochs`: the number of epochs used client-side for training the model
        - `optimizer`: the parameters used for the optimizer. 
          It must be a dictionary with the following fields:
            - `lr`: the learning rate
          All other parameters are optional.
        - `scheduler`: (optional) the parameters used for the learning rate scheduler. 
          It must be a dictionary with the following fields:
            - `step_size`: the step size used for the learning rate scheduler
            - `gamma`: the gamma used for the learning rate scheduler
        
    All other hyper-parameters added to either `client` or `server` are algorithm-specific. 
    For example, the `fedprox` algorithm requires also the following hyperparameters for `client`:
      - `mu`: the mu used for the FedProx algorithm
      - `lambda`: the lambda used for the FedProx algorithm

To run an algorithm, you need to run the following command:
```bash
python fl_bench/main.py --config=EXP_CONFIG run ALG_CONFIG
```

This is an example of command:
```bash
python fl_bench/main.py --config=configs/exp_settings.json run configs/fedavg.json
```
the command run the `fedavg` algorithm on the `mnist` dataset (see `exp_settings.json`) with the 
all the parameters specified in the `exp_settings.json` file and the `fedavg.json` file.

To date, the following (nn-based) federated algorithms are implemented:
- FedAvg (`fedavg`)
- FedSGD (`fedsgd`)
- FedBN (`fedbn`)
- SCAFFOLD (`scaffold`)
- FedProx (`fedprox`)
- FedOpt (`fedopt`)
- MOON (`moon`)
- FedExP(`fedexp`)
- Ditto (`ditto`)
- APFL (`apfl`)
- FedRep (`fedrep`)
- FedPer (`fedper`)
- SuPerFed (`superfed`)
- LG-FedAvg (`lgfedavg`) [**To be tested**]
- FedNova (`fednova`) [**To be tested**]
- pFedMe (`pfedme`) [**To be tested**]
- FedDyn (`feddyn`) [**To be tested**]

FL-bench also offers the following (non nn-based) federated algorithms:
- Adaboost.F (`adaboostf`)
- Adaboost.F2 (`adaboostf2`)
- Preweak.F (`preweakf`)
- Preweak.F2 (`preweakf2`)
- Distboost.F (`distboostf`)

Inside the `net.py` file, you can find the definition of some neural networks. 

## TODO and Work in progress
- [ ] Check the correctness of pFedMe -- **Work in progress**
- [ ] Check the correctness of FedDyn -- **Work in progress**
- [x] Check the correctness of FedNova -- **Work in progress**
- [x] Check the correctness of LG-FedAvg -- **Work in progress**
- [ ] Add documentation + check typing -- **Work in progress**
- [x] Implement configuration file validator

## Desiderata
- [ ] Add support to validation
- [ ] FedSGD: add support to `batch_size != 0`, i.e., the client can perform a local update on a subset (the only batch!) of the data
- [ ] Add support to tensorboard
- [ ] Set up pypi package
