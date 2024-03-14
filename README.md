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
    - `dataset`: the dataset used for training the model. It must be one of the following: `mnist`, `mnistm`, `emnist`, `cifar10`, `svhn`, and `femnist`.
    - `validation_split`: the percentage of the local datasets used as test/validation set
    - `standardize`: boolean value that indicates whether the features have to be standardized or not
    - `distribution`: the data distribution used for training the model. 
      It must be one of the following: 
        - `iid`: iid
        - `qnt`: quantity skewed
        - `classqnt`: classwise quantity skewed
        - `lblqnt`: label quantity skewed
        - `dir`: label dirichlet skewed
        - `path`: label pathological skewed
        - `covshift`: covariate shift
    - `sampling_perc`: percentage of the dataset considered for training. 
      If `1.0`, the whole dataset is used for training.
      
    And only for FEMNIST:
    - `num_features`: the number of features (784)
    - `num_classes`: the number of classes (62)
    
- `exp`: must contains the other settings for the experiment:
    - `seed`: the seed used for reproducibility purposes
    - `device`: the device used for training the model. If `auto`, the device is automatically selected
    - `average`: the averaging method using in the evaluation (e.g., "macro", "micro")
    - `checkpoint`: the checkpoint configuration. It must be a dictionary with the following fields:
        - `save`: if `true`, the model and the client optimizer are saved after each round
        - `path`: the path where the checkpoint is saved
        - `load`: if `true`, the checkpoint is loaded from the `path` before starting the training
        
- `log`: must contains a dictionary with the settings for the logging:
      - `logger`: the logger used for logging the results. It must be one of the following:
          - `local`: the results are logged locally
          - `wandb`: the results are logged on wandb
      - `eval_every`: the number of rounds after which the model(s) is/are evaluated
      - `wandb_params`: the parameters used for logging on wandb. Used only if `logger` is set to `wandb`.
        It must be a dictionary with the following fields:
          - `project`: the name of the project on wandb
          - `entity`: the name of the entity on wandb
          - `tags`: the tags used for logging on wandb


The `ALG_CONFIG` is a json file containing the following fields:
- `name`: the name of the algorithm
- `model`: the model to train. It must be a valid PyTorch model defined in the `net.py` file
- `hyperparameters`: contains the dictinaries for the hyperparameters for clients and server:
    - `server`: must contains a dictionary with the server hyperparameters (e.g., `{"weighted": true}`)

    - `client`: must contains a dictionary with the client hyperparameters, for example:
        - `batch_size`: the batch size used client-side for training the model
        - `loss`: the loss function used for training the model. It must be a valid PyTorch loss function
        - `local_epochs`: the number of epochs used client-side for training the model
        - `optimizer_parameters`: the parameters used for the optimizer. 
          It must be a dictionary with the following fields:
            - `lr`: the learning rate
            - `scheduler_kwargs`: the parameters used for the learning rate scheduler. 
              It must be a dictionary with the following fields:
                - `step_size`: the step size used for the learning rate scheduler
                - `gamma`: the gamma used for the learning rate scheduler
        
        For example, the `fedprox` algorithm requires also the following hyperparameters:
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

If you want, for example, to change the number of clients, you can run the following command:
```bash
python fl_bench/main.py --config=configs/exp_settings.json run configs/fedavg.json --n-clients=10
```

or you can change the number of clients in the `exp_settings.json` file and run the above command without
specifying the `--n-clients` option.

To date, the following (nn-based) federated algorithms are implemented:
- FedAvg (`fedavg`)
- FedSGD (`fedsgd`)
- FedBN (`fedbn`)
- SCAFFOLD (`scaffold`)
- FedProx (`fedprox`)
- FedOpt (`fedopt`)
- MOON (`moon`)
- FedExP(`fedexp`)
- FedNova (`fednova`) [**Work in progress**]
- pFedMe (`pfedme`) [**Work in progress**]
- FedDyn (`feddyn`) [**Work in progress**]
- Ditto (`ditto`)
- APFL (`apfl`)
- FedRep (`fedrep`)
- FedPer (`fedper`)

FL-bench also offers the following (non nn-based) federated algorithms:
- Adaboost.F (`adaboostf`)
- Adaboost.F2 (`adaboostf2`)
- Preweak.F (`preweakf`)
- Preweak.F2 (`preweakf2`)
- Distboost.F (`distboostf`)

Inside the `net.py` file, you can find the definition of some neural networks. 

## TODO and Work in progress
- [ ] Check the correctness of pFedMe -- **Work in progress**
- [ ] Implement FedNova - https://arxiv.org/abs/2007.07481 -- **Work in progress**
- [ ] Check the correctness of FedDyn -- **Work in progress**
- [x] Implement LG-FedAvg - https://arxiv.org/abs/2001.01523 (to be tested)
- [x] Implement SuPerFed - https://arxiv.org/abs/2109.07628v3
- [ ] Add documentation + check typing -- **Work in progress**
- [ ] Implement configuration file validator -- **Work in progress**

## Desiderata
- [ ] Add support to validation
- [ ] FedSGD: add support to `batch_size != 0`, i.e., the client can perform a local update on a subset (the only batch!) of the data
- [ ] Add support to tensorboard
- [ ] Set up pypi package
