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
- `name`: the name of the experiment
- `seed`: the seed used for reproducibility purposes
- `device`: the device used for training the model. If `auto`, the device is automatically selected
- `n_clients`: the number of clients in the federated learning system
- `n_rounds`: the number of communication rounds
- `batch_size`: the batch size used client-side for training the model
- `n_epochs`: the number of epochs used client-side for training the model
- `eligibility_percentage`: the percentage of clients that are elegible for training in each round
- `loss`: the loss function used for training the model. It must be a valid PyTorch loss function
- `distribution`: the data distribution used for training the model. 
  It must be one of the following: 
    - `iid`: iid
    - `qnt`: quantity skewed
    - `classqnt`: classwise quantity skewed
    - `lblqnt`: label quantity skewed
    - `dir`: label dirichlet skewed
    - `path`: label pathological skewed
    - `covshift`: covariate shift
- `model`: the model to train. It must be a valid PyTorch model defined in the `net.py` file
- `dataset`: the dataset used for training the model. It must be one of the following:
    - `mnist`: MNIST
    - `mnistm`: MNISTM
    - `emnist`: EMNIST
    - `cifar10`: CIFAR10
    - `svhn`: SVHN
- `validation`: the percentage of the training dataset used for validation
- `sampling`: percentage of the dataset considered for training. 
  If `1.0`, the whole dataset is used for training.
- `logger`: the logger used for logging the results. It must be one of the following:
    - `local`: the results are logged locally
    - `wandb`: the results are logged on wandb
- `wandb_params`: the parameters used for logging on wandb. Used only if `logger` is set to `wandb`.
  It must be a dictionary with the following fields:
    - `project`: the name of the project on wandb
    - `entity`: the name of the entity on wandb
    - `tags`: the tags used for logging on wandb


The `ALG_CONFIG` is a json file containing the following fields:
- `name`: the name of the algorithm
- `optimizer_parameters`: the parameters used for the optimizer. 
  It must be a dictionary with the following fields:
    - `lr`: the learning rate
    - `scheduler_kwargs`: the parameters used for the learning rate scheduler. 
      It must be a dictionary with the following fields:
        - `step_size`: the step size used for the learning rate scheduler
        - `gamma`: the gamma used for the learning rate scheduler
- `hyperparameters`: the hyperparameters used for the algorithm. Depending on the algorithm, it can be empty
  or it can contain some fields. For example, the `fedavg` algorithm does not require any hyperparameter,
    while the `fedprox` algorithm requires the following fields:
        - `mu`: the mu used for the FedProx algorithm
        - `lambda`: the lambda used for the FedProx algorithm

To run an algorithm, you need to run the following command:
```bash
python fl_bench/main.py --config=EXP_CONFIG run ALG_CONFIG [OPTIONS]
```
where the OPTIONS are:
```bash
--dataset                   [mnist|mnistm|svhn|emnist|cifar10]  Dataset [default: (mnist)]  
--n-clients                 INTEGER  Number of clients [default: 5]
--n-rounds                  INTEGER  Number of rounds [default: 100] 
--n-epochs                  INTEGER  Number of client-side epochs [default: 5]
--batch-size                INTEGER  Client-side batch size [default: 225]
--elegibility-percentage    FLOAT    Elegibility percentage [default: 0.5]
--distribution              INTEGER  Data distribution [default: 1] 
--seed                      INTEGER  Seed [default: 42]
--logger                    [local|wandb]   Log method [default: (local)]
--device                    [cpu|cuda|auto] Device to use [default: (auto)] 
```

the optional arguments override the values in the `EXP_CONFIG` file.

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

To date, the following algorithms are implemented:
- FedAvg (`fedavg`)
- FedSGD (`fedsgd`)
- FedBN (`fedbn`)
- SCAFFOLD (`scaffold`)
- FedProx (`fedprox`)
- FedOpt (`fedopt`)
- FLHalf (`flhalf`)
- MOON (`moon`)

Inside the `net.py` file, you can find the definition of some neural networks. 

## TODO ASAP
- [x] Check the seed consistency
- [x] Check the correctness of SCAFFOLD --> The implementation seems ok however I can not "perfecly" replicate the results, e.g., https://link.springer.com/article/10.1007/s40747-022-00895-3. It seems that global_step > 1 works better than global_step = 1 although in the paper only global_step = 1 is used.
- [ ] Check the correctness of FedBN
- [ ] Implement FedNova - https://arxiv.org/abs/2007.07481 -- **Work in progress**
- [ ] Implement FedDyn - https://openreview.net/pdf?id=B7v4QMR6Z9w -- **Work in progress**
- [ ] Implement Ditto - https://arxiv.org/pdf/2012.04221.pdf
- [x] Test logging on wandb
- [x] Add support to learning rate scheduler
- [ ] Add support to validation
- [x] Add client-side evaluations - useful for evaluating FedBN
- [ ] Add documentation + check typing -- **Work in progress**
- [ ] Add load/save checkpoints -- **Work in progress**
- [x] Implement "macro" averaging for evaluation

## Desiderata
- [ ] FedSGD: add support to `batch_size != 0`, i.e., the client can perform a local update on a subset (the only batch!) of the data
- [x] Configuration via file yaml/json
- [ ] Implement FedADMM - https://arxiv.org/pdf/2204.03529.pdf
- [ ] Add more algorithms -- **Work in progress**
- [ ] Add more datasets -- **Work in progress**
- [ ] Add support to tensorboard
- [ ] Set up pypi package
