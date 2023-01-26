# fl-bench
Python module to benchmark federated learning algorithms

## Running the code

### Install the required python modules:
```bash
pip install -r requirements.txt
```

### Run a federated algorithm
```bash
python main.py run ALGORITHM DATASET [options]
```
where the options are:
```bash
--n-clients                INTEGER  Number of clients [default: 5]
--n-rounds                 INTEGER  Number of rounds [default: 100] 
--n-epochs                 INTEGER  Number of client-side epochs [default: 5]
--batch-size               INTEGER  Client-side batch size [default: 225]
--elegibility-percentage   FLOAT    Elegibility percentage [default: 0.5]
--distribution             INTEGER  Data distribution [default: 1] 
--seed                     INTEGER  Seed [default: 42]
```

The following algorithms are implemented:
- FedAvg (`fedavg`)
- FedSGD (`fedsgd`)
- FedBN (`fedbn`)
- SCAFFOLD (`scaffold`)
- FedProx (`fedprox`)
- FedOpt (`fedopt`)
- FLHalf (`flhalf`)

Currently, the following datasets are implemented:
- MNIST (`mnist`)
- MNISTM (`mnistm`)
- EMNIST (`emnist`)
- CIFAR10 (`cifar10`)
- SVHN (`svhn`)

Inside the `net.py` file, you can find the definition of some neural networks. 

:warning: **As of now, the network and some of the hyperparameters are hardcoded in the `main.py` file.**

### Compare the performance of different algorithms

Compare the performance of different algorithms on the same dataset and distribution:
```bash
python main.py compare --dataset=DATASET --n_clients=N_CLIENTS --distribution=DISTRIBUTION --n-rounds=N_ROUNDS [--local]
```
where `--local` compares the performance of the algorithms on the local test dataset, while the possible distributions are:
- `iid`: iid
- `qnt`: quantity skewed
- `classqnt`: classwise quantity skewed
- `lblqnt`: label quantity skewed
- `dir`: label dirichlet skewed
- `path`: label pathological skewed
- `covshift`: covariate shift


## TODO ASAP
- [x] Check the seed consistency
- [x] Check the correctness of SCAFFOLD --> The implementation seems ok however I can not "perfecly" replicate the results, e.g., https://link.springer.com/article/10.1007/s40747-022-00895-3. It seems that global_step > 1 works better than global_step = 1 although in the paper only global_step = 1 is used.
- [ ] Check the correctness of FedBN
- [ ] Implement FedNova - https://arxiv.org/abs/2007.07481
- [ ] Implement FedDyn - https://openreview.net/pdf?id=B7v4QMR6Z9w
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
- [ ] Configuration via file yaml/json -- **Work in progress**
- [ ] Implement FedADMM - https://arxiv.org/pdf/2204.03529.pdf
- [ ] Add more algorithms -- **Work in progress**
- [ ] Add more datasets -- **Work in progress**
- [ ] Add support to tensorboard
- [ ] Set up pypi package
