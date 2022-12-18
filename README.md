# fl-bench
Python module to benchmark federated learning algorithms

## Running the code

- Install the required python modules:
```bash
pip install -r requirements.txt
```

- Run a federated algorithm
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
- FEMNIST (`femnist`)
- SVHN (`svhn`)

Inside the `net.py` file, you can find the definition of some neural networks. 
> :warning: As of now, the network and some of the hyperparameters are hardcoded in the `main.py` file.

## TODO ASAP
- [ ] Check for seed consistency
- [ ] Check the correctness of SCAFFOLD
- [ ] Implement FedNova - https://arxiv.org/abs/2007.07481
- [ ] Implement FedDyn - https://openreview.net/pdf?id=B7v4QMR6Z9w
- [ ] Implement FedADMM - https://arxiv.org/pdf/2204.03529.pdf
- [ ] FedSGD: add support to `batch_size != 0`, i.e., the client can perform a local update on a subset (the only batch!) of the data
- [ ] Test logging on wandb
- [ ] Add support to validation
- [ ] Add client-side evaluations - useful to evaluate FedBN
- [ ] Add documentation + check typing

## Desiderata
- [ ] Add more algorithms
- [ ] Add more datasets
- [ ] Add support to tensorboard
- [ ] Set up pypi package