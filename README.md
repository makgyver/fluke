<a href="https://makgyver.github.io/fluke"><img src="https://img.shields.io/github/actions/workflow/status/makgyver/fluke/doc-publish.yml?style=for-the-badge&label=DOCUMENTATION"/></a>

# FLUKE: Federated Learning Utility frameworK for Experimentation 

## Install with `pip`

Install the package with ``pip`` is as simple as running the following command:

```bash
pip install fluke
```

## Using fluke w/o installation

First, clone the repository:
```bash
git clone https://github.com/makgyver/fluke.git
```

Then, install the required python modules:
```bash
cd fluke
pip install -r requirements.txt
```

Finally, you can run the code as described in the next section.

### Run a federated algorithm

To run an algorithm in ``fluke`` you need to create two configuration files:
- `EXP_CONFIG`: the experiment configuration file
- `ALG_CONFIG`: the algorithm configuration file

Some examples of these files can be found in the `configs` folder of the repository.

The `EXP_CONFIG` is a yaml file containing the configurations for the experiment
(independentrly from the algorithm). It contains the following fields:

```yaml
# Dataset configuration
data:
  # Dataset loading config
  dataset:
    # Dataset's name 
    # Currently supported: mnist, svhn, mnistm, femnist, emnist, cifar10, cifar100, tiny_imagenet,
    #                      shakespeare, fashion_mnist, cinic10
    name: mnist
    # Potential parameters for loading the dataset correctly (see the documentation of fluke.data.datasets)
    params: null
  # IID/non-IID data distribution
  distribution:
    # Currently supported: 
    # - iid: Independent and Identically Distributed data.
    # - qnt: Quantity skewed data.
    # - classqnt: Class-wise quantity skewed data.
    # - lblqnt: Label quantity skewed data.
    # - dir: Label skewed data according to the Dirichlet distribution.
    # - path : Pathological skewed data (each client has data from a small subset of the classes).
    # - covshift: Covariate shift skewed data.
    name: iid
    # Potential parameters of the disribution, e.g., `beta` for `dir` (see the documentation of fluke.data.DataSplitter)
    params: null
  # Sampling percentage when loading the dataset
  sampling_perc: 1
  # Client-side test set split percentage
  client_split: 0
  # Whether to keep the test set as provided by the dataset
  keep_test: true
  # Whether the server has a test set
  server_test: true
  # The size of the server split (only used when keep_test=false)
  server_split: 0.0
  # Whether to use client-side a iid test set regardless of the training data distribution
  uniform_test: false
# Generic settings for the experiment
exp:
  # The device to load the tensors (auto, cpu, cuda, mps, cuda:0, etc.)
  device: cpu
  # The seed (reproducibility)
  seed: 42
# Logger configuration
logger:
  # `Log` is the standard output, `WandBLog` log everything on weights and bias
  name: local
  # `wand` parameters
  params: null
# FL protocol configuration
protocol:
  # % of eligible clients in each round
  eligible_perc: 1
  # Total number of clients partcipating in the federation
  n_clients: 100
  # Total number of rounds
  n_rounds: 100
```

The `ALG_CONFIG` is a yaml file containing the hyper-parameters of the federated algorithm. It 
contains the following fields:
```yaml
# Hyperparameters (HPs) of the algorithm
# Name of the algorithm: this must be the full path to the algorithm's class
name: fluke.algorithms.fedavg.FedAVG
# Please refer to the algorithm's file to know which are the HPs 
hyperparameters:
  # HPs of the clients
  client:
    # Batch size
    batch_size: 10
    # Number of local epochs
    local_epochs: 5
    # The loss function
    loss: CrossEntropyLoss
    # HPs of the optimizer (the type of optimizer depends on the algorithm)
    optimizer:
      # if omitted, the default optimizer is SGD
      name: SGD
      lr: 0.8
    # HPs of the scheduler (scheduler name from torch.optim.lr_scheduler)
    scheduler:
      # if omitted, the default scheduler is StepLR
      name: StepLR
      gamma: 0.995
      step_size: 10
  # HPs of the server
  server:
    # whether to weight the client's contribution
    weighted: true
  # The model (neural network) to federate.
  # This can be the name of a neuranet included in fluke.nets or the fullname of a 
  # user defined class
  model: MNIST_2NN
```

If you installed ``fluke`` with pip, you can run the following command to start the experiment:
```bash
fluke --config=EXP_CONFIG federation ALG_CONFIG
```

Otherwise, to run a **federated** algorithm, you need to run the following command:
```bash
python -m fluke.run --config=EXP_CONFIG federate ALG_CONFIG
```

To run a **centralized (baseline) algorithm**, you need to run the following command:
```bash
fluke --config=EXP_CONFIG centralized ALG_CONFIG
```
or
```bash
python -m fluke.run --config=EXP_CONFIG centralized ALG_CONFIG
```

Finally, to run the learning process only on clients, you need to run the following command:
```bash
fluke --config=EXP_CONFIG clients-only ALG_CONFIG
```
or
```bash
python -m fluke.run --config=EXP_CONFIG clients-only ALG_CONFIG
```

To date, the following federated algorithms are implemented:
- APFL
- CCVR
- Ditto
- FedAMP
- FedAvg
- FedAvgM
- FedBABU
- FedBN
- FedDyn
- FedExP
- FedHP
- FedLC
- FedNH
- FedNova
- FedOpt (FedAdam, FedAdagrad, FedYogi)
- FedPer
- FedProto
- FedProx
- FedRep
- FedSGD
- LG-FedAvg
- MOON
- PerFedAVG (PerFedAVG-FO, PerFedAVG-HF)
- pFedMe
- SCAFFOLD
- SuPerFed (SuPerFed-MM, SuPerFed-LM)


Inside the `nets.py` file, you can find the definition of some neural networks. 

## TODO and Work in progress
- [ ] Documentation -- **Work in progress**
- [ ] Unit/Functional/Integration testing -- **Work in progress**

## Desiderata
- [ ] Add support to validation
- [ ] Add support to tensorboard
