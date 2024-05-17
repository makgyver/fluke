(configuration)=

# **``fluke``** configuration


To run an algorithm using the ``fluke`` command you need to create two configuration files:
- `EXP_CONFIG_FILE`: the experiment configuration file (independent from the algorithm);
- `ALG_CONFIG_FILE`: the algorithm configuration file.

## Experiment configuration

The `EXP_CONFIG_FILE` is a yaml file containing the configurations for the experiment.
In the following we provide an example of the `EXP_CONFIG_FILE` with the comments explaining 
each field.

```{eval-rst}

.. tab:: YAML w/ comments

    .. code-block::

        # Dataset configuration
        data:
            # Dataset loading config
            dataset:
                # Dataset's name 
                # Currently supported: mnist, svhn, mnistm, femnist, emnist, cifar10, cifar100,
                #                      tiny_imagenet, shakespeare, fashion_mnist, cinic10
                name: mnist
                # Potential parameters for loading the dataset correctly
                # (see the documentation of fluke.data.datasets)
                # if no parameters are needed, simply do not specify anything
                params: null
            # IID/non-IID data distribution
            distribution:
                # Currently supported: 
                # - iid: Independent and Identically Distributed data.
                # - qnt: Quantity skewed data.
                # - classqnt: Class-wise quantity skewed data.
                # - lblqnt: Label quantity skewed data.
                # - dir: Label skewed data according to the Dirichlet distribution.
                # - path : Pathological skewed data (each client has data from few classes).
                # - covshift: Covariate shift skewed data.
                name: iid
                # Potential parameters of the disribution, e.g., `beta` for `dir`
                # (see the documentation of fluke.data.DataSplitter)
                # if no parameters are needed, simply do not specify anything
                params: null
            # Sampling percentage when loading the dataset.
            # Thought to be used for debugging purposes
            # The sampling is repeated at each round
            sampling_perc: 1
            # Client-side test set split percentage.
            # If set to 0, the clients do not have a test set
            # and the evaluation is done on the server side
            client_split: 0
            # Whether to keep the test set as provided by the dataset
            keep_test: true
            # Whether the server has a test set
            # Note: if `keep_test` is set to true, than the server will have such a test set
            server_test: true
            # The size of the server split 
            # (only used when `keep_test=false` and `server_test=true`)
            server_split: 0.0
            # Whether to use client-side a iid test set distribution regardless
            # of the training data distribution
            uniform_test: false
        # Generic settings for the experiment
        exp:
            # The device to load the tensors (auto, cpu, cuda, mps, cuda:0, etc.)
            device: cpu
            # The seed (reproducibility)
            seed: 42
            # Logger configuration
        logger:
            # `Log` is the standard output, `WandBLog` logs everything on weights and bias
            name: Log
            # `wandb` parameters. Leave empty if `name` is `Log`
            params: null
            # FL protocol configuration
        protocol:
            # % of eligible clients, i.e., participants, in each round
            eligible_perc: 1
            # Total number of clients partcipating in the federation
            n_clients: 100
            # Total number of rounds
            n_rounds: 100

.. tab:: YAML w/o comments

    .. code-block::

        data:
            dataset:
                name: mnist
            distribution:
                name: iid
            sampling_perc: 1
            client_split: 0
            keep_test: true
            server_test: true
            uniform_test: false
        exp:
            device: cpu
            seed: 42
        logger:
            name: Log
        protocol:
            eligible_perc: 1
            n_clients: 100
            n_rounds: 100
```

## Algorithm configuration

The `ALG_CONFIG_FILE` is a yaml file containing the hyper-parameters of the federated algorithm.
It must be structured as follows:

```{eval-rst}

.. tab:: YAML w/ comments

    .. code-block::
    
        # Hyperparameters (HPs) of the algorithm
        # Name of the algorithm: this must be the full path to the algorithm's class
        name: fluke.algorithms.fedavg.FedAVG
        # Please refer to the algorithm's implementation to know which are its HPs 
        hyperparameters:
            # HPs of the clients
            client:
                # Batch size
                batch_size: 10
                # Number of local epochs
                local_epochs: 5
                # The loss function (loss name from torch.nn)
                loss: CrossEntropyLoss
                # HPs of the optimizer (the type of optimizer depends on the algorithm)
                optimizer:
                    # if omitted, the default optimizer is SGD (optimizer name from torch.optim)
                    name: SGD
                    lr: 0.8
                # HPs of the scheduler (scheduler name from torch.optim.lr_scheduler)
                # this is optional
                scheduler:
                    # if omitted, the default scheduler is StepLR
                    name: StepLR
                    gamma: 0.995
                    step_size: 10
            # HPs of the server
            server:
                # Whether to weight the client's contribution
                # If not hyperparametera are needed set to {}
                weighted: true
        # The model (neural network) to federate.
        # This can be the name of a neural network included in `fluke.nets` or  
        # the fully qualified name of an user defined class
        model: MNIST_2NN

.. tab:: YAML w/o comments

    .. code-block::

        name: fluke.algorithms.fedavg.FedAVG
        hyperparameters:
            client:
                batch_size: 10
                local_epochs: 5
                loss: CrossEntropyLoss
                optimizer:
                    name: SGD
                    lr: 0.8
                scheduler:
                    name: StepLR
                    gamma: 0.995
                    step_size: 10
            server:
                weighted: true
            model: MNIST_2NN

```