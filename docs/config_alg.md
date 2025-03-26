(config_alg)=

# Algorithm configuration

The `ALG_CONFIG_FILE` is a yaml file containing the hyper-parameters of the federated algorithm.
The set of hyper-parameters are divided into two main categories: the hyper-parameters of the clients and the hyper-parameters of the server. Additionally, there is the `model` parameter that specifies the neural network to federate.

```{eval-rst}

    .. code-block:: yaml

        hyperparameters:
            client:
                # HPs of the clients
            
            server:
                # HPs of the server
            
        model: # The model (neural network) to federate

```


## Client hyper-parameters

The client hyper-parameters are divided into three main categories: the hyper-parameters of the client overall training process, the hyper-parameters of the optimizer, and the hyper-parameters of the scheduler.

### Training process HPs

This set of hyper-parameters are related to the training process of the client excluding the optimizer and the scheduler. Here you can specify the batch size, the number of local epochs, the loss function and other algorithm-specific hyper-parameters.

For example:

```{eval-rst}

    .. code-block:: yaml

        client:
            # Batch size
            batch_size: 10
            # Number of local epochs
            local_epochs: 5
            # The loss function (loss name from torch.nn)
            loss: CrossEntropyLoss
            # Algorithm-specific hyper-parameters
            param: value
            ...

```

### The optimizer HPs

The optimizer hyper-parameters are related to the optimizer used to train the client's model. Here you can specify the learning rate, the momentum, the weight decay, etc.

For example:

```{eval-rst}

    .. code-block:: yaml

        client:
            ...
            optimizer:
                name: SGD
                lr: 0.8
                momentum: 0.9
                weight_decay: 0.0001
                ...

```

The `name` parameter is the name of the optimizer from `torch.optim`. If omitted, the default optimizer is `SGD`.

### The scheduler HPs

The scheduler hyper-parameters are related to the scheduler used to adjust the learning rate during the training process. Here you can specify the gamma, the step size, the mode, etc.

For example:

```{eval-rst}

    .. code-block:: yaml

        client:
            ...
            scheduler:
                name: StepLR
                gamma: 0.995
                step_size: 10
                ...

```

The `name` parameter is the name of the scheduler from `torch.optim.lr_scheduler`. If omitted, the default scheduler is `StepLR`.

## Server hyper-parameters

Although less common, the server hyper-parameters are related to the server-side of the federated algorithm. Here you can specify whether to weight the client's contribution to the global model.

For example:

```{eval-rst}

    .. code-block:: yaml

        server:
            weighted: true

```


## Federated model

The `model` parameter specifies the neural network to federate. This can be the name of a neural network included in `fluke.nets` or the fully qualified name of an user defined class.

If the model class accepts arguments, you can specify them in the configuration file using the key `net_args`.

For example:

```{eval-rst}

    .. code-block:: yaml

        model: MNIST_2NN
        net_args:
            hidden_size: [100, 50]

```
