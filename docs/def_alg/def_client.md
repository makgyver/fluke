# ``Client`` class

We suggest to start from the [Client](../fluke.client.md) class when you want to implement a new federated learning algorithm. 
Often, the `Client` class is the most complex class to implement and maybe the "only one" (you have to implement the class for the algorithm too but it is usually a metter of overriding a couple of get methods) that you need to implement.

## Overview

In ``fluke``, the `Client` class represents the client-side logic of a federated learning algorithm. This is generally the part where most of the magic happens. 

## Client initialization

The `Client` constructor is responsible for initializing the client. Usually, there is not much more to it than setting the client's attributes. However, there are some important notions that you should be aware of:

- all the client's hyperparameters should be set in the `hyper_params` attribute that is a [DDict](../fluke.md). This best practice ensure that the hyperparameters are easily accessible and stored in a single place;
- the optimizer and the scheduler are not initialized in the constructor becuase the client does not own a model yet. They are initialized in the `fit` method. This should be done using the `optimizer_cfg` (see [OptimizerConfigurator](../fluke.config.md)) attribute that is a callable that returns the optimizer and the scheduler. This is done to allow the optimizer to be initialized with the correct model parameters.

The following excperts show the constructor of the [Client](../fluke.client.md) class an hypothetical new client class.

```{eval-rst}

.. tab:: Client constructor

    .. code-block:: python
        :linenos:

        def __init__(self,
                    index: int,
                    train_set: FastDataLoader,
                    test_set: FastDataLoader,
                    optimizer_cfg: OptimizerConfigurator,
                    loss_fn: torch.nn.Module,
                    local_epochs: int):
            self.hyper_params: DDict = DDict(
                loss_fn=loss_fn,
                local_epochs=local_epochs
            )

            self._index: int = index
            self.train_set: FastDataLoader = train_set
            self.test_set: FastDataLoader = test_set
            self.model: Module = None
            self.optimizer_cfg: OptimizerConfigurator = optimizer_cfg
            self.optimizer: Optimizer = None
            self.scheduler: LRScheduler = None
            self.device: device = FlukeENV().get_device()
            self._channel: Channel = None
            self._last_round: int = 0

.. tab:: New client constructor

    .. code-block:: python
        :linenos:

        def __init__(self,
                    index: int,
                    train_set: FastDataLoader,
                    test_set: FastDataLoader,
                    optimizer_cfg: OptimizerConfigurator,
                    loss_fn: torch.nn.Module,
                    local_epochs: int,
                    my_hp1: float,
                    my_hp2: float):
            super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
            self.hyper_params.update(hp1=my_hp1, hp2=my_hp2)
        
```

## Client-side training 

The main method that characterizes the client's behaviour is the `fit` method which is responsible for training the local model;
This method is wrapped inside the method `local_update` which is also responsible of the communication with the server and to perform the evaluation.

The following figure shows the sequence of operations of the `Client` class during the `local_update` method.


```{eval-rst}

.. admonition:: Disclaimer
    
    For brevity, many details have been omitted or simplified. However, the figure below shows the key methods and calls involved.
    For a complete description of the ``Client`` class, please refer to the :ref:`Client's API documentation <fluke.client>`.

```

```{figure} ../_static/imgs/client_fit_sequence.png
Sequence of operations of the `Client` class during the `fit` method. 
This image has been created with [TikZ](https://texample.net/tikz/) [[source]](https://github.com/makgyver/fluke/blob/main/docs/_static/tex/client_sequence.tex).
```

The `local_update` method is called by the server when it is time to train the local model on the client's data. 

```{eval-rst}

.. attention::
    
    In general, the communication between the server and the client should be done through a :ref:`Channel <fluke.comm>`. **Direct methods calls must be only used to trigger events and not to exchange data.**

```

The client receives the global model from the server, trains the local model on its data, and sends the updated model back to the server:

- `receive_model`: this method simply retrieves the global model sent by the server. It is indeed important to make sure that the server has sent the model before calling this method. Although it is named `receive_model`, the message may also contain additional information that the client may need to process/use (for example, see [SCAFFOLD](../algo/SCAFFOLD.md)).

- `send_model`: this method sends the updated model back to the server.

Usualy, most of the logic of a federated learning algorithm is implemented in the `fit` method, where the training loop happens!

### The training loop

There is not much to say about the training loop itself. `fluke` is design to work with `PyTorch` models even though it can be easily extended to work with other frameworks. The training loop is the same as any other training loop in `PyTorch`. We suggest that you take a look at the [PyTorch documentation](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) for more information on how to train a model in PyTorch.

```{eval-rst}

.. tip::
    
    Make sure to move the model to the correct device before training it. Be careful to move it back to the CPU before sending it to the server.
    Cleaning up the CUDA cache is also a good practice to avoid memory leaks :ref:`fluke.utils.clear_cuda_cache <flake.utils.funct>`.
    This loading-unloading process on GPU may lead to a performance penalty in some cases (e.g., with many small models that could fit in the GPU memory at the same time). We plan to introduce a more efficient way to handle this in the future.

```

The following code snippet shows the ``fit`` method of the ``Client`` class.

```{eval-rst}

.. code-block:: python
    :linenos:

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (override_local_epochs if override_local_epochs
                       else self.hyper_params.local_epochs)
        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        running_loss: float = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.model.to("cpu")
        clear_cuda_cache()
        return running_loss
```

## Finalization

After all the rounds are completed, the server may call the ``finalize`` method on the client (see [server customization](def_server.md)). This method is responsible for any finalization steps that the client may need to perform. For example, performing fine-tuning on the local model, saving the model, or sending any final information to the server. In its default implementation, the ``finalize`` method simply receives the final global model from the server. This will allow the client to have the most up-to-date global model to be used for an eventual evaluation.

## Personalized Federated Learning Client

If you are implementing a new personalized federated learning algorithm, you should inherit from the [PFLClient](../fluke.client.md) class instead of the [Client](../fluke.client.md) class. The personalized version of the client class has an additional attribute (`personalized_model`) representing the personalized model.
Differently from the usual local model, the personalized one is initialized by the client itself and hence the constructor requires an additional argument `model` that is the personalized model. The last difference lies in the evaluation method (`evaluate`) that uses the personalized model instead of the local model. 
As always, you can override all the methods you need to customize the behavior of the client.


## Creating your `Client` class

To create your own `Client` class, you need to inherit from the [Client (or PFLCLient)](../fluke.client.md) class. The suggested steps to create a new `Client` class are:

1. Define the constructor of the class and set the hyperparameters in the `hyper_params` attribute. All the inherited attributes should be set calling the super constructor. Here, you can also set any additional attributes that you may need.

2. Override the `fit` method. This method is responsible for training the local model on the client's data and sending the updated model to the server. This is where most of the logic of your algorithm should be implemented. If you need some support methods, define them privately (i.e., use the prefix '_' to indicate that the method is private).

3. If necessary, override the `finalize` method. 


Likewise the `Server` class, you should follow the following best practices:

- **Communication**: in `fluke`, generally, clients and server should not call each other methods directly.
  There are very few exceptions to this rule, for example, when the server needs to trigger an event client-side and viceversa. In all other cases, the communication between the server and the clients should be done
  through the `Channel` class (see the [Channel](../fluke.comm.md) API reference). The `Channel` instance is available in the `Client` class
  (`_channel` private instance or `channel` property) and it must be used to send/receive messages.
  Messages must be encapsulated in a [Message](../fluke.comm.md) object.
  Using a channel allows `fluke`, through the logger (see [Log](../fluke.utils.md)), to keep track of the exchanged messages and so it will 
  automatically compute the communication cost. The following is the implementation of the `send_model` method that uses the
  `Channel` to send the global model to the clients:
    
  ```{eval-rst}
  
  .. code-block:: python
      :linenos:
  
      def send_model(self) -> None:
          self.channel.send(Message(self.model, "model", self.index), "server")
  ```

- **Minimal changes principle**: this principle universally applies to software development but it is particularly important when overriding the `fit` method. Start by copying the standard implementation of the `fit` method and then modify only the parts that are specific to your federated protocol. This will help you to keep the code clean and to avoid introducing nasty bugs.

The following is an example of the `FedProxClient` class (see [FedProx](../algo/FedProx.md)) where we highlighted in the `fit` method the only lines of code that differ from the `FedAVG` implementation.

```{eval-rst}

.. code-block:: python
    :linenos:
    :emphasize-lines: 22,34

    class FedProxClient(Client):
        def __init__(self,
                    index: int,
                    train_set: FastDataLoader,
                    test_set: FastDataLoader,
                    optimizer_cfg: OptimizerConfigurator,
                    loss_fn: torch.nn.Module,
                    local_epochs: int,
                    mu: float):
            super().__init__(index, train_set, test_set, optimizer_cfg, loss_fn, local_epochs)
            self.hyper_params.update(mu=mu) # Add the mu hyperparameter

        # Support method to compute the proximal term
        def _proximal_loss(self, local_model, global_model):
            proximal_term = 0.0
            for w, w_t in zip(local_model.parameters(), global_model.parameters()):
                proximal_term += torch.norm(w - w_t)**2
            return proximal_term

        def fit(self, override_local_epochs: int = 0) -> float
            epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
            W = deepcopy(self.model)
            self.model.to(self.device)
            self.model.train()
            if self.optimizer is None:
                self.optimizer, self.scheduler = self.optimizer_cfg(self.model)
            for _ in range(epochs):
                loss = None
                for _, (X, y) in enumerate(self.train_set):
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    y_hat = self.model(X)
                    loss = self.hyper_params.loss_fn(y_hat, y) + \
                            (self.hyper_params.mu / 2) * self._proximal_loss(self.model, W)
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

            self.model.to("cpu")
            clear_cuda_cache()
```

## Observer pattern

The `Client` class triggers callbacks to the observers that have been registered to the client.
The default notifications are:

- `notify(event="start_fit", ...)`: triggered at the beginning of the `fit` method. It calls `ClientObserver.start_fit` on each observer;
- `notify(event="end_fit", ...)`: triggered at the end of the `fit` method. It calls `ClientObserver.end_fit` on each observer;
- `notify(event="client_evaluation", ...)`: it should be triggered after an evaluation has been performed. It calls `ClientObserver.evaluation` on each observer;

:::{hint}
    
Refer to the API documentation of the [ClientObserver](fluke.utils.ClientObserver) inerface and the [ObserverSubject](fluke.ObserverSubject) intarface for more details.

:::