# ``Server`` class

This class is the core of the federated learning simulation in `fluke`. When you start to extend it, 
make sure to have a clear understanding of what are the data exchanged between the server and the clients and
how the learning process is orchestrated. This is crucial to avoid introducing bugs and to keep the code clean. 

## Overview

The `Server` class is the one responsible for coordinating the federated learning process. 
The learning process starts when the `fit` method is called on the `Server` object. During the
`fit` method, the server will iterate over the number of rounds specified in the argument `n_rounds`.
Each important server's operation trigger a callback to the observers that have been registered to the server.
Finally, at the end of the `fit` method, the server will finalize the federated learning process.

## Server initialization

The `Server` constructor is responsible for initializing the server. Usually, there is not much more to it than setting the server's attributes.
However, there is an important notion that you should be aware of: all the server's hyperparameters should be set in the `hyper_params` attribute that is a [DDict](#fluke.DDict). This best practice ensure that the hyperparameters are easily accessible and stored in a single place.

## Sequence of operations of a single round

The standard behaviour of the `Server` class (as provided in the class [Server](#fluke.server.Server))
follows the sequence of operations of a standard Federate Averaging algorithm. The main methods
of the `Server` class involved in a single round are:

- `get_eligibile_clients`: this method is called at the beginning of each round to select the clients
  that will participate in the round. The selection is based on the `eligible_perc` argument of the
  `fit` method. 
- `broadcast_model`: this method is called at the beginning of each round to send the global model to the clients
  that will participate in the round.
- `aggregate`: this method is called at the end of each round to aggregate the models of the clients
  that participated in the round.
- `evaluate`: this method is called at the end of each round to evaluate the global model on the server-side test set (if any).

The following figure shows the sequence of operations of the `Server` class during the `fit` method.

```{eval-rst}

.. admonition:: Disclaimer
    
    For brevity, many details have been omitted or simplified. However, the figure below shows the key methods and calls involved in a round.
    For a complete description of the ``Server`` class, please refer to the :ref:`Server's API documentation <fluke.server>`.

```


```{figure} ../_static/imgs/server_fit_sequence.png
Sequence of operations of the `Server` class during the `fit` method. 
This image has been created with [TikZ](https://texample.net/tikz/) [[source]](https://github.com/makgyver/fluke/blob/main/docs/_static/tex/server_sequence.tex).
```

The sequence diagram above shows the sequence of operations of the `Server` class during a single round.
It highlights the dependencies between the methods of the `Server` class and the `Client` class. Moreover,
it shows that the communication between the server and the clients is done through the `Channel`.
The only direct call between the server and the client is the `fit` method of the `Client` class that is
called to trigger the beginning of the training process on the client side.

## Finalization

At the end of the `fit` method, the `Server` class will finalize the federated learning process by
calling the `finalize` method. Ideally, this method should be used to perform any final operation,
for example, to get the final evaluation of the global (and/or local) model(s), or to save the model(s).
It can also be used to trigger fine-tuning operations client-side as it happens in personalized federated learning.
In its standard implementation, the `finalize` method will call the `evaluate` method to get the final evaluation
of the global model on the server-side test set (if any) and it also performs the evaluation client-side
after the global model has been broadcasted for the last time.

## Observer pattern

As mentioned above, the `Server` class triggers callbacks to the observers that have been registered to the server.
The default notifications are:

- `_notify_start_round`: triggered at the beginning of each round. It calls `ServerObserver.start_round` on each observer;
- `_notify_selected_clients`: triggered after the clients have been selected for the round. It calls `ServerObserver.selected_clients` on each observer;
- `_notify_end_round`: triggered at the end a round. It calls `ServerObserver.end_round` on each observer;
- `_notify_evaluation`: it should be triggered after an evaluation has been performed. It calls `ServerObserver.evaluation` on each observer;
- `_notify_finalize`: triggered at the end of the `finalize` method. It calls `ServerObserver.finished` on each observer.

:::{hint}
    
Refer to the API documentation of the [ServerObserver](fluke.utils.ServerObserver) inerface and the [ObserverSubject](fluke.ObserverSubject) intarface for more details.

:::


## Creating your ``Server`` class

Creating a custom ``Server`` class is straightforward. You need to create a class that inherits from the ``Server`` class
and override the methods that you want to customize. As long as the federated protocol you are implementing follows the
standard Federated Averaging protocol, you can reuse the default implementation of the ``fit`` method and override only the
methods that are specific to your federated protocol.

Let's see an example of a custom ``Server`` class that overrides the ``aggregate`` method while keeping the default implementation
of the other methods.

```{eval-rst}

.. hint::

    Here we show a single example but you can check all the following algorithm implementations to see
    other examples of custom ``Server.aggregate``:

    - :ref:`APFL <fluke.algorithms.apfl>`;
    - :ref:`FedAMP <fluke.algorithms.fedamp>`;
    - :ref:`FedExP <fluke.algorithms.fedexp>`;
    - :ref:`FedAvgM <fluke.algorithms.fedavgm>`;
    - :ref:`FedNova <fluke.algorithms.fednova>`;
    - :ref:`FedOpt <fluke.algorithms.fedopt>`;
    - :ref:`PFedMe <fluke.algorithms.pfedme>`;
    - :ref:`Scaffold <fluke.algorithms.scaffold>` here also the ``broadcast_model`` method is overridden;
    - :ref:`FedNH <fluke.algorithms.fednh>` here also the ``evaluate`` method is overridden;

```

The example follows the implementation of the ``FedExP`` algorithm. We also report the standard implementation of the ``aggregate`` method for comparison.


```{eval-rst}

.. tab:: FedExP Server

    .. code-block:: python
        :linenos:
        :emphasize-lines: 3,5,6,9,18,19,20

        @torch.no_grad()
        def aggregate(self, eligible: Iterable[Client]) -> None:
            W = flatten_parameters(self.model)
            clients_model = self.get_client_models(eligible, state_dict=False)
            Wi = [flatten_parameters(client_model) for client_model in clients_model]
            eta = self._compute_eta(W, Wi)

            clients_sd = [client.model.state_dict() for client in eligible]
            avg_model_sd = deepcopy(self.model.state_dict())
            for key in self.model.state_dict().keys():
                if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                    continue

                if key.endswith("num_batches_tracked"):
                    mean_nbt = torch.mean(torch.Tensor([c[key] for c in clients_sd])).long()
                    avg_model_sd[key] = max(avg_model_sd[key], mean_nbt)
                    continue

                avg_model_sd[key] = avg_model_sd[key] - eta * torch.mean(
                    torch.stack([avg_model_sd[key] - client_sd[key] for client_sd in clients_sd]),
                    dim=0)
            self.model.load_state_dict(avg_model_sd)
        
        def _compute_eta(self, clients_diff: Iterable[dict], eps: float = 1e-4) -> float:
            ...

.. tab:: FedAVG Server

    .. code-block:: python
        :linenos:

        @torch.no_grad()
        def aggregate(self, eligible: Iterable[Client]) -> None:
            avg_model_sd = OrderedDict()
            clients_sd = self.get_client_models(eligible)
            weights = self._get_client_weights(eligible)
            for key in self.model.state_dict().keys():
                if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                    avg_model_sd[key] = self.model.state_dict()[key].clone()
                    continue
                
                if key.endswith("num_batches_tracked"):
                    mean_nbt = torch.mean(torch.Tensor([c[key] for c in clients_sd])).long()
                    avg_model_sd[key] = max(avg_model_sd[key], mean_nbt)
                    continue

                for i, client_sd in enumerate(clients_sd):
                    if key not in avg_model_sd:
                        avg_model_sd[key] = weights[i] * client_sd[key]
                    else:
                        avg_model_sd[key] += weights[i] * client_sd[key]
            self.model.load_state_dict(avg_model_sd)

```

Let's start by summarizing the implementation of the FedAVG's ``aggregate`` method. The goal of this method
is to aggregate the models of the clients that participated in the round to update the global model.
The aggregation is done by computing the weighted average of the models of the clients. Thus, the method
first collects the models of the clients that participated in the round (``self.get_client_models(eligible)``)
and then computes the weighted average (the for loop) using the weights of the clients (``self._get_client_weights(eligible)``).
Finally, the global model is updated with the weighted average model (``self.model.load_state_dict(avg_model_sd)``).

The custom implementation of the ``aggregate`` method for the ``FedExP`` algorithm follows a slightly different approach.
The main difference lies in the update rule of the global model that is based on the model differences rather than the models themselves.
For this reason, the method first computes the differences between the models of the clients and the global model.
Then, it computes the global learning rate `eta` and the average model difference (``eta, mu_diff = self._compute_eta(clients_diff)``) that is needed to update the global model. The rest of the method remains the same as the standard implementation.
Please, refer to the [original paper of the ``FedExP``](https://arxiv.org/pdf/2301.09604) algorithm for more details on the update rule.

```{eval-rst}

.. tip::

    In general, when you extend the ``Server`` class, you should start overriding the methods from the implementation provided in the ``Server`` class and
    then modify only those aspects that do not suit your federated protocol trying to preserve as much as possible the default implementation.
```

Similar considerations can be made for the other cases when the there is no need to override the ``fit``.

```{eval-rst}

.. attention::

    When overriding methods that require to notify the observers, make sure to call the corresponding
    notification method of the ``ObserverSubject`` interface. For example, if you override the ``finalize`` method
    you should call the ``_notify_finalize`` method at the end of the method. For example, see the implementation
    of :ref:`FedBABU <fluke.algorithms.fedbabu>`.
```


### The `fit` method

Sometimes you might also need to override the ``fit`` method of the ``Server`` class. This is the case when the federated protocol
you are implementing requires a different sequence of operations than the standard Federated Averaging protocol.
This is quite uncommon but it can happen. Currently, in `fluke`, the only algorithms that overrides the ``fit`` method are
[FedHP](../algo/FedHP) and [FedDyn](../algo/FedDyn). In both these cases, the protocol differs from
the standard Federated Averaging protocol only in the starting phase of the learning and hence the ``fit`` method is overridden to
add such additional operations and then the `super().fit()` is called to trigger the standard behaviour.

When overriding the ``fit`` method, you should follow the following best practices:

- **Progress bars**: track the progress of the learning process using progress bars. In `fluke`, this this is done using the `rich` library.
  In `rich`, progress bars and status indicators use a *live* display that is an instance of the `Live` class. You can reuse the `Live` instance
  of `fluke` from the `GlobalSettings` using the `get_live_renderer` method. In this live display, you can show the progress of the client-side and
  server-side learning already available in the `GlobalSettings` using `get_progress_bar("clients")` and `get_progress_bar("server")`. Then to update the
  progress bars and to get more information on how to use the `rich` library, please refer to the [official documentation](https://rich.readthedocs.io/en/latest/).
  
  The following is an excert of the `fit` method, showing how to initialize the progress bars:

  ```{eval-rst}
  .. code-block:: python
      :linenos:

      with GlobalSettings().get_live_renderer():
          progress_fl = GlobalSettings().get_progress_bar("FL")
          progress_client = GlobalSettings().get_progress_bar("clients")
          client_x_round = int(self.n_clients * eligible_perc)
          task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
          task_local = progress_client.add_task("[green]Local Training", total=client_x_round)
          ...
  ```

- **Communication**: in `fluke`, generally, clients and server should not call each other methods directly.
  There are very few exceptions to this rule, for example, when the server needs to trigger the training on the client side or when
  the server asks to perform the evaluation. In all other cases, the communication between the server and the clients should be done
  through the `Channel` class (see the [Channel](#fluke.comm.Channel) API reference). The `Channel` instance is available in the `Server` class
  (`_channel` private instance or `channel` property) and it must be used to send/receive messages.
  Messages must be encapsulated in a [Message](#fluke.comm.Message) object.
  Using a channel allows `fluke`, through the logger (see [Log](#fluke.utils.log.Log)) to keep track of the exchanged messages and so it will 
  automatically compute the communication cost. The following is the implementation of the `broadcast_model` method that uses the
  `Channel` to send the global model to the clients:
    
  ```{eval-rst}
  
  .. code-block:: python
      :linenos:
  
      def broadcast_model(self, eligible: Iterable[Client]) -> None:
          self.channel.broadcast(Message(self.model, "model", self), eligible)
  ```

- **Minimal changes principle**: this principle universally applies to software development but it is particularly important when overriding the `fit` method
  because it represents the point where the whole simulation is orchestrated. Start by copying the standard implementation of the `fit` method and then
  modify only the parts that are specific to your federated protocol. This will help you to keep the code clean and to avoid introducing nasty bugs.
