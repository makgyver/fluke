# Add an algorithm to ``fluke``

## Introduction

In this section, we will show how to add a new algorithm to be used in ``fluke``.

```{eval-rst}

.. attention::

   This page will go into the details of the implementation of the server, the client, and the federated learning algorithm.
   For a gentle introduction, please refer to :ref:`New federated algorithm with fluke <tutorials>` tutorial.

```

When a new federated learning algorithm is defined, its functioning can be described by detailing
the behavior of the server and the clients. This is exactly what you need to do to add a new algorithm
to be used in ``fluke``. You must define:

- The `Server` class that inherits from the [fluke.server.Server](#fluke.server.Server) class;
- The `Client` class that inherits from the [fluke.client.Client](#fluke.client.Client) class.

You are not obliged to redefine both the server and the client classes, it depends on the algorithm you want to implement.
For example, if your algorithm only differs from the standard FedAvg only in the way the clients behave, you only need to redefine the client class.

After having defined the server and the client classes, you must define the class representing the federated learning algorithm itself.
This class must inherit from the [CentralizedFL](#fluke.algorithms.CentralizedFL) class and it is responsible for the initialization of the server and the clients. There is no much more to it, as the actual execution of the algorithm is delegated to the server.
However, you must override the following methods:

- `get_client_class`: this method must return the client class you defined (if any);
- `get_server_class`: this method must return the server class you defined (if any).

As pointed out before, only one of the two methods can be overridden according to the needs of the algorithm you are implementing.

The details of the implementation of the server, the client, and the federated learning algorithm are provided in the following sections:

```{eval-rst}

.. toctree::
   :maxdepth: 2

   def_alg/def_client
   def_alg/def_server
   def_alg/def_centralizedfl
   def_alg/def_caching

```






## Running your algorithm using `fluke`

Using the `fluke` CLI to run your algorithm is as easy as changing a few lines in the configuration file of the algorithm.
Let's assume to have defined our algorithm in a python file named `my_algorithm.py` and that the class representing the algorithm is named `MyAlgorithm`.
The configuration file of the algorithm must be structured as follows (please make sure to run `fluke` from the right directory):

```{eval-rst}

.. tab:: Configuration file

    .. code-block:: yaml
      :linenos:
      :emphasize-lines: 1,2,3,11,12,13,20,21,22

      # THIS IS KEY
      # The name of the algorithm must be the fully qualified name to the algorithm's class
      name: my_algorithm.MyAlgorithm
      # Please refer to the algorithm's implementation to know which are its HPs
      hyperparameters:
         # HPs of the clients
         client:
            batch_size: 10 # only an example
            local_epochs: 5 # only an example
            loss: CrossEntropyLoss # only an example
            # Your client-side extra hyper-parameters (if any)
            hyperparam1: value1
            hyperparam2: value2
            ...
            optimizer:
               ...
            scheduler:
               ...
         server:
            # Your server-side hyper-parameters (if any)
            hyperparam1: value1
            hyperparam2: value2
            ...
      model: ...


.. tab:: my_algorithm.py

   .. code-block:: python
      :linenos:

      from fluke.algorithms import CentralizedFL
      from typing import Collection
      from fluke.client import Client
      from fluke.server import Server
      import numpy as np


      class MyServer(Server):

         # we override the aggregate method to implement our aggregation strategy
         def aggregate(self, eligible: Sequence[Client]) -> None:
            # eligible is a list of clients that participated in the last round
            # here we randomly select only two of them

            selected = np.random.choice(eligible, 2, replace=False)

            # we call the parent class method to aggregate the selected clients
            return super().aggregate(selected)


      class MyClient(Client):

         # we override the fit method to implement our training "strategy"
         def fit(self, override_local_epochs: int = 0) -> float:
            # we can override the number of local epochs and call the parent class method
            new_local_epochs = np.random.randint(1, self.hyper_params.local_epochs + 1)
            return super().fit(new_local_epochs)


      class MyFLAlgorithm(CentralizedFL):

         def get_client_class(self) -> type[Client]:
            return MyClient

         def get_server_class(self) -> type[Server]:
            return MyServer

```


