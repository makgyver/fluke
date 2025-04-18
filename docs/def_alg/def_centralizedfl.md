# ``CentralizedFL`` class

Your new algorithm must inherit from the [CentralizedFL](#fluke.algorithms.CentralizedFL) class.
`CentralizedFL` is the class representing a generic centralized federated learning algorithm and it is responsible
for the initialization of the server and the clients, and that's it. 
This is why you should extend it after having implemented the server and the client classes.

The main methods of the `CentralizedFL` class are:

- `__init__`: the constructor of the class; this method simply calls the `init_server` and `init_clients` methods. It also prepares the model to be used by the server. Generally, this method should not be overridden;
- `run`: the method that runs the algorithm. It actually delegates the execution to the server and thus
  in most of the cases it is not necessary to override it;
- `init_server`: the method that initializes the server given the configuration. It simply instantiates the server class;
- `init_clients`: the method that initializes the clients given the configuration. It instantiates all the clients. This method requires to be overridden if 
  the initialization of the clients requires non-standard operations. Currently, all algorithms in `fluke` use the implementation provided by the `CentralizedFL` class;
- `get_client_class`: the method that returns the client class. This method must be overridden if you defined a custom client class;
- `get_server_class`: the method that returns the server class. This method must be overridden if you defined a custom server class;
- `get_optimizer_class`: the method that returns the optimizer class. This method must be overridden if you defined a custom (client-side) optimizer class;
- `can_override_optimizer`: method that returns a boolean value that specifies whether the optimizer can be overridden. If set to `True`, the optimizer can be overridden using the configuration. If set to `False`, the optimizer cannot be overridden. The default value is `True`.

The following is the code for the [PerFedAVG](../algo/Per-FedAvg.md) class:

```{eval-rst}

.. code-block:: python
   :linenos:

    class PerFedAVG(CentralizedFL):

        def get_client_class(self) -> type[Client]:
            return PerFedAVGClient

        def can_override_optimizer(self) -> bool:
            return False

        def get_optimizer_class(self) -> type[Optimizer]:
            return PerFedAVGOptimizer

```

## Personalized Federated Learning algorithm

If you want to implement a personalized federated learning algorithm, you should extend the [PersonalizedFL](#fluke.algorithms.PersonalizedFL) class instead of the `CentralizedFL` class. The only difference between the two classes is that the `PersonalizedFL` class assume clients to be instances of the [PFLClient](#fluke.client.PFLClient) class, which is a subclass of the `Client` class designed for personalized federated learning. The main peculiarity of the `PFLClient` class is that it has a `personalized_model` attribute that must be initialized in the `__init__` method (take a look a [this](def_client.md) for more details). For this reason, in the `PersonalizedFL` class, the `init_clients` method is overridden to call the constructor of the `PFLClient` class instead of the `Client` class.