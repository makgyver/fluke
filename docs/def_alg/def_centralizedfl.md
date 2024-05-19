# ``CentralizedFL`` class

Your new algorithm must inherit from the [CentralizedFL](../fluke.algorithms.md) class.
`CentralizedFL` is the class representing a generic centralized federated learning algorithm and it is responsible
for the initialization of the server and the clients, and that's it. 
This is why you should extend it after having implemented the server and the client classes.

The main methods of the `CentralizedFL` class are:

- `__init__`: the constructor of the class;
- `run`: the method that runs the algorithm. It actually delegates the execution to the server and thus
  in most of the cases it is not necessary to override it;
- `init_server`: the method that initializes the server given the configuration. It simply instantiates the server class;
- `init_clients`: the method that initializes the clients given the configuration. It instantiates all the clients. This method requires to be overridden if 
  the initialization of the clients requires non-standard operations. Currently, all algorithms in `fluke` use the implementation provided by the `CentralizedFL` class;
- `get_client_class`: the method that returns the client class. This method must be overridden if you defined a custom client class;
- `get_server_class`: the method that returns the server class. This method must be overridden if you defined a custom server class;
- `get_optimizer_class`: the method that returns the optimizer class. This method must be overridden if you defined a custom (client-side) optimizer class.

The following is the code for the `SCAFFOLD` class:

```{eval-rst}

.. code-block:: python
   :linenos:

    class SCAFFOLD(CentralizedFL):
        def get_optimizer_class(self) -> torch.optim.Optimizer:
            return SCAFFOLDOptimizer

        def get_client_class(self) -> Client:
            return SCAFFOLDClient

        def get_server_class(self) -> Server:
            return SCAFFOLDServer

```