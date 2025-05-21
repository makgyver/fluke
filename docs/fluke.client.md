(fluke.client)=

# ``fluke.client``

```{eval-rst}

.. automodule:: fluke.client
   :no-members:

```

<h3>

Classes included in ``fluke.client``

</h3>

```{eval-rst}

.. currentmodule:: fluke.client

.. autosummary::
   :nosignatures:

   Client
   PFLClient
   
```

## Classes

<h3>

{bdg-primary}`class` ``fluke.client.Client``

</h3>


```{eval-rst}

.. currentmodule:: fluke.client.Client

.. autosummary:: 
   :nosignatures:

   channel
   evaluate
   finalize
   fit
   index
   load
   local_model
   local_update
   model
   optimizer
   n_examples
   receive_model
   save
   scheduler
   send_model
   set_channel
   state_dict

   _load_from_cache
   _save_to_cache

```

```{eval-rst}

.. autoclass:: fluke.client.Client
   :member-order: bysource
   :show-inheritance:
   :members: channel, evaluate, finalize, fit, index, load, local_model, local_update, model, optimizer, n_examples, receive_model, save, scheduler, send_model, set_channel, state_dict, _load_from_cache, _save_to_cache

```

<h3>

{bdg-primary}`class` ``fluke.client.PFLClient``

</h3>


```{eval-rst}

.. currentmodule:: fluke.client.PFLClient

.. autosummary:: 
   :nosignatures:

   evaluate
   local_model
   personalized_model
   pers_scheduler
   pers_optimizer

.. currentmodule:: fluke.client

.. autoclass:: PFLClient
   :show-inheritance:
   :members:

```