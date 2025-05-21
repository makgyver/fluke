(fluke.comm)=

# ``fluke.comm``

```{eval-rst}

.. automodule:: fluke.comm
   :no-members:

```


```{eval-rst}

.. autosummary:: 
   :nosignatures:

   Message
   Channel
   ChannelObserver

```

<h2>

{bdg-primary}`class` ``fluke.comm.Message``

</h2>

```{eval-rst}

.. currentmodule:: fluke.comm.Message

.. autosummary:: 
   :nosignatures:

   cache
   clone
   id
   msg_type
   payload
   ram
   sender
   size

```

```{eval-rst}
.. autoclass:: fluke.comm.Message
   :member-order: bysource
   :members: cache, clone, id, msg_type, payload, ram, sender, size

```


<h2>

{bdg-primary}`class` ``fluke.comm.Channel``

</h2>

```{eval-rst}

.. currentmodule:: fluke.comm.Channel

.. autosummary:: 
   :nosignatures:

   __getitem__
   broadcast
   buffer
   clear
   receive
   send
   
```

```{eval-rst}

.. autoclass:: fluke.comm.Channel
   :show-inheritance:
   :members:  __getitem__, broadcast, buffer, clear, receive, send

```

<h2>

{bdg-secondary}`interface` ``fluke.comm.ChannelObserver``

</h2>

```{eval-rst}

.. autoclass:: fluke.comm.ChannelObserver
   :members:

```