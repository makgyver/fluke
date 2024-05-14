(fluke.comm)=

# ``fluke.comm``

```{eval-rst}

.. automodule:: fluke.comm
   :no-members:

```

TODO


```{eval-rst}

.. autosummary:: 
   :nosignatures:

   fluke.comm.Message
   fluke.comm.Channel
   fluke.comm.ChannelObserver

```

<h2>

{bdg-primary}`class` ``fluke.comm.Message``

</h2>

```{eval-rst}
.. autoclass:: fluke.comm.Message
   :members: get_size, clone

```


<h2>

{bdg-primary}`class` ``fluke.comm.Channel``

</h2>

```{eval-rst}

.. autoclass:: fluke.comm.Channel
   :show-inheritance:
   :members: buffer, __getitem__, send, receive, broadcast, clear

```

<h2>

{bdg-secondary}`interface` ``fluke.comm.ChannelObserver``

</h2>

```{eval-rst}

.. autoclass:: fluke.comm.ChannelObserver
   :members: message_received

```