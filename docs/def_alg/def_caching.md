(def_caching)=

# Memory management

`fluke` provides the possibilities of minimizing the memory usage of your algorithm by using the caching (on disk) mechanism.
This can be activated by setting the `inmemory` parameter in the experiment configuration file to `false`.
By default, the `inmemory` parameter is set to `true` and so everything is stored in RAM and, if in use, in the GPU memory.

When defining your own algorithm you can improve the memory management by using this built-in functionality of `fluke`.
This aspect of you algorithm's implementation is not mandatory, but it can help you to avoid memory leaks and to optimize the memory usage.

To do so, when you define the client class of your algorithm, you need to make sure to move the model and other memory-consuming objects to the cache when they are not in use.


```{eval-rst}

.. attention::

    The server-side model is not cached, although it is possible to do so. 

```

## Caching the model

Usually, the client-side model should be cached right after the training phase.
This can be done by calling the private method [_save_to_cache](#fluke.client.Client._save_to_cache) of the client class which in turn
calls the [cache_obj](#fluke.utils.cache_obj) function of the `fluke.utils` module.

[_save_to_cache](#fluke.client.Client._save_to_cache) autmotically caches the model, the optimizer, and the scheduler of the client.
This is possible only if the triplet `(model, optimizer, scheduler)` is defined in the client class
as a [fluke.utils.model.ModOpt](#fluke.utils.model.ModOpt) object. If there are additional objects related to the model that
need to be cached, you can add them to the [ModOpt](#fluke.utils.model.ModOpt) object as a dictionary in the `additional`
attribute. See the definition of the [ModOpt](#fluke.utils.model.ModOpt) object in the `fluke.utils.model` module for more details.

In case of additional objects that need to be cached, you can add them to the private iterable `_attr_to_cache` of the client class.

### The importance of the `ModOpt` object

The [ModOpt](#fluke.utils.model.ModOpt) object is a container for the model, the optimizer, and the scheduler of the client.
The function of this object is crucial in the success of the caching mechanism. 


### Mind the channel

When caching the model, you need to make sure that also the messages exchanged using the `channel` are cached properly.

As long as the channel is used properly, the only precaution to take is when you are broadcasting a message, i.e., using `self.channel.broadcast`.
By default, the behaviour of [Channel.broadcast](#fluke.comm.Channel.broadcast) is to broadcast the message and then move it to the RAM, if already in the RAM nothing happens. However, to avoid to copy the payload that is right afterwars stored in the cache (automatically), you should move the message to the disk cache before broadcasting it. This ensure that all sent messages will only copy the reference to the payload and not the payload itself. Then, the reference to the object in the message that you use as 'master' will be moved back to the RAM and then it will be collected by the garbage collector.

So, inside the `Server` class when you are broadcasting a message, you should do something like:

```python
# None ensure that this would work fine even if the caching is not active
msg = Message(payload, self, inmemory=None) 
self.channel.broadcast(msg)
```