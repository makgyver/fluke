# **``fluke``** API overview

TODO

## [`fluke`](fluke.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke

.. autosummary::
   :nosignatures:

   DDict
   GlobalSettings
   ObserverSubject
   Singleton

```

## [`fluke.data`](fluke.data.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.data

.. autosummary::
   :nosignatures:
   
   DataContainer
   FastDataLoader
   DataSplitter
   DummyDataSplitter

```


## [`fluke.client`](fluke.client.md)

**Classes**

```{eval-rst}
.. currentmodule:: fluke.client

.. autosummary::
   :nosignatures:
   
   Client
   PFLClient
```

## [`fluke.server`](fluke.server.md)

**Classes**

```{eval-rst}
.. currentmodule:: fluke.server

.. autosummary::
   :nosignatures:
   
   Server
```

## [`fluke.comm`](fluke.comm.md)

**Classes**

```{eval-rst}
.. currentmodule:: fluke.comm

.. autosummary::
   :nosignatures:
   
   Message
   Channel
   ChannelObserver
```

## [`fluke.nets`](fluke.nets.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.nets

.. autosummary::
   :nosignatures:
   
   EncoderHeadNet
   GlobalLocalNet
   HeadGlobalEncoderLocalNet
   EncoderGlobalHeadLocalNet
   MNIST_2NN
   MNIST_CNN
   FedBN_CNN
   MNIST_LR
   CifarConv2
   ResNet9
   FEMNIST_CNN
   VGG9
   ResNet18
   ResNet34
   ResNet50
   ResNet18GN
   MoonCNN
```

## [`fluke.utils`](fluke.utils.md)

**Classes**

```{eval-rst}
.. currentmodule:: fluke.utils

.. autosummary::
   :nosignatures:
   
   OptimizerConfigurator
   Configuration
   Log
   WandBLog
   ServerObserver
```

**Functions**

```{eval-rst}
.. currentmodule:: fluke.utils

.. autosummary::

   import_module_from_str
   get_class_from_str
   get_loss
   get_model
   get_scheduler
   clear_cache
   get_full_classname
```

**Enums**

```{eval-rst}
.. currentmodule:: fluke.utils

.. autosummary::
   :nosignatures:
   
   LogEnum
```

## [`fluke.evaluation`](fluke.evaluation.md)

**Classes**

```{eval-rst}
.. currentmodule:: fluke.evaluation

.. autosummary::
   :nosignatures:
   
   Evaluator
   ClassificationEval
```