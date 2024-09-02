# API overview

In this page you can find the list of modules/submodules defined in `fluke` with their classes and functions.

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

**Submodules**

```{eval-rst}

.. currentmodule:: fluke.data

.. autosummary::
   :nosignatures:
   
   datasets
   support

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
   ClientObserver
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

**Submodules**

```{eval-rst}

.. currentmodule:: fluke.utils

.. autosummary::
   :nosignatures:
   
   log
   model

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

## [`fluke.algorithms`](fluke.algorithms.md)

**Classes**

```{eval-rst}
.. currentmodule:: fluke.algorithms

.. autosummary::
   :nosignatures:
   
   CentralizedFL
   PersonalizedFL

```

**Submodules**

```{eval-rst}

.. currentmodule:: fluke.algorithms

.. autosummary::
   :nosignatures:

   apfl
   ccvr
   ditto
   fedala
   fedamp
   fedavg
   fedavgm
   fedaws
   fedbabu
   fedbn
   feddyn
   fedexp
   fedhp
   fedlc
   fednh
   fednova
   fedopt
   fedper
   fedproto
   fedprox
   fedrep
   fedrod
   fedsgd
   lg_fedavg
   moon
   per_fedavg
   pfedme
   scaffold
   superfed

```