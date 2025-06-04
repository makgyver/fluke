# API overview

In this page you can find the list of modules/submodules defined in `fluke` with their classes and functions.

## Modules

`fluke` is organized in the following modules:

- [`fluke`](fluke.md): contains the core classes and utilities;
- [`fluke.algorithms`](fluke.algorithms.md): contains classes for federated learning algorithms;
- [`fluke.client`](fluke.client.md): contains classes for client-side functionalities;
- [`fluke.comm`](fluke.comm.md): contains classes for communication;
- [`fluke.config`](fluke.config.md): contains classes for configuration;
- [`fluke.data`](fluke.data.md): contains classes for data handling;
- [`fluke.data.datasets`](fluke.data.datasets.md): contains classes for datasets loading;
- [`fluke.distr`](fluke.distr.md): contains classes for distributed functionalities;
- [`fluke.evaluation`](fluke.evaluation.md): contains classes for evaluation;
- [`fluke.nets`](fluke.nets.md): contains classes for neural networks;
- [`fluke.server`](fluke.server.md): contains classes for server-side functionalities;
- [`fluke.utils`](fluke.utils.md): contains utility classes and functions;
- [`fluke.utils.log`](fluke.utils.log.md): contains classes for logging;
- [`fluke.utils.model`](fluke.utils.model.md): contains classes for model manipulation.


## [`fluke`](fluke.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke

.. autosummary::
   :nosignatures:

   DDict
   FlukeCache
   FlukeCache.ObjectRef
   FlukeENV
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
   DummyDataContainer
   FastDataLoader
   DataSplitter

```

## [`fluke.data.datasets`](fluke.data.datasets.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.data.datasets

.. autosummary::
   :nosignatures:

   Datasets

```

## [`fluke.distr`](fluke.distr.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.distr

.. autosummary::
   :nosignatures:

   ParallelAlgorithm

```


## [`fluke.distr.client`](fluke.distr.client.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.distr.client

.. autosummary::
   :nosignatures:

   ParallelClient

```


## [`fluke.distr.server`](fluke.distr.server.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.distr.server

.. autosummary::
   :nosignatures:

   ParallelServer

```

## [`fluke.distr.utils`](fluke.distr.utils.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.distr.utils

.. autosummary::
   :nosignatures:

   ModelBuilder

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
   FedAVGCNN
   ResNet18
   ResNet34
   ResNet50
   ResNet18GN
   MoonCNN
   LeNet5
   Shakespeare_LSTM

```

## [`fluke.utils`](fluke.utils.md)

**Classes**

```{eval-rst}
.. currentmodule:: fluke.utils

.. autosummary::
   :nosignatures:

   ClientObserver
   ServerObserver

```

**Functions**

```{eval-rst}
.. currentmodule:: fluke.utils

.. autosummary::
   :nosignatures:

   bytes2human
   cache_obj
   clear_cuda_cache
   get_class_from_str
   get_class_from_qualified_name
   get_full_classname
   get_loss
   get_model
   get_optimizer
   get_scheduler
   flatten_dict
   import_module_from_str
   memory_usage
   plot_distribution
   retrieve_obj

```

## [`fluke.utils.log`](fluke.utils.log.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.utils.log

.. autosummary::
   :nosignatures:

   Log
   TensorboardLog
   WandBLog
   ClearMLLog

```

**Functions**

```{eval-rst}

.. currentmodule:: fluke.utils.log

.. autosummary::
   :nosignatures:

   get_logger
```

## [`fluke.utils.model`](fluke.utils.model.md)

**Classes**

```{eval-rst}

.. currentmodule:: fluke.utils.model

.. autosummary::
   :nosignatures:

   AllLayerOutputModel
   ModOpt
   MMMixin
   LinesLinear
   LinesConv2d
   LinesLSTM
   LinesEmbedding
   LinesBN2d
```

**Functions**

```{eval-rst}

.. currentmodule:: fluke.utils.model

.. autosummary::
   :nosignatures:

   aggregate_models
   batch_norm_to_group_norm
   check_model_fit_mem
   diff_model
   flatten_parameters
   get_activation_size
   get_global_model_dict
   get_local_model_dict
   get_output_shape
   get_trainable_keys
   merge_models
   mix_networks
   set_lambda_model
   safe_load_state_dict
   state_dict_zero_like
```

## [`fluke.evaluation`](fluke.evaluation.md)

**Classes**

```{eval-rst}
.. currentmodule:: fluke.evaluation

.. autosummary::
   :nosignatures:

   Evaluator
   ClassificationEval
   PerformanceTracker

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
   dpfedavg
   fat
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
   fedld
   fednh
   fednova
   fedopt
   fedper
   fedproto
   fedprox
   fedrep
   fedrod
   fedrs
   fedsam
   fedsgd
   gear
   kafe
   lg_fedavg
   moon
   per_fedavg
   pfedme
   scaffold
   superfed

```