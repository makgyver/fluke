(parallel_run)=

# Parallel training  [Experimental]

Since v0.7.8 (and v0.7.9), it is possible to parallelize the training on multiple GPUs.

## Parallel training for a single client (v0.7.8+)

In `fluke` v0.7.8 and later, to enable parallel training on a single client, you should simply set
more than one GPU in the `device` configuration,  e.g.:

```yaml

# This is the algorithm configuration file

...
exp:
    device: [cuda:0, cuda:1]
...

```

This will automatically enable parallel training on the specified GPUs.

This functionality relies on the [torch.nn.DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) module, which is a PyTorch feature that allows
you to parallelize the training of a single model across multiple GPUs.

## Parallel training for multiple clients - one per GPU (v0.7.9+)

In `fluke` v0.7.9 and later, you can also run multiple clients in parallel, each on a separate GPU.
In this case, additionally to the `device` configuration as shown above, your algorithm object
must inherit from the [ParallelAlgorithm](#fluke.distr.ParallelAlgorithm) class, and similarly, your client
and server objects must inherit from [ParallelClient](#fluke.distr.client.ParallelClient) and
[ParallelServer](#fluke.distr.server.ParallelServer), respectively.

[ParallelAlgorithm](#fluke.distr.ParallelAlgorithm)  is a ready-to-use base class that provides the necessary
functionality for parallel FedAVG training.

To use it, you simply need to set the class as algorithm name in the configuration file:

```yaml

# This is the algorithm configuration file

client:
    ...
server:
    ...
name: fluke.distr.ParallelAlgorithm

```

Clearly, if you define your own algorithm class, the name should be the full path to your class, e.g. `my_package.my_module.MyParallelAlgorithm`.

This functionality relies on the `torch.multiprocessing` module, which is a PyTorch feature that allows
you to run multiple processes in parallel, each on a separate GPU.