(config_log)=

# Log configuration

Inside the [experiment configuration file](configuration.md), the `logger` key is used to specify
the logging class to be used during the experiment. `fluke` comes with a few built-in loggers, and
all of them are desgined to keep track of the results of the evaluation of the model(s).

## Local logger

The *local logger*, i.e., the logger that logs to the console, is implemented in the class
[Log](#fluke.utils.log.Log). This logger does not require any additional argument and you can set it
in the experiment configuration file as follows:

```yaml
...
logger:
  name: Log
...
```

## Debug logger

The *debug logger* is implemented in the class [DebugLog](#fluke.utils.log.DebugLog). This logger
is similar to the local logger, but it logs additional information useful for debugging purposes.
This logger does not require any additional argument and you can set it in the experiment
configuration file as follows:

```yaml
...
logger:
  name: DebugLog
...
```


## Weights & Biases logger

Weights & Biases is a popular tool for experiment tracking and visualization. 
`fluke` natively supports Weights & Biases logging through the [WandBLog](#fluke.utils.log.WandBLog)
logger. Being a third-party tool, you need to have an account on the Weights & Biases platform and
configure `wandb` with your credentials. You can find more information on how to do this in the
[Weights & Biases documentation](https://docs.wandb.ai/quickstart), in particular step 1 and 2.

After configuring `wandb`, you can set the `wandb` logger in the experiment configuration file as
follows:

```yaml
...
logger:
  name: WandBLog
  project: my_project
  entity: my_entity
  tags: [fluke, experiment]
...
```

where `project` (mandatory) is the name of the project you want to log to, `entity` (mandatory) is
the name of the entity you want to log to, and `tags` (optional) is a list of tags you want to
associate with the experiment. You can also pass other parameters according to the
[`wandb` API](https://docs.wandb.ai/ref/python/init).


## Tensorboard logger

Tensorboard is a popular tool for experiment tracking and visualization in the TensorFlow ecosystem.
Please, refer to the [Tensorboard documentation](https://www.tensorflow.org/tensorboard) for more
information. `fluke` natively supports Tensorboard logging through the
[TensorboardLog](#fluke.utils.log.TensorboardLog) logger. You can set the `tensorboard` logger in
the experiment configuration file as follows:

```yaml
...
logger:
  name: TensorboardLog
  log_dir: /path/to/log/dir
...
```

where `log_dir` (optional) is the path to the directory where the logs will be saved. 
If `log_dir` is not provided, the logs will be saved in the `runs` directory of the current working
directory. Results can be visualized through the Tensorboard web interface by running the following
command:

```bash
tensorboard --logdir /path/to/log/dir
```

and then opening a browser and navigating to `http://localhost:6006` (by default).

## ClearML logger

[ClearML](https://clear.ml/) is another tool for experiment tracking and visualization. 
`fluke` natively supports ClearML logging through the [ClearMLLog](#fluke.utils.log.ClearMLLog)
logger. Being a third-party tool, you need to have an account on the ClearML platform and
configure `clearml` with your credentials. You can find more information on how to do this in the
[ClearML documentation](https://clear.ml/docs/latest/docs/).

After configuring `clearml`, you can set the `clearml` logger in the experiment configuration file
as follows:

```yaml
...
logger:
  name: ClearMLLog
  project_name: my_project
...
```

where `my_project` (mandatory) is the name of the project you want to log to.
You can also pass other parameters according to the
[`clearml` API](https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk).

```{eval-rst}

.. important::
  When logging to `ClearML`, you are also implicitly logging to `Tensorboard`, so you can visualize
  the results through the Tensorboard web interface as described above.

```


## Custom logger

In `fluke` it is possible to define custom loggers, however *custom loggers cannot be set in the 
configuration file* but can be used using the `fluke` API. 

In `fluke` logging happens via the Observer pattern and the observable objects are the server,
the clients, and the channel. Depending on your needs, you can create a custom logger by extending
one or more of the following classes:

- [ServerObserver](#fluke.utils.ServerObserver): to log server-side events;
- [ClientObserver](#fluke.utils.ClientObserver): to log client-side events;
- [ChannelObserver](#fluke.comm.ChannelObserver): to log channel events.

Our suggestion is to extend the [Log](#fluke.utils.log.Log) class that already provides a set of
methods to log to the console, and then add the custom logic you need.
Each observer class has a set of methods that can be overridden to log what you need.
After creating your custom logger, you can attach it to the federated algorithm using the 
method [set_callbacks](#fluke.algorithms.CentralizedFL.set_callbacks) of the 
[CentralizedFL](#fluke.algorithms.CentralizedFL) or 
[PersonalizedFL](#fluke.algorithms.PersonalizedFL) class.