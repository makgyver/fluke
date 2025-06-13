(config_eval)=

# Evaluation configuration

In Federated Learning contexts, the evaluation of the model(s) can happen in different ways.
In `fluke`, it is possible to evaluate the model(s) at the client-side, at the server-side, or both.

## Server-side evaluation of the global model
The server-side evaluation is a convenient way to evaluate the global model on a held-out test set that is kept server-side.
In a real FL scenario, usually, the server does not have access to any data, but for convenience, `fluke` allows to specify a server-side test set that can be seen as a held-out test set that is shared among all clients.
To enable the server-side evaluation, you need to set the on the `eval` section of the experiment configuration file:

```yaml
eval:
    global: true # Enable server-side evaluation
```

This evaluation happens at the end of each round (*), after the global model has been updated with the local models of the clients.

(*) Note that the server-side evaluation is performed with a frequency that depends on the `eval_every` parameter in the `eval` section.


## Client-side evaluation
The client-side evaluation can be performed in two ways:
1. **Pre-fit evaluation**: This evaluation is performed before the client starts the local training. It is useful to understand how the client model performs on the local test set before the local training starts.
2. **Post-fit evaluation**: This evaluation is performed after the client finishes the local training. It is useful to understand how the client model has improved during the local training.
To enable the client-side evaluation, you need to set the following parameters in the `eval` section of the experiment configuration file:

```yaml
eval:
    pre_fit: true # Enable pre-fit evaluation
    post_fit: true # Enable post-fit evaluation
```

## Server-side evaluation of client models
Finally, it is possible to evaluate the client models (i.e., the clients models before aggregation) on a held-out test set that is kept server-side.
To enable the server-side evaluation of client models, you need to set the following parameter in the `eval` section of the experiment configuration file:

```yaml
eval:
    locals: true # Enable server-side evaluation of client models
```


## Frequency of evaluation
The frequency of the evaluation can be specified using the `eval_every` parameter in the `eval` section of the experiment configuration file.

```yaml
eval:
    eval_every: 1 # Evaluate every round
```

Whatever the value of `eval_every`, the evaluation is also performed at the end of the first and the last round (even in the case of a keyboard interruption).