<!-- ```{eval-rst}

:layout: landing

``` -->

# **``fluke``**


```{eval-rst}

.. button-link:: https://pypi.org/project/fluke-fl/
   :color: primary
   :outline:

   v |version| |release|
```

<h3>Federated Learning Utility frameworK for Experimentation and research.</h3>
<h4>Made by researchers for researchers!</h4>

**``fluke``** is a benchmarking tool for Federated Learning (FL). It is designed to be
flexible, easy to use, and easy to extend, and can be used to benchmark a wide variety of
federated learning algorithms. **``fluke``** is meant for researchers and practitioners who
want to quickly develop their own federated algorithm and test its performance against
state-of-the-art algorithms on a variety of datasets and conditions. In **``fluke``** **the federation
is simulated**.

## Philosophy

**``fluke``** is designed to minimize the development overhead of adding new algorithms and performing
experiments. It is built on the following principles:

- **Easy to use**: **``fluke``** is designed to be easy to use. It is easy to install, to run, and to
configure. Running a federated learning experiment is as simple as running a single command.
- **Easy to extend**: **``fluke``** is designed to be easy to extend minimazing the overhead of adding
new algorithms. Adding a new method is as simple as adding the definition of the client and the server.
- **Up-to-date**: **``fluke``** implements state-of-the-art federated learning algorithms and datasets
and is regularly updated to include the latest affirmed techniques.
- **Simulated**: in **``fluke``** the federation is simulated. This means that the communication between
the clients and the server is happens in a simulated channel and the data is not actually sent over
the network. The simulated environment frees the user from aspects not related to the algorithm itself.

## Explore **``fluke``**

::::{grid} 3
:::{grid-item-card} <i class="fa-solid fa-rocket"></i> Getting Started
:link: ./getting_started.html
Is it your first time using **``fluke``**? Start here.
:::
:::{grid-item-card} <i class="fa-solid fa-code"></i> API Reference
:link: ./api_reference.html
Explore the **``fluke``** API.
:::
:::{grid-item-card} <i class="fa-solid fa-laptop-code"></i> Tutorials
:link: ./tutorials.html
Check out the tutorials to get acquainted with **``fluke``**.
:::
::::


```{eval-rst}

.. toctree::
    :caption: Getting started
    :hidden:

    install
    first_run
    parallel_run
    examples/run
    configuration
    add_algorithm
    tutorials

.. toctree::
    :caption: API DOCUMENTATION
    :hidden:

    api_reference
    fluke
    fluke.algorithms
    fluke.client
    fluke.comm
    fluke.config
    fluke.data
    fluke.distr
    fluke.evaluation
    fluke.nets
    fluke.server
    fluke.utils

.. toctree::
    :caption: Contribute
    :hidden:

    helpus
    CODE_OF_CONDUCT

```