(fluke.data)=

# ``fluke.data``

```{eval-rst}

.. automodule:: fluke.data
   :no-members:
   :exclude-members: support

```

TODO

## Submodules

```{eval-rst}

.. currentmodule:: fluke.data

.. autosummary::
   :nosignatures:

   datasets

```

<h2>

Classes

</h2>

```{eval-rst}

.. currentmodule:: fluke.data

.. autosummary:: 
   :nosignatures:

   DataContainer
   FastDataLoader
   DataSplitter
   DummyDataSplitter
```

<h3>

{bdg-primary}`class` ``fluke.data.DataContainer``

</h3>

```{eval-rst}

.. autoclass:: fluke.data.DataContainer
   :members:

```

<h3>

{bdg-primary}`class` ``fluke.data.FastDataLoader``

</h3>

```{eval-rst}

.. currentmodule:: fluke.data.FastDataLoader

.. autosummary::
   :nosignatures:

   __getitem__
   set_sample_size

```

```{eval-rst}

.. autoclass:: fluke.data.FastDataLoader
   :members: __getitem__, set_sample_size

```


<h3>

{bdg-primary}`class` ``fluke.data.DataSplitter``

</h3>

```{eval-rst}

.. currentmodule:: fluke.data.DataSplitter

.. autosummary::
   :nosignatures:

   num_classes
   assign
   iid
   quantity_skew
   classwise_quantity_skew
   label_quantity_skew
   label_dirichlet_skew
   label_pathological_skew
   covariate_shift

```

```{eval-rst}

.. autoclass:: fluke.data.DataSplitter
   :members:

```

<h3>

{bdg-primary}`class` ``fluke.data.DummyDataSplitter``

</h3>

```{eval-rst}

.. autoclass:: fluke.data.DummyDataSplitter
   :members: assign
   :show-inheritance:

```


```{eval-rst}

.. toctree::
   :maxdepth: 2
   :hidden:

   fluke.data.datasets

```