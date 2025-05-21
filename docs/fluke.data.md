(fluke.data)=

# ``fluke.data``

```{eval-rst}

.. automodule:: fluke.data
   :no-members:
   :exclude-members: support

```

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
   DummyDataContainer
   FastDataLoader
   DataSplitter
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
   as_dataloader
   batch_size
   set_sample_size

```

```{eval-rst}

.. autoclass:: fluke.data.FastDataLoader
   :members: __getitem__, as_dataloader, batch_size, set_sample_size

```


<h3>

{bdg-primary}`class` ``fluke.data.DataSplitter``

</h3>

```{eval-rst}

.. currentmodule:: fluke.data.DataSplitter

.. autosummary::
   :nosignatures:

   assign
   iid
   label_quantity_skew
   label_dirichlet_skew
   label_pathological_skew
   num_classes
   quantity_skew

```

```{eval-rst}

.. autoclass:: fluke.data.DataSplitter
   :members:

```

<h3>

{bdg-primary}`class` ``fluke.data.DummyDataContainer``

</h3>

```{eval-rst}

.. autoclass:: fluke.data.DummyDataContainer
   :members:
   :show-inheritance:

```


```{eval-rst}

.. toctree::
   :maxdepth: 2
   :hidden:

   fluke.data.datasets

```