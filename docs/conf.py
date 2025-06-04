# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))

project = "fluke"
copyright = "2024, Mirko Polato"
author = "Mirko Polato"
version = "0.7.9"
# release = 'alpha'

# autodoc_mock_imports = ['algorithms'] # Fix the "No module named 'algorithms'" error


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'myst_parser',
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinx_tags",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    # 'nbsphinx',
    "myst_nb",
    # 'sphinx_autodoc_typehints'
]

myst_enable_extensions = ["colon_fence"]
# autodoc_typehints = "description"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_inherit_docstrings = False
tags_create_tags = True
tags_extension = ["rst", "md"]
tags_create_badges = True
nb_number_source_lines = True
nb_remove_code_outputs = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # 'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    # 'matplotlib': ('http://matplotlib.org/stable', None),
    "torch": ("https://pytorch.org/docs/main/", None),
    "torchvision": ("https://pytorch.org/vision/stable/", None),
    "torchmetrics": ("https://torchmetrics.readthedocs.io/en/latest/", None),
    "omegaconf": ("https://omegaconf.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
suppress_warnings = ["config.cache"]

html_context = {
    "source_type": "github",
    "source_user": "makgyver",
    "source_repo": "fluke",
    "source_version": "main",  # Optional
    "source_docs_path": "/docs/",  # Optional
}

ISSUE_URL = (
    "https://github.com/makgyver/fluke/issues/new"
    + "?assignees=&labels=&projects=&template={}.md&title="
)
html_theme_options = {
    # "globaltoc_expand_depth": 1,
    # "toctree_collapse": True
    # "toctree_titles_only": True
    "nav_links": [
        {
            "title": "Tutorials",
            "url": "tutorials",
            "children": [
                {
                    "title": "Run fluke",
                    "url": "examples/run",
                    "summary": "Use fluke out of the box",
                },
                {
                    "title": "Add an FL algorithm",
                    "url": "examples/tutorials/fluke_custom_alg",
                    "summary": "Create your own FL algorithm",
                },
                {
                    "title": "Custom dataset in fluke",
                    "url": "examples/tutorials/fluke_custom_dataset",
                    "summary": "Use a custom dataset in a FL experiment",
                },
                {
                    "title": "Custom model in fluke",
                    "url": "examples/tutorials/fluke_custom_nn",
                    "summary": "Use a custom model in a FL experiment",
                },
                {
                    "title": "Custom evaluation in fluke",
                    "url": "examples/tutorials/fluke_custom_eval",
                    "summary": "Define custom evaluation metrics",
                },
            ],
        },
        {
            "title": "Help us",
            "url": "helpus",
            "children": [
                {
                    "title": "Report a bug",
                    "url": ISSUE_URL.format("bug_report"),
                    "summary": "Help us to improve fluke",
                },
                {
                    "title": "Request a feature",
                    "url": ISSUE_URL.format("feature_request"),
                    "summary": "Tell us what you need",
                },
                {
                    "title": "Contribute",
                    "url": "https://github.com/makgyver/fluke/pulls",
                    "summary": "Join the development team",
                },
            ],
        },
    ]
}
