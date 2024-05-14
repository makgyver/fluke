# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

project = 'fluke'
copyright = '2024, Mirko Polato'
author = 'Mirko Polato'
release = '0.0.1'

# autodoc_mock_imports = ['algorithms'] # Fix the "No module named 'algorithms'" error


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx_design',
    'sphinx_tags'
    # 'sphinx_autodoc_typehints'
]

myst_enable_extensions = ["colon_fence"]
# autodoc_typehints = "description"
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_inherit_docstrings = False
tags_create_tags = True
tags_extension = ['rst', 'md']

tags_create_badges = True
# tags_badge_colors = {
#     "interface": "primary",
#     "tag2": "secondary",
#     "tag3": "success",
# }

intersphinx_mapping = {
    'python': ('http://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    # 'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    # 'matplotlib': ('http://matplotlib.org/stable', None),
    'torch': ("https://pytorch.org/docs/master/", None)
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_book_theme'
# html_theme = 'sphinx_rtd_theme'
html_theme = 'shibuya'
html_static_path = ['_static']

html_theme_options = {
    # "globaltoc_expand_depth": 1,
    # "toctree_collapse": True
    # "toctree_titles_only": True
}

suppress_warnings = ["config.cache"]

# html_sidebars = {
#     "**": [
#         "sidebars/localtoc.html",
#         "sidebars/repo-stats.html",
#         "sidebars/edit-this-page.html"
#     ]
# }


html_context = {
    "source_type": "github",
    "source_user": "makgyver",
    "source_repo": "fluke",
    "source_version": "deepcopy",  # Optional
    "source_docs_path": "/docs/",  # Optional
}
