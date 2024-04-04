# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = u"reservoir_computing"
copyright = u"2024, Filippo Maria Bianchi"
author = u"Filippo Maria Bianchi"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
#    "sphinx.ext.mathjax",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_book_theme"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "FilippoMB",  # Username
    "github_repo": "Time-series-classification-and-clustering-with-Reservoir-Computing",  # Repo name
    "github_version": "master",  # Branch
    "conf_py_path": "/docs/",  # Path in the repository to conf.py
    "source_suffix": ".md", # File extension for your documentation source files
    "edit_uri": "", # Custom edit URI
    # "github_url": "https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing",
    # "source_url_prefix": "https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing",
}

# html_show_sourcelink = True
# github_url = "https://github.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing"



# -- Options for myst-nb ------------------------------------------------------
# See more options for configuring myst_nb here: https://myst-nb.readthedocs.io/en/latest/authoring/jupyter-notebooks.html#configuration
myst_enable_extensions = [
    "amsmath",
    "deflist",
    "dollarmath",
#    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

nb_execution_mode = "off" # Do not execute the notebooks included in the doc


# --- AutoAPI options --------------------------------------------------------
autoapi_dirs = ["../reservoir_computing"]

autoapi_options = [
    'members',
    'undoc-members',
    # 'private-members',  # Disable generating doc for private members (starting with _)
    'special-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]