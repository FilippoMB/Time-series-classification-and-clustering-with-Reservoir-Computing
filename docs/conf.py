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
]
autoapi_dirs = ["../reservoir_computing"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# def skip(app, what, name, obj, skip, options):
#     if name == "_initialize_internal_weights_Circ":
#         return True
#     return None  # Otherwise, let Sphinx handle it.

# def setup(app):
#     app.connect("autodoc-skip-member", skip)

# autodoc_default_options = {
#     'exclude-members': '__weakref__',
#     'private-members': False,  # Exclude private members
#     'special-members': '__init__',
#     'undoc-members': True,
#     'show-inheritance': True,
# }

def skip_private_members(app, what, name, obj, skip, options):
    if name.startswith('_'):
        return True  # Skip private members
    return None  # Otherwise, let Sphinx decide

def setup(app):
    app.connect('autodoc-skip-member', skip_private_members)