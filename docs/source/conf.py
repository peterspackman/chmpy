# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_path)

project = 'chmpy'
copyright = '2022, Peter Spackman'
author = 'Peter Spackman'
release = '1.0b1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ['_templates']
exclude_patterns = []
master_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
