# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "scvi": ("https://docs.scvi-tools.org/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- Project information -----------------------------------------------------
needs_sphinx = "5.0"
templates_path = ["_templates"]

info = metadata("biolord")
project_name = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
release = info["Version"]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

# autodoc
autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
# myst
myst_heading_anchors = 3  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
pygments_style = "default"

# bibliography
bibtex_bibfiles = ["references.bib"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "_static/img/light_mode_logo.png"
html_static_path = ["_static"]
html_title = project_name
html_theme_options = {
    "repository_url": f"https://github.com/nitzanlab/{project_name}",
    "repository_branch": "main",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "logo": {
        "image_dark": "_static/img/dark_mode_logo.png",
    },
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
    },
}
