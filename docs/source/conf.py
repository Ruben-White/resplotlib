# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# %%
import shutil
import sys
import tomllib
from pathlib import Path

DIR_PATH_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIR_PATH_REPO / "src"))


# %%
def load_project_metadata(file_path_toml):
    with open(file_path_toml, "rb") as f:
        toml_data = tomllib.load(f)

    project_data = toml_data.get("project", {})

    project = project_data.get("name", "Unknown Project")
    authors = project_data.get("authors", [])
    author = authors[0]["name"] if authors else "Unknown Author"
    release = project_data.get("version", "Unknown Version")

    return {
        "project": project,
        "author": author,
        "release": release,
    }


metadata = load_project_metadata(DIR_PATH_REPO / "pyproject.toml")
project = metadata["project"]
author = metadata["author"]
release = metadata["release"]

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "nbsphinx",
]

intersphinx_mapping = {
    "rioxarray": ("https://corteva.github.io/rioxarray/stable/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
    "xugrid": ("https://deltares.github.io/xugrid/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "contextily": ("https://contextily.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "ipyleaflet": ("https://ipyleaflet.readthedocs.io/en/latest/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "odc.geo": ("https://odc-geo.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_logo = "../logo/resplotlib_logo_white.png"
html_theme_options = {
    "logo_only": True,
}

# -- nbsphinx configuration -----------------------------------------------
nbsphinx_execute = "always"
nbsphinx_timeout = 300


# -- Custom pre-build tasks --------------------------------------------------
def copy_notebooks(dir_path_notebooks, dir_path_notebooks_copy):
    """
    Copy Jupyter notebooks from the source directory to the docs directory.
    """
    if dir_path_notebooks_copy.exists():
        shutil.rmtree(dir_path_notebooks_copy)
    dir_path_notebooks_copy.mkdir(parents=True, exist_ok=True)

    file_path_notebooks = list(dir_path_notebooks.glob("*.ipynb"))

    for file_path_notebook in file_path_notebooks:
        file_path_notebook_copy = dir_path_notebooks_copy / file_path_notebook.name
        shutil.copy(file_path_notebook, file_path_notebook_copy)


def on_builder_inited(app):
    copy_notebooks(
        DIR_PATH_REPO / "notebooks" / "usage_examples",
        DIR_PATH_REPO / "docs" / "source" / "usage_examples",
    )


def setup(app):
    print("Setting up Sphinx app...")
    app.connect("builder-inited", on_builder_inited)
