import importlib

from .basemap import __basemaps__
from .colormaps import __cmaps__
from .guidelines import Guidelines
from .map import Map
from .resplotclass import Resplotclass, rpc

__version__ = importlib.metadata.version(__package__)

__all__ = ["Guidelines", "Map", "Resplotclass", "__basemaps__", "__cmaps__", "rpc"]
