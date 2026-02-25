import importlib

from .colormaps import __cmaps__
from .guidelines import Guidelines
from .map import Map
from .resplotclass import Resplotclass, rpc

__version__ = importlib.metadata.version(__package__)

__all__ = ["Guidelines", "Map", "Resplotclass", "rpc", "__cmaps__"]
