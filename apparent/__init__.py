from importlib.metadata import version as _version

__version__ = _version("apparently")

from apparent.apparent import Apparent

__all__ = ["Apparent"]
