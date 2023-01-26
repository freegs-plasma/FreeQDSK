__all__ = ["geqdsk", "aeqdsk"]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # Package hasn't been installed
    pass

from . import aeqdsk
from . import geqdsk
