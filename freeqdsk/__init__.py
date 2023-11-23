__all__ = ["geqdsk", "aeqdsk", "peqdsk"]

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    # Python 3.7
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # Package hasn't been installed
    pass

from . import aeqdsk
from . import geqdsk
from . import peqdsk
