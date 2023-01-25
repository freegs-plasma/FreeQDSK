__all__ = ["geqdsk", "aeqdsk"]

from importlib.metadata import version, PackageNotFoundError

__version__ = version(__name__)

from . import aeqdsk
from . import geqdsk
