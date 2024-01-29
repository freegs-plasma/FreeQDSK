"""Some type aliases, mostly for hinting about arrays of
numbers. `numpy.typing.ArrayLike` covers anything that can be
converted to an array of any type, and we mostly want to restrict to
arrays of floats.

"""

from typing import Union, Any, List
import numpy as np

try:
    from numpy.typing import NDArray

    float_ = np.floating[Any]
    FloatArray = NDArray[np.floating[Any]]
except ImportError:
    float_ = float  # type: ignore
    FloatArray = np.ndarray  # type: ignore

ArrayLike = Union[int, float, float_, FloatArray, List[float_]]
