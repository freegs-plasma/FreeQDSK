from pathlib import Path

import numpy as np

from freeqdsk import geqdsk


def scramble_geqdsk(filename: Path, output: Path, magnitude: float = 0.1) -> None:
    """
    Used to redact G-EQDSK data, converting actual data into random numbers with the
    same format. Intended for creating test data from actual G-EQDSK files that may
    contain sensitive information.

    Parameters
    ----------
    filename: Path
        The G-EQDSK file to scramble.
    output: Path
        The path to write the new file to.
    magnitude: float, default 0.1
        The factor by which floats are adjusted, following:
        ::
            x = x * (1.0 + magntiude * (rand() - 0.5))
    """
    with open(filename) as f:
        data = geqdsk.read(f)

    data_new = {}
    fixed_keys = ("nx", "ny", "nbdry", "nlim")
    for key, value in data.items():
        if key in fixed_keys:
            data_new[key] = value
        else:
            value = np.asarray(value)
            factors = 1.0 + magnitude * (np.random.random_sample(value.shape) - 0.5)
            data_new[key] = factors * value

    with open(output, "w") as f:
        geqdsk.write(data_new, f)
