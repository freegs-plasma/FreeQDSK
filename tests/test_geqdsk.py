"""
SPDX-FileCopyrightText: © 2016 Ben Dudson, University of York.

SPDX-License-Identifier: MIT
"""

from io import StringIO
from difflib import unified_diff
from pathlib import Path

import pytest
import numpy as np
from numpy.random import rand
from numpy.testing import assert_allclose

from freeqdsk import geqdsk

# Test that data can be successfully written and read back


@pytest.fixture(
    params=[
        {"nx": 15, "ny": 15},
        {"nx": 12, "ny": 15},
        {"nx": 15, "ny": 16},
        {"nx": 17, "ny": 61},
    ]
)
def data_dict(request):
    nx, ny = request.param["nx"], request.param["ny"]
    return {
        "nx": nx,
        "ny": ny,
        "rdim": 2 + rand(),
        "zdim": 1 + rand(),
        "rcentr": 1.5 + 0.1 * rand(),
        "bcentr": 2 + rand(),
        "rleft": rand(),
        "zmid": 0.1 * rand(),
        "rmagx": 1 + rand(),
        "zmagx": 0.1 + 0.05 * (1 - rand()),
        "simagx": -rand(),
        "sibdry": rand(),
        "cpasma": 1e6 * (1 + rand()),
        "fpol": rand(nx),
        "pres": rand(nx),
        "qpsi": rand(nx),
        "psi": rand(nx, ny),
    }


def test_roundtrip(data_dict):
    """
    Test that data can be written then read back
    """
    output = StringIO()

    # Write to string
    geqdsk.write(data_dict, output)

    # Move to the beginning of the buffer
    output.seek(0)

    # Read from string
    data2 = geqdsk.read(output)

    # Check that data and data2 are the same
    for key in data_dict:
        assert_allclose(data2[key], data_dict[key])


# Test with real G-EQDSK files

_data_path = Path(__file__).parent / "test_data" / "geqdsk"

_test_1_data = {
    "nx": 101,
    "ny": 101,
    "zdim": 3.65,
    "cpasma": 5.83933250e5,
    "sibdry": 5.74827987e-2,
    "nbdry": 256,
    "nlim": 256,
}

_test_2_data = {
    "nx": 69,
    "ny": 175,
    "bcentr": 2.4,
    "zmagx": 0.0,
    "cpasma": 2.1e7,
    "sibdry": 2.2030412,
    "nbdry": 501,
    "nlim": 500,
}


@pytest.mark.parametrize(
    "path,expected",
    [
        (_data_path / "test_1.geqdsk", _test_1_data),
        (_data_path / "test_2.geqdsk", _test_2_data),
    ],
)
def test_read(path, expected):
    """
    Read pre-prepared test G-EQDSK files, check that the data dict matches expectations
    and no errors are thrown.
    """
    with open(path) as f:
        data = geqdsk.read(f)
    for key in expected:
        assert np.isclose(data[key], expected[key])

    # Check arrays are the correct shape
    assert data["fpol"].shape == (data["nx"],)
    assert data["pres"].shape == (data["nx"],)
    assert data["ffprime"].shape == (data["nx"],)
    assert data["pprime"].shape == (data["nx"],)
    assert data["qpsi"].shape == (data["nx"],)
    assert data["psi"].shape == (data["nx"], data["ny"])
    assert data["rbdry"].shape == (data["nbdry"],)
    assert data["zbdry"].shape == (data["nbdry"],)
    assert data["rlim"].shape == (data["nlim"],)
    assert data["zlim"].shape == (data["nlim"],)


@pytest.mark.parametrize(
    "path", [_data_path / "test_1.geqdsk", _data_path / "test_2.geqdsk"]
)
def test_write(path, tmp_path):
    """
    Read pre-prepared test G-EQDSK files, write back out to a temporary directory.
    Then check that the files are identical (except for the header).

    TODO: try to ensure the headers will match perfectly too!
    """
    # Create tmp dir
    d = tmp_path / "geqdsk"
    d.mkdir(exist_ok=True)
    out = d / path.name

    # Read test file
    with open(path) as f:
        data = geqdsk.read(f)

    # Write out to tmp dir
    with open(out, "w") as f:
        geqdsk.write(data, f)

    # Read both files back in as plain lists of str
    with open(path) as original, open(out) as new:
        # Ignore header line
        original_lines = original.readlines()[1:]
        new_lines = new.readlines()[1:]

    # Ensure we managed to read/write something
    assert original_lines
    assert new_lines
    # Check that the diff is zero
    diff = [*unified_diff(original_lines, new_lines)]
    assert not diff


# Test with broken G-EQDSK files
# TODO write: missing required data
# TODO write: required data wrong type
# TODO write: arrays wrong size
# TODO read: catch warn if duplicated entries don't match
# TODO read: raise if arrays are wrong size
# TODO read: raise if nbdry, nlim line missing
