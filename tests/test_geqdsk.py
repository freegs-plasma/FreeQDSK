"""
SPDX-FileCopyrightText: © 2016 Ben Dudson, University of York.

SPDX-License-Identifier: MIT
"""

from io import StringIO
from difflib import unified_diff
from pathlib import Path

import fortranformat as ff
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


# Test with preprepared G-EQDSK files


_data_path = Path(__file__).parent / "data" / "geqdsk"

_test_1_data = {
    "nx": 101,
    "ny": 101,
    "zdim": 3.62482049,
    "cpasma": 6.11940815e5,
    "sibdry": 5.82339193e-2,
    "nbdry": 256,
    "nlim": 256,
}

_test_2_data = {
    "nx": 69,
    "ny": 175,
    "bcentr": 2.36591466,
    "zmagx": 0.0,
    "cpasma": 2.06432536e7,
    "sibdry": 2.16552103,
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


def test_write_unrecoverable_missing_data(tmp_path):
    # read in test data
    with open(_data_path / "test_1.geqdsk") as f:
        data = geqdsk.read(f)

    # Delete necessary data
    data.pop("psi")

    # Write out again
    d = tmp_path / "geqdsk"
    d.mkdir(exist_ok=True)
    out = d / "unrecoverable.geqdsk"
    with open(out, "w") as f, pytest.raises(KeyError):
        geqdsk.write(data, f)


def test_write_recoverable_missing_data(tmp_path):
    # read in test data
    with open(_data_path / "test_1.geqdsk") as f:
        data = geqdsk.read(f)

    # Delete superfluous data
    data.pop("nx")

    # Write out again
    d = tmp_path / "geqdsk"
    d.mkdir(exist_ok=True)
    out = d / "recoverable.geqdsk"
    with open(out, "w") as f:
        geqdsk.write(data, f)


def test_write_bad_data(tmp_path):
    # read in test data
    with open(_data_path / "test_1.geqdsk") as f:
        data = geqdsk.read(f)

    # Set data to incompatible type
    data["fpol"] = "hello world!"

    # Write out again
    d = tmp_path / "geqdsk"
    d.mkdir(exist_ok=True)
    out = d / "bad_data.geqdsk"
    with open(out, "w") as f, pytest.raises(ValueError):
        geqdsk.write(data, f)


def test_write_wrong_array_size(tmp_path):
    # read in test data
    with open(_data_path / "test_1.geqdsk") as f:
        data = geqdsk.read(f)

    # Reshape data
    data["fpol"] = np.array([1.0, 2.0, 3.0])

    # Write out again
    d = tmp_path / "geqdsk"
    d.mkdir(exist_ok=True)
    out = d / "wrong_size.geqdsk"
    with open(out, "w") as f, pytest.raises(ValueError):
        geqdsk.write(data, f)


_duplicate_entries = {
    "rmagx": {"line": 3, "idx": 3},
    "zmagx": {"line": 4, "idx": 0},
    "simagx": {"line": 3, "idx": 1},
    "sibdry": {"line": 2, "idx": 3},
}


@pytest.mark.parametrize("duplicate", _duplicate_entries)
def test_read_duplicate_entry_differences(duplicate, tmp_path):
    d = tmp_path / "geqdsk"
    d.mkdir(exist_ok=True)
    out = d / "duplicate_differences.geqdsk"
    # read in test data, copy to new file, modifying one of the duplicates
    dup_line = _duplicate_entries[duplicate]["line"]
    dup_idx = _duplicate_entries[duplicate]["idx"]
    with open(_data_path / "test_1.geqdsk") as f1, open(out, "w") as f2:
        for line_idx, line in enumerate(f1):
            if line_idx == dup_line:
                record = ff.FortranRecordReader("(5e16.9)").read(line)
                record[dup_idx] = 3.142
                f2.write(ff.FortranRecordWriter("(5e16.9)").write(record) + "\n")
            else:
                f2.write(line)

    # Expect warning that duplicated entries don't match
    with open(out) as f, pytest.warns(UserWarning, match="duplicate"):
        geqdsk.read(f)


@pytest.mark.filterwarnings("ignore")
def test_read_missing_values(tmp_path):
    # The values should be duplicated
    d = tmp_path / "geqdsk"
    d.mkdir(exist_ok=True)
    out = d / "missing_values.geqdsk"
    # read in test data, copy to new file, deleting a value
    with open(_data_path / "test_1.geqdsk") as f1, open(out, "w") as f2:
        for line_idx, line in enumerate(f1):
            if line_idx == 3:
                record = ff.FortranRecordReader("(5e16.9)").read(line)
                f2.write(ff.FortranRecordWriter("(4e16.9)").write(record[:-1]) + "\n")
            else:
                f2.write(line)

    # Expect this to fail messily, by reading the wrong lines into the wrong arrays,
    # before suddenly running out of file to read
    with open(out) as f, pytest.raises(Exception):
        geqdsk.read(f)


def test_read_missing_bdry_lim(tmp_path):
    # The values should be duplicated
    d = tmp_path / "geqdsk"
    d.mkdir(exist_ok=True)
    out = d / "missing_values.geqdsk"
    # read in test data, copy to new file, missing the bdry/lim line and subsequent
    # bits of the file
    with open(_data_path / "test_1.geqdsk") as f1, open(out, "w") as f2:
        for line_idx, line in enumerate(f1):
            if len(line) == 11:  # (2i5) plus '\n'
                break
            else:
                f2.write(line)

    with open(out) as f, pytest.raises(EOFError):
        geqdsk.read(f)
