from io import StringIO
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from freeqdsk import peqdsk

_data_path = Path(__file__).parent / "data" / "peqdsk"


def test_roundtrip():
    # Read data and write to a string buffer
    with open(_data_path / "test_1.peqdsk") as f:
        data = peqdsk.read(f)
    output = StringIO()
    peqdsk.write(*data, output)

    # Read from the string buffer, ensure all elements are equivalent
    output.seek(0)
    data2 = peqdsk.read(output)
    assert_allclose(data.profiles, data2.profiles)
    assert_allclose(data.species, data2.species)
    for key, value in data.units.items():
        assert data2.units[key] == value


def test_write(tmp_path):
    # Create artificial data
    psinorm = np.linspace(0.0, 1.0, 5)
    arr = np.arange(5, dtype=float)
    profiles = pd.DataFrame(
        data={
            "psinorm": psinorm,
            "x": arr,
            "dx/dpsiN": 5 * np.ones(5),
            "y": 2 * arr,
            "dy/dpsiN": 10 * np.ones(5),
        }
    )
    species = pd.DataFrame(
        data={
            "N": [3.0, 1.0, 1.0],
            "Z": [3.0, 1.0, 1.0],
            "A": [6.0, 2.0, 2.0],
        }
    )
    units = {"x": "ft.lb", "y": "fl oz"}

    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_write.peqdsk"
    with open(out, "w") as f:
        peqdsk.write(profiles, species, units, f)

    # Read line by line, check correctness
    with open(out) as f:
        lines = f.readlines()

    def _read_data(line: str) -> np.ndarray:
        return np.fromiter(line.split(), dtype=float)

    assert lines[0] == "5 psinorm x(ft.lb) dx/dpsiN\n"
    for idx, line in enumerate(lines[1:6]):
        assert_allclose(_read_data(line), [idx * 0.25, idx * 1.0, 5.0])
    assert lines[6] == "5 psinorm y(fl oz) dy/dpsiN\n"
    for idx, line in enumerate(lines[7:12]):
        assert_allclose(_read_data(line), [idx * 0.25, idx * 2.0, 10.0])
    assert lines[12] == "3 N Z A of ION SPECIES\n"
    assert_allclose(_read_data(lines[13]), [3, 3, 6])
    for line in lines[13:15]:
        assert_allclose(_read_data(lines[14]), [1, 1, 2])


def test_read(tmp_path):
    # Create artificial data
    pfile = dedent(
        """\
        3 psinorm a(m) da/dpsiN
         0.000000   1.500000   -1.000000
         0.500000   0.500000   -1.000000
         1.000000   -0.500000   -1.000000
        3 psinorm b(kg) db/dpsiN
         0.000000   -3.000000   2.000000
         0.500000   -1.000000   2.000000
         1.000000   1.000000   2.000000
        3 N Z A of ION SPECIES
         4.000000   4.000000   8.000000
         1.000000   1.000000   2.000000
         1.000000   1.000000   2.000000
        """
    )
    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_read.peqdsk"
    with open(out, "w") as f:
        f.write(pfile)
    # Check it matched expected values
    expected_profiles = pd.DataFrame(
        data={
            "psinorm": [0.0, 0.5, 1.0],
            "a": [1.5, 0.5, -0.5],
            "da/dpsiN": [-1.0, -1.0, -1.0],
            "b": [-3.0, -1.0, 1.0],
            "db/dpsiN": [2.0, 2.0, 2.0],
        }
    )
    expected_species = pd.DataFrame(
        data={
            "N": [4.0, 1.0, 1.0],
            "Z": [4.0, 1.0, 1.0],
            "A": [8.0, 2.0, 2.0],
        }
    )
    expected_units = {"a": "m", "b": "kg"}
    with open(out) as f:
        profiles, species, units = peqdsk.read(f)

    assert_allclose(profiles, expected_profiles)
    assert_allclose(species, expected_species)
    for key, value in units.items():
        assert value == expected_units[key]


def test_read_dimensions():
    # Read real data and check dimensionality
    with open(_data_path / "test_1.peqdsk") as f:
        profiles, species, units = peqdsk.read(f)
    assert len(profiles) == 201
    assert len(profiles.columns) == 43
    assert len(species) == 3
    assert len(species.columns) == 3
    assert len(units) == 21  # no units for psinorm or derivatives


@pytest.mark.parametrize("col", ["N", "Z", "A"])
def test_write_bad_species(col, tmp_path):
    """Should fail if species is missing columns"""
    profiles = pd.DataFrame(
        data={
            "psinorm": [0.0, 1.0],
            "x": [0.0, 1.0],
            "dx/dpsiN": [1.0, 1.0],
        }
    )
    units = {"x": "s"}
    species = pd.DataFrame(
        data={
            "N": [6.0, 1.0, 1.0],
            "Z": [6.0, 1.0, 1.0],
            "A": [12.0, 2.0, 2.0],
        }
    )
    species.rename(columns={col: "X"}, inplace=True)
    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_write_bad_species.peqdsk"
    with open(out, "w") as f, pytest.raises(ValueError):
        peqdsk.write(profiles, species, units, f)


def test_write_bad_profiles(tmp_path):
    """Should fail if profiles is missing psinorm"""
    profiles = pd.DataFrame(
        data={
            "psinorm": [0.0, 1.0],
            "x": [0.0, 1.0],
            "dx/dpsiN": [1.0, 1.0],
        }
    )
    units = {"x": "s"}
    species = pd.DataFrame(
        data={
            "N": [6.0, 1.0, 1.0],
            "Z": [6.0, 1.0, 1.0],
            "A": [12.0, 2.0, 2.0],
        }
    )
    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_write_bad_profiles.peqdsk"
    with open(out, "w") as f, pytest.raises(ValueError):
        peqdsk.write(profiles[["x", "dx/dpsiN"]], species, units, f)
