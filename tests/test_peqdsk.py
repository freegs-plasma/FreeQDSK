from io import StringIO
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from numpy.testing import assert_allclose

from freeqdsk import peqdsk

_data_path = Path(__file__).parent / "data" / "peqdsk"


def test_roundtrip():
    # Read data and write to a string buffer
    with open(_data_path / "test_1.peqdsk") as f:
        data = peqdsk.read(f)
    output = StringIO()
    peqdsk.write(data, output)

    # Read back from the string buffer
    output.seek(0)
    data2 = peqdsk.read(output)

    # Ensure all elements are equivalent
    for name, profile in data["profiles"].items():
        assert name in data2["profiles"]
        assert profile["units"] == data2["profiles"][name]["units"]
        assert_allclose(profile["psinorm"], data2["profiles"][name]["psinorm"])
        assert_allclose(profile["data"], data2["profiles"][name]["data"])
        assert_allclose(profile["derivative"], data2["profiles"][name]["derivative"])
    for species1, species2 in zip(data["species"], data2["species"]):
        assert_allclose(species1["N"], species2["N"])
        assert_allclose(species1["Z"], species2["Z"])
        assert_allclose(species1["A"], species2["A"])


def test_write(tmp_path):
    # Create artificial data
    psinorm = np.linspace(0.0, 1.0, 5)
    arr = np.arange(5, dtype=float)
    profiles = {
        "x": {
            "psinorm": psinorm,
            "data": arr,
            "derivative": 5 * np.ones(5),
            "units": "ft.lb",
        },
        "y": {
            "psinorm": psinorm,
            "data": 2 * arr,
            "derivative": 10 * np.ones(5),
            "units": "fl oz",
        },
    }
    species = [
        {"N": 3.0, "Z": 3.0, "A": 6.0},
        {"N": 1.0, "Z": 1.0, "A": 2.0},
        {"N": 1.0, "Z": 1.0, "A": 2.0},
    ]

    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_write.peqdsk"
    with open(out, "w") as f:
        peqdsk.write({"profiles": profiles, "species": species}, f)

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
    expected_profiles = {
        "a": {
            "psinorm": np.array([0.0, 0.5, 1.0]),
            "data": np.array([1.5, 0.5, -0.5]),
            "derivative": np.array([-1.0, -1.0, -1.0]),
            "units": "m",
        },
        "b": {
            "psinorm": np.array([0.0, 0.5, 1.0]),
            "data": np.array([-3.0, -1.0, 1.0]),
            "derivative": np.array([2.0, 2.0, 2.0]),
            "units": "kg",
        },
    }
    expected_species = [
        {"N": 4.0, "Z": 4.0, "A": 8.0},
        {"N": 1.0, "Z": 1.0, "A": 2.0},
        {"N": 1.0, "Z": 1.0, "A": 2.0},
    ]
    with open(out) as f:
        data = peqdsk.read(f)
    actual_profiles = data["profiles"]
    actual_species = data["species"]

    for name, expected in expected_profiles.items():
        actual = actual_profiles[name]
        assert_allclose(expected["psinorm"], actual["psinorm"])
        assert_allclose(expected["data"], actual["data"])
        assert_allclose(expected["derivative"], actual["derivative"])
        assert expected["units"] == actual["units"]

    for expected, actual in zip(expected_species, actual_species):
        assert_allclose(expected["N"], actual["N"])
        assert_allclose(expected["Z"], actual["Z"])
        assert_allclose(expected["A"], actual["A"])


def test_read_dimensions():
    # Read real data and check dimensionality
    with open(_data_path / "test_1.peqdsk") as f:
        data = peqdsk.read(f)
    assert len(data["profiles"]) == 21
    assert len(data["species"]) == 3
    for profile in data["profiles"].values():
        assert len(profile["psinorm"]) == 201
        assert len(profile["data"]) == 201
        assert len(profile["derivative"]) == 201
    for species in data["species"]:
        assert len(species) == 3


def test_write_bad_species(tmp_path):
    """Should fail if species is missing columns"""
    profiles = {
        "x": {
            "psinorm": np.array([0.0, 1.0]),
            "data": np.array([0.0, 1.0]),
            "derivative": np.array([1.0, 1.0]),
            "units": "s",
        }
    }
    species = [
        {"N": 6.0, "Z": 6.0, "A": 12.0},
        {"N": 1.0, "Z": 1.0},  # missing "A"
        {"N": 1.0, "Z": 1.0, "A": 2.0},
    ]
    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_write_bad_species.peqdsk"
    with open(out, "w") as f, pytest.raises(KeyError):
        peqdsk.write({"profiles": profiles, "species": species}, f)


def test_write_bad_profiles(tmp_path):
    """Should fail if profiles is missing data"""
    # missing psinorm
    profiles = {
        "x": {
            "data": np.array([0.0, 1.0]),
            "derivative": np.array([1.0, 1.0]),
            "units": "s",
        }
    }
    species = [
        {"N": 6.0, "Z": 6.0, "A": 12.0},
        {"N": 1.0, "Z": 1.0, "A": 2.0},
        {"N": 1.0, "Z": 1.0, "A": 2.0},
    ]
    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_write_bad_profiles.peqdsk"
    with open(out, "w") as f, pytest.raises(KeyError):
        peqdsk.write({"profiles": profiles, "species": species}, f)


def test_write_missing_profiles(tmp_path):
    """Should fail if profiles is missing"""
    species = [
        {"N": 6.0, "Z": 6.0, "A": 12.0},
        {"N": 1.0, "Z": 1.0, "A": 2.0},
        {"N": 1.0, "Z": 1.0, "A": 2.0},
    ]
    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_write_missing_profiles.peqdsk"
    with open(out, "w") as f, pytest.raises(KeyError):
        peqdsk.write({"species": species}, f)


def test_write_missing_species(tmp_path):
    """Should fail if species is missing"""
    profiles = {
        "x": {
            "psinorm": np.array([0.0]),  # psinorm too short
            "data": np.array([0.0, 1.0]),
            "derivative": np.array([1.0, 1.0]),
            "units": "s",
        }
    }
    # Write to temp file
    d = tmp_path / "peqdsk"
    d.mkdir(exist_ok=True)
    out = d / "test_write_missing_species.peqdsk"
    with open(out, "w") as f, pytest.raises(KeyError):
        peqdsk.write({"profiles": profiles}, f)
