"""
SPDX-FileCopyrightText: © 2016 Ben Dudson, University of York.

SPDX-License-Identifier: MIT
"""

from io import StringIO

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from freeqdsk._fileutils import (
    ChunkOutput,
    f2s,
    read_array,
    read_line,
    write_array,
    write_line,
)


def test_f2s():
    assert f2s(0.0) == " 0.000000000E+00"
    assert f2s(1234) == " 1.234000000E+03"
    assert f2s(-1.65281e12) == "-1.652810000E+12"
    assert f2s(-1.65281e-2) == "-1.652810000E-02"


def test_ChunkOutput():
    output = StringIO()
    co = ChunkOutput(output)

    for val in [1.0, -3.2, 6.2e5, 8.7654e-12, 42.0, -76]:
        co.write(val)

    expected = "".join(
        [
            " 1.000000000E+00",
            "-3.200000000E+00",
            " 6.200000000E+05",
            " 8.765400000E-12",
            " 4.200000000E+01",
            "\n   -76",
        ]
    )

    assert output.getvalue() == expected


@pytest.mark.parametrize(
    "data,fmt,expected",
    [
        ("123456789\n", "(3i3)", (123, 456, 789)),
        ("hello world!\n", "(2a6)", ("hello ", "world!")),
        ("testing1212\n", "(a7,4i1)", ("testing", 1, 2, 1, 2)),
    ],
)
def test_read_line(data, fmt, expected):
    fh = StringIO(data)
    actual = read_line(fh, fmt)
    assert isinstance(actual, list)  # Unlike read_array, should not return np.ndarray
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "data,fmt,expected",
    [
        ((123, 456, 789), "(3i3)", "123456789\n"),
        (("hello ", "world!"), "(2a6)", "hello world!\n"),
        (("testing", 1, 2, 1, 2), "(a7,4i1)", "testing1212\n"),
    ],
)
def test_write_line(data, fmt, expected):
    fh = StringIO()
    write_line(data, fh, fmt)
    fh.seek(0)
    actual = fh.readline()
    assert actual == expected


def test_read_line_EOF():
    fh = StringIO()
    with pytest.raises(EOFError):
        read_line(fh, "(i6)")


def test_read_line_bad_fmt():
    fh = StringIO("hello world!\n")
    with pytest.raises(ValueError):
        read_line(fh, "(i6)")


def test_write_line_bad_fmt():
    fh = StringIO()
    with pytest.raises(ValueError):
        write_line(["hello", "world"], fh, "(i6)")


@pytest.mark.parametrize(
    "data,fmt,expected",
    [
        ("123\n456\n789\n", "(i3)", (123, 456, 789)),
        ("123456789\n", "(3i3)", (123, 456, 789)),
        (" 0.10E+01-0.15E+01\n 0.21E+01\n", "(2e9.2)", (1.0, -1.5, 2.1)),
        ("123456789\n", "(3i3)", (123, 456, 789)),
        ("012\n345\n67\n", "(3i1)", np.arange(8).reshape((2, 4), order="F")),
    ],
)
def test_read_array(data, fmt, expected):
    fh = StringIO(data)
    actual = read_array(np.shape(expected), fh, fmt)
    assert isinstance(actual, np.ndarray)
    assert_array_equal(actual.shape, np.shape(expected))
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "data,fmt,expected",
    [
        ((123, 456, 789), "(i3)", "123\n456\n789\n"),
        ((123, 456, 789), "(3i3)", "123456789\n"),
        ((1, -1.5, 2.1), "(2e9.2)", " 0.10E+01-0.15E+01\n 0.21E+01\n"),
        ((123, 456, 789), "(3i3)", "123456789\n"),
        (np.arange(8).reshape((2, 4), order="F"), "(3i1)", "012\n345\n67\n"),
    ],
)
def test_write_array(data, fmt, expected):
    fh = StringIO()
    write_array(data, fh, fmt)
    fh.seek(0)
    actual = "".join(fh.readlines())
    assert actual == expected


def test_read_array_EOF():
    fh = StringIO()
    with pytest.raises(EOFError):
        read_array(5, fh, "(i6)")


def test_read_array_bad_fmt():
    fh = StringIO(" 0.10E+05-0.56E-02\n")
    with pytest.raises(ValueError):
        read_array(3, fh, "(3i3)")


def test_read_array_bad_shape():
    fh = StringIO("123456789\n")
    with pytest.raises(ValueError):
        read_array("hello world", fh, "(3i3)")


def test_read_array_all():
    fh = StringIO("012\n345\n67\n")
    result = read_array("all", fh, "(3i1)")
    assert_array_equal(result, np.arange(8))


def test_read_array_hanging_values():
    # catch warning that the last value has been skipped
    fh = StringIO("123\n456\n789\n")
    with pytest.warns(UserWarning, match="beyond the end"):
        read_array(8, fh, "(3i1)")
