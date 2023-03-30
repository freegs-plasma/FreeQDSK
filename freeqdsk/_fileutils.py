"""
Utilities for writing and reading files compatible with Fortran

SPDX-FileCopyrightText: © 2016 Ben Dudson, University of York.

SPDX-License-Identifier: MIT

"""
from __future__ import annotations  # noqa

import re
import warnings
from contextlib import contextmanager
from typing import Any, Generator, Iterable, List, TextIO, Union

import fortranformat as ff
import numpy as np
from numpy.typing import ArrayLike


def f2s(f: float) -> str:
    r"""
    Format a string containing a float.

    Positive floats require a extra space in front to ensure that positive and negative
    values have the same width.

    Parameters
    ----------
    f:
        A single float to be converted to a string.
    """
    warnings.warn(
        "freeqdsk._fileutils.f2s is deprecated, and will be removed in version 0.4",
        DeprecationWarning,
    )
    return f"{' ' if f >= 0.0 else ''}{f:1.9E}"


class ChunkOutput:
    r"""
    Outputs values in lines, inserting newlines when needed.

    Parameters
    ---------
    filehandle:
        Output to write to.
    chunksize:
        Number of values per line.
    extraspaces:
        Number of extra spaces between outputs
    """

    def __init__(self, filehandle: TextIO, chunksize: int = 5, extraspaces: int = 0):
        warnings.warn(
            "freeqdsk._fileutils.ChunkOutput is deprecated, and will be removed in "
            "version 0.4",
            DeprecationWarning,
        )
        self.fh = filehandle
        self.counter = 0
        self.chunk = chunksize
        self.extraspaces = extraspaces

    def write(self, value: Union[int, float, List[Any]]) -> None:
        r"""
        Write a value to the output, adding a newline if needed

        Distinguishes between:
        - list  : Iterates over the list and writes each element
        - int   : Converts using str
        - float : Converts using f2s to Fortran-formatted string

        Parameters
        ----------
        value:
            Either a single int or float, or a list to output. If provided with a list,
            the function is called recursively on each element.
        """
        if isinstance(value, list):
            for elt in value:
                self.write(elt)
            return

        self.fh.write(" " * self.extraspaces)

        if isinstance(value, int):
            self.fh.write(f"   {value}")
        else:
            self.fh.write(f2s(value))

        self.counter += 1
        if self.counter == self.chunk:
            self.fh.write("\n")
            self.counter = 0

    def newline(self) -> None:
        """
        Ensure that the file is at the start of a new line. If the file is already
        at a newline, does nothing.
        """
        if self.counter != 0:
            self.endblock()

    def endblock(self) -> None:
        """
        Make sure next block of data is on new line.
        """
        self.fh.write("\n")
        self.counter = 0

    def __enter__(self):
        """
        Entry point for ``with`` statements.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit point for ``with`` statements.

        Ensures that the chunk finishes with a new line.
        """
        self.newline()


def write_1d(values: Iterable[Any], out: ChunkOutput) -> None:
    r"""
    Writes a 1D variable to a ChunkOutput file handle.

    Parameters
    ----------
    values:
        List of values to write.
    out:
       File handle managed by a ``ChunkOutput`` object.
    """
    warnings.warn(
        "freeqdsk._fileutils.write_1d is deprecated, and will be removed in "
        "version 0.4",
        DeprecationWarning,
    )
    for value in values:
        out.write(value)
    out.newline()


def write_2d(values: ArrayLike, out: ChunkOutput) -> None:
    r"""
    Writes a 2D array to a ChunkOutput file handle.

    Note that this transposes the array, looping over the first index fastest

    Parameters
    ----------
    values:
        List of values to write.
    out:
       File handle managed by a ``ChunkOutput`` object.

    Raises
    ------
    ValueError
        If values is not a 2D array.
    """
    warnings.warn(
        "freeqdsk._fileutils.write_2d is deprecated, and will be removed in "
        "version 0.4",
        DeprecationWarning,
    )
    # Alt: Check values is 2D, then write_1d(np.asarray(values).T.ravel(), out)
    nx, ny = np.shape(values)
    for y in range(ny):
        for x in range(nx):
            out.write(values[x, y])
    out.newline()


def next_value(fh: TextIO) -> Generator[Union[int, float], None, None]:
    """
    A generator which yields values from a file handle.

    Checks if the value is a float or int, returning the correct type depending on
    whether '.' is in the string.

    Parameters
    ----------
    fh:
        File handle for text file to be read.

    Yields
    ------
    Union[int, float]
        Yields either an int or a float, depending on whether the string contains a
        decimal point.
    """
    warnings.warn(
        "freeqdsk._fileutils.next_value is deprecated, and will be removed in "
        "version 0.4",
        DeprecationWarning,
    )
    pattern = re.compile(r"[ +\-]?\d+(?:\.\d+(?:[Ee][\+\-]\d\d)?)?")

    # Go through each line, extract values, then yield them one by one
    for line in fh:
        matches = pattern.findall(line)
        for match in matches:
            if "." in match:
                yield float(match)
            else:
                yield int(match)


@contextmanager
def _fortranformat_written_vars_only():
    original = ff.config.RET_WRITTEN_VARS_ONLY
    ff.config.RET_WRITTEN_VARS_ONLY = True
    yield
    ff.config.RET_WRITTEN_VARS_ONLY = original


def read_line(fh: TextIO, fmt: str) -> List[Any]:
    r"""
    Reads a single line from a Fortran formatted ASCII data file. Advances the
    file handle a single line.

    Parameters
    ---------
    fh:
        File handle. Should be in a text read mode, i.e. ``open(filename, "r")``.
    fmt:
        A Fortran format string, such as ``'(6a8,3i4)'``

    Raises
    ------
    ValueError
        If attempting to read a line which does not match the supplied format.
    EOFError
        If the file handle has reached the end of the file.
    """
    with _fortranformat_written_vars_only():
        line = fh.readline()
        if not line:
            raise EOFError("Encountered EOF while reading array")
        return ff.FortranRecordReader(fmt).read(line)


def read_array(shape: Union[int, str, ArrayLike], fh: TextIO, fmt: str) -> np.ndarray:
    r"""
    Reads from a Fortran formatted ASCII data file. It is assumed that the array is
    flattened and stored in Fortran order (column-major). Information is read from a
    file handle until the requested array is filled.

    Parameters
    ---------
    shape:
        The shape of the array to return. If provided as an int, a 1D array is returned
        of length ``shape``. If passed the string ``"all"``, will read until the end of
        the file.
    fh:
        File handle. Should be in a text read mode, i.e. ``open(filename, "r")``.
    fmt:
        A Fortran format string, such as ``'(5e16.9)'``

    Raises
    ------
    ValueError
        If ``shape`` is not a scalar, a 1D iterable, or the string ``"all"``. Also
        raised if attempting to read a line which does not match the supplied format.
    EOFError
        If encountering end-of-file while reading an array, and shape if not ``"all"``

    Warns
    -----
    UserWarning
        When reading an array over multiple lines, raises a warning if the array reaches
        the requested size but there are still elements on the last line. These elements
        will be discarded, and the filehandle will move on the next line.
    """
    # If shape is "all", re-run the function using a special shape.
    # This instructs the function to read until end-of-file and return an array of
    # undetermined shape
    shape_all = -44379512921
    if shape == "all":
        return read_array(shape_all, fh, fmt)

    shape = np.asanyarray(shape, dtype=int)

    # Reject >1D shapes
    if shape.ndim not in (0, 1):
        raise ValueError("'shape' should be a scalar or a 1D array")

    # For ND arrays, read in the total number of elements, then reshape the result
    if shape.ndim == 1:
        return read_array(np.prod(shape), fh, fmt).reshape(shape, order="F")

    # Read a 1D array and append to result list until finished
    reader = ff.FortranRecordReader(fmt)
    with _fortranformat_written_vars_only():
        result = []
        if shape == shape_all:
            # Read until EOF
            for line in fh.readlines():
                result.extend(reader.read(line))
        else:
            while len(result) < shape:
                line = fh.readline()
                if not line:
                    raise EOFError("Encountered EOF while reading array")
                result.extend(reader.read(line))
            if len(result) != shape:
                warnings.warn(
                    "Additional elements were detected beyond the end of the "
                    "requested array. These have been discarded."
                )
                result = result[:shape]
    return np.array(result)


def write_line(data: Iterable[Any], fh: TextIO, fmt: str) -> None:
    r"""
    Writes to a Fortran formatted ASCII data file. The file handle will be left on a
    newline.

    Parameters
    ---------
    data:
        The data to write.
    fh:
        File handle. Should be in a text write mode, i.e. ``open(filename, "w")``.
    fmt:
        A Fortran IO format string, such as ``'(6a8,3i3)'``.
    """
    fh.write(ff.FortranRecordWriter(fmt).write(data))
    fh.write("\n")


def write_array(arr: ArrayLike, fh: TextIO, fmt: str) -> None:
    r"""
    Writes to a Fortran formatted ASCII data file. The provided array is flattened and
    written in Fortran order (column-major). Information is written to a file handle
    until the requested array is written. The file handle will be left on a newline.

    Parameters
    ---------
    arr:
        The array to write.
    fh:
        File handle. Should be in a text write mode, i.e. ``open(filename, "w")``.
    fmt:
        A Fortran IO format string, such as ``'(5e16.9)'``.
    """
    arr = np.asanyarray(arr)
    # Quit early if given empty array
    if arr.size == 0:
        return
    if arr.ndim == 0:
        # If given scalar, convert to array of shape (1,)
        write_array((arr,), fh, fmt)
    elif arr.ndim == 1:
        # Write 1D array
        fh.write(ff.FortranRecordWriter(fmt).write(arr))
        fh.write("\n")
    else:
        # If given ND array, flatten to 1D and write
        write_array(arr.ravel(order="F"), fh, fmt)
