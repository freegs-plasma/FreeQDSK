"""
Utilities for writing and reading files compatible with Fortran

SPDX-FileCopyrightText: © 2016 Ben Dudson, University of York.

SPDX-License-Identifier: MIT

"""
from __future__ import annotations  # noqa

import re
from typing import Any, Generator, Iterable, List, TextIO, Union

import numpy as np
from numpy.typing import ArrayLike


def f2s(f: float) -> str:
    r"""
    Format a string containing a float.

    Positive floats require a extra space in front to ensure that positive and negative
    values have the same width.

    Parameters
    ----------
    f: float
        A single float to be converted to a string.

    Returns
    -------
    str
        Converted string.
    """
    return f"{' ' if f >= 0.0 else ''}{f:1.9E}"


class ChunkOutput:
    r"""
    Outputs values in lines, inserting newlines when needed.

    Parameters
    ---------
    filehandle: TextIO
        Output to write to.
    chunksize: int, default 5
        Number of values per line.
    extraspaces: int, default 0
        Number of extra spaces between outputs
    """

    def __init__(self, filehandle: TextIO, chunksize: int = 5, extraspaces: int = 0):
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
        value: Union[int, float, List[Any]]
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
    values: Iterable[Any]
        List of values to write.
    out: ChunkOutput
       File handle managed by a ``ChunkOutput`` object.
    """
    for value in values:
        out.write(value)
    out.newline()


def write_2d(values: ArrayLike, out: ChunkOutput) -> None:
    r"""
    Writes a 2D array to a ChunkOutput file handle.

    Note that this transposes the array, looping over the first index fastest

    Parameters
    ----------
    values: ArrayLike
        List of values to write.
    out: ChunkOutput
       File handle managed by a ``ChunkOutput`` object.

    Raises
    ------
    ValueError
        If values is not a 2D array.
    """
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
    fh: TextIO
        File handle for text file to be read.

    Yields
    ------
    Union[int, float]
        Yields either an int or a float, depending on whether the string contains a
        decimal point.
    """
    pattern = re.compile(r"[ +\-]?\d+(?:\.\d+(?:[Ee][\+\-]\d\d)?)?")

    # Go through each line, extract values, then yield them one by one
    for line in fh:
        matches = pattern.findall(line)
        for match in matches:
            if "." in match:
                yield float(match)
            else:
                yield int(match)
