r"""
P-EQDSK files describe kinetics profiles as a function of :math:`\psi_N`, the normalised
poloidal flux function. This is defined such that :math:`\psi_N = 0` on the magnetic
axis of a tokamak, and :math:`\psi_N=1` on the last closed flux surface. Values in
the range :math:`[0,1]` are used to index each nested flux surface.

The data contained within a P-EQDSK file varies depending on the source, but it will
typically contain information such as ion/electron number densities and temperatures.
It also contains the derivatives of each variaible with respect to :math:`\psi_N`.
Data is represented in a csv-like format, and separated into blocks within the file.
Each block of kinetics profile data has the form::

    nrows psinorm var dvar/dpsiN
     0.000000   3.141592   2.718282
     ...
     1.000000   42.000000   -1.000000

where ``nrows`` is an integer specifying the number of rows in the block, and ``var``
is the variable described by the block.

At the bottom of a P-EQDSK file, a small block contains information describing the
atomic number, charge (in units of :math:`e`), and atomic mass of the ions::

    nrows N Z A of ION SPECIES
     6.000000   6.000000   12.000000
     1.000000   1.000000   2.000000
     1.000000   1.000000   2.000000

:: warning
    whitespace is not conserved when reading and writing P-EQDSK files. If an external
    P-EQDSK reader depends on a certain amount of whitespace between columns, further
    processing of the generated files will be necessary. Please get in touch on our
    `GitHub`_ page if this problem is affecting you.

.. _GitHub: https://github.com/freegs-plasma/FreeQDSK/issues
"""

import re
from io import StringIO
from typing import Dict, Generator, NamedTuple, TextIO

import pandas as pd


class PEQDSKData(NamedTuple):
    r"""
    NamedTuple containig a DataFrame of profiles and species. Returned by the
    `peqdsk.read` function.
    """
    #: Kinetics profiles, such as density, temperature, etc. and their derivatives with
    #: respect to :math:`\psi_N`.
    profiles: pd.DataFrame

    #: Atomic number, charge, and atomic mass of each species.
    species: pd.DataFrame

    #: The units of each profile.
    units: Dict[str, str]


def _peqdsk_blocks(fh: TextIO) -> Generator[pd.DataFrame, None, None]:
    r"""
    Given a file handle, reads a block from a P-EQDSK file into a Pandas DataFrame.
    Advances the file handle to the next block.

    Parameters
    ----------
    fh:
        File handle to read from.
    """
    # Each block starts with a header containing the number of lines in the block and
    # the column names. As pandas can't handle the exact csv format of a P-EQDSK file,
    # we read each block into a string buffer, and then pass that to pandas to create
    # each dataframe.
    while True:
        header = fh.readline().split()
        if not header:  # EOF
            return
        nrows = int(header[0])
        buf = StringIO("".join(fh.readline() for _ in range(nrows)))
        yield pd.read_csv(buf, names=header[1:4], delim_whitespace=True)


def read(fh: TextIO) -> PEQDSKData:
    r"""
    Given a file handle, reads a P-EQDSK file and returns a named tuple containing
    profile and species data. See the examples below for methods for retrieving this
    info.

    Parameters
    ----------
    fh:
        File handle. Should be in a text read mode, ``open(filename, "r")``.

    Examples
    --------

    As a named tuple::

        >>> with open("myfile.peqdsk") as f:
        ...    data = peqdsk.read(f)
        >>> profiles = data.profiles
        >>> species = data.species
        >>> units = data.units

    With tuple unpacking::

        >>> with open("myfile.peqdsk") as f:
        ...    profiles, species, units = peqdsk.read(f)

    Read only one type of data::

        >>> with open("myfile.peqdsk") as f:
        ...    data = peqdsk.read(f).profiles
    """
    blocks = _peqdsk_blocks(fh)
    profiles = next(blocks)
    species = pd.DataFrame()
    for df in blocks:
        if "psinorm" in df.columns:
            profiles = profiles.merge(df)
        else:
            species = df
    units = {}
    cols_to_rename = []
    for col_name in profiles.columns:
        match = re.search(r"(.*)\((.*)\)", col_name)
        if match is not None:
            cols_to_rename.append(col_name)
            units[match[1]] = match[2]
    profiles.rename(columns=dict(zip(cols_to_rename, units)), inplace=True)
    return PEQDSKData(profiles, species, units)


def _write_block(df: pd.DataFrame, units: Dict[str, str], fh: TextIO) -> None:
    """
    Utility function for writing each block to a file handle.

    Parameters
    ----------
    df:
        Pandas dataframe to write
    units:
        Dict of units to add to the column names
    fh:
        File handle.
    """
    # Write header separately to include nrows and add units
    cols = [(f"{col}({units[col]})" if col in units else col) for col in df.columns]
    if all(x in cols for x in ("N", "Z", "A")):
        cols.append("of ION SPECIES")
    fh.write(" ".join([str(len(df))] + cols) + "\n")
    # Write to string buffer, then out to file, as otherwise pandas will close the
    # file handle
    buf = StringIO()
    df.to_csv(buf, sep=" ", float_format="%.6f", header=False, index=False)
    buf.seek(0)
    fh.write(buf.read())


def write(
    profiles: pd.DataFrame,
    species: pd.DataFrame,
    units: Dict[str, str],
    fh: TextIO,
) -> None:
    r"""
    Given a file handle and the components of a PEQDSKData named tuple object, write a
    P-EQDSK file. See the examples below for examples of how to use this function.

    Parameters
    ----------
    profiles:
        Pandas DataFrame containing kinetics profile data with respect to ``"psinorm"``.
    species:
        Pandas DataFrame containing atomic number, charge number, and mass number for
        the impurity, majority, and fast ion species.
    units: Dict[str,str],
        A dict of units for the columns within profiles.
    fh:
        File handle. Should be opened in a text write mode, ``open(filename, "w")``.

    Raises
    ------
    ValueError
        If ``"psinorm"`` is missing from profiles, or the species DataFrame is
        malformed.

    Examples
    --------

    If we read a file as follows::

        >>> with open("myfile.peqdsk") as f:
        ...    data = peqdsk.read(f)

    We can write by referencing each component of ``data`` in turn::

        >>> with open("newfile.peqdsk", "w") as f:
        ...    peqdsk.write(data.profiles, data.species, data.units, f)

    Or we can simplify this using tuple unpacking::

        >>> with open("newfile.peqdsk", "w") as f:
        ...    peqdsk.write(*data, f)
    """
    if "psinorm" not in profiles:
        raise ValueError("profiles DataFrame missing 'psinorm'")
    if not all(species.columns == ["N", "Z", "A"]):
        raise ValueError("species DataFrame has incorrect column headings")

    for cols in zip(profiles.columns[1::2], profiles.columns[2::2]):
        _write_block(profiles[["psinorm", *cols]], units, fh)
    _write_block(species, {}, fh)
