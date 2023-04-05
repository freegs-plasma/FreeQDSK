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
"""

import csv
import re
from typing import Dict, Generator, List, TextIO, Tuple, TypedDict, Union

import numpy as np


class ProfileDict(TypedDict):
    r"""
    TypedDict describing an individual kinetics profile.
    """
    #: :math:`\psi_N` grid, where :math:`\psi_N=0` on the magnetic axis and
    #: :math:`\psi_N=1` on the last closed flux surface
    psinorm: np.ndarray

    #: Kinetics profile
    data: np.ndarray

    #: Derivative of ``profile`` with respect to ``psinorm``
    derivative: np.ndarray

    #: Units of ``profile``
    units: str


class SpeciesDict(TypedDict):
    r"""
    TypedDict describing each species.
    """
    #: Atomic number
    N: float

    #: Charge (units of :math:`e`)
    Z: float

    #: Atomic mass
    A: float


class PEQDSKDict(TypedDict):
    r"""
    TypedDict returned by the read function.
    """
    #: Dict of kinetics profiles. The names of each profile are used as keys, while
    #: the data is presented in a ``ProfileDict``.
    profile: Dict[str, ProfileDict]

    #: List of species.
    species: List[SpeciesDict]


_newline = "\n"

#: keywords to pass to ``csv.reader/writer``
_fmt_kwargs = {
    "delimiter": " ",
    "skipinitialspace": True,
    "quoting": csv.QUOTE_NONNUMERIC,
    "lineterminator": _newline,
}


def _read_peqdsk_blocks(
    fh: TextIO,
) -> Generator[Tuple[str, Union[ProfileDict, List[SpeciesDict]]], None, None]:
    r"""
    Given a file handle, reads a block from a P-EQDSK file and returns either profile
    data or species data. When reading profile data, the first object returned will
    be the name of the profile. When reading species data, the first object returned
    will be the string ``"__species__"``.

    Parameters
    ----------
    fh:
        File handle to read from.
    """
    units_regex = re.compile(r"(?P<name>.*)\((?P<units>.*)\)")
    while True:
        header = fh.readline().split()
        if not header:  # EOF
            return
        nrows = int(header[0])

        if header[-1] == "SPECIES":
            reader = csv.DictReader(fh, fieldnames=("N", "Z", "A"), **_fmt_kwargs)
            species = []
            for _, row in zip(range(nrows), reader):
                species.append(row)

            yield "__species__", species
            continue

        # Get name and units
        match = units_regex.search(header[2])
        if match is None:
            raise ValueError(f"Unrecognised header format: '{' '.join(header)}'")
        name = match["name"]
        units = match["units"]

        # Read each row into a numpy array
        reader = csv.reader(fh, **_fmt_kwargs)
        psinorm = np.empty(nrows)
        data = np.empty(nrows)
        derivative = np.empty(nrows)
        for idx, row in zip(range(nrows), reader):
            psinorm[idx], data[idx], derivative[idx] = row

        # Assemble into ProfileDict and return
        profile: ProfileDict = {
            "psinorm": psinorm,
            "data": data,
            "derivative": derivative,
            "units": units,
        }
        yield name, profile


def read(fh: TextIO) -> PEQDSKDict:
    r"""
    Given a file handle, reads a P-EQDSK file and returns a dict containing
    profile and species data.

    The returned dict has two entries:

    - ``data["profiles"]`` is a dict of keys and ``ProfileDict``. For example,
      to access the electron density data, use ``data["profiles"]["ne"]["data"]``. To
      access the ion density dervative with respect to :math:`\psi_N`, use
      ``data["profiles"]["ni"]["derivative"]``.
    - ``data["species"]`` is a list of ``SpeciesDict``. To access the atomic number of
      the first entry, use ``data["species"][0]["N"]``.

    Parameters
    ----------
    fh:
        File handle. Should be in a text read mode, ``open(filename, "r")``.
    """
    profiles = {}
    species = {}
    for name, block in _read_peqdsk_blocks(fh):
        if name == "__species__":
            species = block
        else:
            profiles[name] = block
    result: PEQDSKDict = {"profiles": profiles, "species": species}
    return result


def _write_profile(name: str, profile: ProfileDict, fh: TextIO) -> None:
    """
    Utility function for writing each profile block to a file handle.
    """
    nrows = len(profile["psinorm"])
    header = f"{nrows} psinorm {name}({profile['units']}) d{name}/dpsiN{_newline}"
    fh.write(header)
    for psi, x, dx in zip(profile["psinorm"], profile["data"], profile["derivative"]):
        # Rather than using built-in csv writer, using f-strings as they give more
        # control over whitespace
        fh.write(f" {psi:.6f}   {x:.6f}   {dx:.6f}{_newline}")


def write(data: PEQDSKDict, fh: TextIO) -> None:
    r"""
    Given a file handle and a ``PEQDSKDict`` dict, write a P-EQDSK file. The provided
    dict should have the same structure as that returned by the ``read`` function.

    Parameters
    ----------
    data:
        Dict of P-EQDSK data. Should be in the format of a ``PEQDSKDict``, which is
        itself composed of ``ProfileDict`` and ``SpeciesDict`` dicts.
    fh:
        File handle. Should be opened in a text write mode, ``open(filename, "w")``.
    """
    for name, profile in data["profiles"].items():
        _write_profile(name, profile, fh)

    n_species = len(data["species"])
    fh.write(f"{n_species} N Z A of ION SPECIES{_newline}")
    for species in data["species"]:
        N, Z, A = species["N"], species["Z"], species["A"]
        # Rather than using built-in csv writer, using f-strings as they give more
        # control over whitespace
        fh.write(f" {N:6f}   {Z:6f}   {A:6f}{_newline}")
