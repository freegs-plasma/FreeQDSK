"""
SPDX-FileCopyrightText: © 2020 Ben Dudson, University of York.

SPDX-License-Identifier: MIT

"""

from __future__ import annotations  # noqa

import itertools
import warnings
from dataclasses import dataclass
from textwrap import dedent
from typing import Dict, Optional, TextIO, Union

import fortranformat as ff
import numpy as np
from numpy.typing import ArrayLike

from ._fileutils import read_array, write_array


#: Default format for all float data
_data_format = "(4e16.9)"

#: Default format for the line beginning the 'extended' portion of the file
_extended_sizes_format = "(4i4)"

#: Default format for time stamps
_time_format = "(1e16.9)"


@dataclass
class Field:
    """
    Helper ``dataclass`` for defining each entry in a A-EQDSK file.
    """

    #: Name of the field used in the resulting dict.
    name: str

    #: Description of the field.
    description: str = ""

    #: Default value to be assigned to missing fields when writing out an A-EQDSK dict.
    #: If set to None, the attributes `has_length` and `length_of` are used to
    #: determine a default
    default: Optional[float] = None

    #: If ``default is None`` and ``has_length is not None``, the default value is
    #: determined by looking up the key assigned to ``has_length``, and creating a list
    #: of 0.0 with that length. If this key does not exist, sets the value to ``[]``.
    has_length: Optional[str] = None

    #: If ``default is None`` and ``length_of is not None``, the default value is
    #: determined by looking up the key assigned to ``length_of``, and returning the
    #: length of the result. If this key does not exist, sets the value to ``0``.
    length_of: Optional[str] = None


#: The initial block of fields up to CO2 laser bits
_general_block_1 = [
    Field(
        "tsaisq",
        description=(
            "total chi2 from magnetic probes, flux loops, Rogowski and external coils"
        ),
        default=0.0,
    ),
    Field(
        "rcencm",
        description="major radius in cm for vacuum field BCENTR",
        default=100.0,
    ),
    Field(
        "bcentr",
        description="vacuum toroidal magnetic field in Tesla at RCENCM",
        default=1.0,
    ),
    Field(
        "pasmat",
        description="measured plasma toroidal current in Ampere",
        default=1e6,
    ),
    Field(
        "cpasma",
        description="fitted plasma toroidal current in Ampere-turn",
        default=1e6,
    ),
    Field("rout", description="major radius of geometric center in cm", default=100.0),
    Field("zout", description="Z of geometric center in cm", default=0.0),
    Field("aout", description="plasma minor radius in cm", default=50.0),
    Field("eout", description="Plasma boundary elongation", default=1.0),
    Field("doutu", description="upper triangularity", default=1.0),
    Field("doutl", description="lower triangularity", default=1.0),
    Field("vout", description="plasma volume in cm3", default=1000.0),
    Field(
        "rcurrt", description="major radius in cm of current centroid", default=100.0
    ),
    Field("zcurrt", description="Z in cm at current centroid", default=0.0),
    Field("qsta", description="equivalent safety factor q*", default=5.0),
    Field("betat", description="toroidal beta in %", default=1.0),
    Field(
        "betap",
        description=(
            "poloidal beta with normalization average poloidal magnetic BPOLAV defined "
            "through Ampere's law"
        ),
        default=1.0,
    ),
    Field(
        "ali",
        description=(
            "li with normalization average poloidal magnetic defined through Ampere's "
            "law"
        ),
        default=0.0,
    ),
    Field("oleft", description="plasma inner gap in cm", default=10.0),
    Field("oright", description="plasma outer gap in cm", default=10.0),
    Field("otop", description="plasma top gap in cm", default=10.0),
    Field("obott", description="plasma bottom gap in cm", default=10.0),
    Field("qpsib", description="q at 95% of poloidal flux", default=5.0),
    Field(
        "vertn",
        description="vacuum field (index? seems to be float) at current centroid",
        default=1.0,
    ),
]

#: Arrays describing CO2 laser bits
_laser_block = [
    Field(
        "rco2v",
        description="1D array : path length in cm of vertical CO2 density chord",
        has_length="mco2v",
    ),
    Field(
        "dco2v",
        description="line average electron density in cm3 from vertical CO2 chord",
        has_length="mco2v",
    ),
    Field(
        "rco2r",
        description="path length in cm of radial CO2 density chord",
        has_length="mco2r",
    ),
    Field(
        "dco2r",
        description="line average electron density in cm3 from radial CO2 chord",
        has_length="mco2r",
    ),
]

#: Further information up to the extended part of the file
_general_block_2 = [
    Field("shearb", default=0.0),
    Field(
        "bpolav",
        description=(
            "average poloidal magnetic field in Tesla defined through Ampere's law"
        ),
        default=1.0,
    ),
    Field("s1", description="Shafranov boundary line integrals", default=0.0),
    Field("s2", description="Shafranov boundary line integrals", default=0.0),
    Field("s3", description="Shafranov boundary line integrals", default=0.0),
    Field("qout", description="q at plasma boundary", default=0.0),
    Field("olefs", default=0.0),
    Field(
        "orighs",
        description="outer gap of external second separatrix in cm",
        default=0.0,
    ),
    Field(
        "otops",
        description="top gap of external second separatrix in cm",
        default=0.0,
    ),
    Field("sibdry", default=1.0),
    Field("areao", description="cross sectional area in cm2", default=100.0),
    Field("wplasm", default=0.0),
    Field("terror", description="equilibrium convergence error", default=0.0),
    Field("elongm", description="elongation at magnetic axis", default=0.0),
    Field("qqmagx", description="axial safety factor q(0)", default=0.0),
    Field("cdflux", description="computed diamagnetic flux in Volt-sec", default=0.0),
    Field(
        "alpha", description="Shafranov boundary line integral parameter", default=0.0
    ),
    Field(
        "rttt", description="Shafranov boundary line integral parameter", default=0.0
    ),
    Field("psiref", description="reference poloidal flux in VS/rad", default=1.0),
    Field(
        "xndnt",
        description=(
            "vertical stability parameter, vacuum field index normalized to "
            "critical index value"
        ),
        default=0.0,
    ),
    Field("rseps1", description="major radius of x point in cm", default=1.0),
    Field("zseps1", default=-1.0),
    Field("rseps2", description="major radius of x point in cm", default=1.0),
    Field("zseps2", default=1.0),
    Field("sepexp", description="separatrix radial expansion in cm", default=0.0),
    Field(
        "obots",
        description="bottom gap of external second separatrix in cm",
        default=0.0,
    ),
    Field(
        "btaxp",
        description="toroidal magnetic field at magnetic axis in Tesla",
        default=1.0,
    ),
    Field(
        "btaxv",
        description="vacuum toroidal magnetic field at magnetic axis in Tesla",
        default=1.0,
    ),
    Field(
        "aaq1",
        description="minor radius of q=1 surface in cm, 100 if not found",
        default=100.0,
    ),
    Field(
        "aaq2",
        description="minor radius of q=2 surface in cm, 100 if not found",
        default=100.0,
    ),
    Field(
        "aaq3",
        description="minor radius of q=3 surface in cm, 100 if not found",
        default=100.0,
    ),
    Field(
        "seplim",
        description=(
            "> 0 for minimum gap in cm in divertor configurations, < 0 absolute "
            "value for minimum distance to external separatrix in limiter "
            "configurations"
        ),
        default=0.0,
    ),
    Field("rmagx", description="major radius in cm at magnetic axis", default=100.0),
    Field("zmagx", default=0.0),
    Field("simagx", description="Poloidal flux at the magnetic axis", default=0.0),
    Field("taumhd", description="energy confinement time in ms", default=0.0),
    Field("betapd", description="diamagnetic poloidal b", default=0.0),
    Field("betatd", description="diamagnetic toroidal b in %", default=0.0),
    Field(
        "wplasmd", description="diamagnetic plasma stored energy in Joule", default=0.0
    ),
    Field("diamag", description="measured diamagnetic flux in Volt-sec", default=0.0),
    Field("vloopt", description="measured loop voltage in volt", default=0.0),
    Field(
        "taudia", description="diamagnetic energy confinement time in ms", default=0.0
    ),
    Field(
        "qmerci",
        description=(
            "Mercier stability criterion on axial q(0), q(0) > QMERCI for stability"
        ),
        default=0.0,
    ),
    Field(
        "tavem", description="average time in ms for magnetic and MSE data", default=0.0
    ),
]

#: Array sizes for the extended part of the file
_extended_sizes = [
    Field(
        "nsilop",
        description="Number of flux loop signals, len(csilop)",
        length_of="csilop",
    ),
    Field(
        "magpri",
        description="Number of flux loop signals, len(cmpr2) (added to nsilop)",
        length_of="cmpr2",
    ),
    Field(
        "nfcoil",
        description="Number of calculated external coil currents, len(ccbrsp)",
        length_of="ccbrsp",
    ),
    Field(
        "nesum", description="Number of measured E-coil currents", length_of="eccurt"
    ),
]

#: The next two arrays are joined together, so need special treatment
_extended_arrays_1 = [
    Field(
        "csilop", description="computed flux loop signals in Weber", has_length="nsilop"
    ),
    Field("cmpr2", has_length="magpri"),
]

#: The following two arrays are stored normally
_extended_arrays_2 = [
    Field(
        "ccbrsp",
        description="computed external coil currents in Ampere",
        has_length="nfcoil",
    ),
    Field(
        "eccurt", description="measured E-coil current in Ampere", has_length="nesum"
    ),
]

#: We finish with another standard block
_extended_general = [
    Field("pbinj", description="neutral beam injection power in Watts", default=0.0),
    Field(
        "rvsin", description="major radius of vessel inner hit spot in cm", default=0.0
    ),
    Field("zvsin", description="Z of vessel inner hit spot in cm", default=0.0),
    Field(
        "rvsout", description="major radius of vessel outer hit spot in cm", default=0.0
    ),
    Field("zvsout", description="Z of vessel outer hit spot in cm", default=0.0),
    Field(
        "vsurfa",
        description="plasma surface loop voltage in volt, E EQDSK only",
        default=0.0,
    ),
    Field(
        "wpdot",
        description="time derivative of plasma stored energy in Watt, E EQDSK only",
        default=0.0,
    ),
    Field(
        "wbdot",
        description="time derivative of poloidal magnetic energy in Watt, E EQDSK only",
        default=0.0,
    ),
    Field("slantu", default=0.0),
    Field("slantl", default=0.0),
    Field("zuperts", default=0.0),
    Field("chipre", description="total chi2 pressure", default=0.0),
    Field("cjor95", default=0.0),
    Field(
        "pp95",
        description="normalized P'(y) at 95% normalized poloidal flux",
        default=0.0,
    ),
    Field("ssep", default=0.0),
    Field("yyy2", description="Shafranov Y2 current moment", default=0.0),
    Field("xnnc", default=0.0),
    Field(
        "cprof", description="current profile parametrization parameter", default=0.0
    ),
    Field("oring", description="not used", default=0.0),
    Field(
        "cjor0",
        description=(
            "normalized flux surface average current density at 99% of normalized "
            "poloidal flux"
        ),
        default=0.0,
    ),
    Field("fexpan", description="flux expansion at x point", default=0.0),
    Field("qqmin", description="minimum safety factor qmin", default=0.0),
    Field("chigamt", description="total chi2 MSE", default=0.0),
    Field(
        "ssi01",
        description="magnetic shear at 1% of normalized poloidal flux",
        default=0.0,
    ),
    Field(
        "fexpvs",
        description="flux expansion at outer lower vessel hit spot",
        default=0.0,
    ),
    Field(
        "sepnose",
        description=(
            "radial distance in cm between x point and external field line at ZNOSE"
        ),
        default=0.0,
    ),
    Field(
        "ssi95",
        description="magnetic shear at 95% of normalized poloidal flux",
        default=0.0,
    ),
    Field(
        "rqqmin",
        description="normalized radius of qmin , square root of normalized volume",
        default=0.0,
    ),
    Field("cjor99", default=0.0),
    Field(
        "cj1ave",
        description=(
            "normalized average current density in plasma outer 5% normalized poloidal "
            "flux region"
        ),
        default=0.0,
    ),
    Field("rmidin", description="inner major radius in m at Z=0.0", default=0.0),
    Field("rmidout", description="outer major radius in m at Z=0.0", default=0.0),
]
# TODO add extra lines from eqtools


def _field_value(
    field: Field, data: Dict[str, Union[float, int, ArrayLike]]
) -> Union[float, int, ArrayLike]:
    """
    Returns field data from dict if present. Otherwise returns a default.
    """
    if field.name in data:
        return data[field.name]
    elif field.default is not None:
        return field.default
    elif field.has_length is not None:
        if field.has_length in data:
            return [0.0] * data[field.has_length]
        else:
            return []
    elif field.length_of is not None:
        if field.length_of in data:
            return len(data[field.length_of])
        else:
            return 0
    else:
        raise KeyError(
            f"The field {field.name} is not in data, and no default could be determined"
        )


def write(
    data: Dict[str, Union[float, int, ArrayLike]],
    fh: TextIO,
    data_format: Optional[str] = None,
    extended_sizes_format: Optional[str] = None,
    time_format: Optional[str] = None,
) -> None:
    """
    Write a dict of A-EQDSK data to a file.

    Parameters
    ----------
    data: Dict[str, Union[float, int, ArrayLike]]
        The A-EQDSK data to write to disk. It may also include header information.
    fh: TextIO
        File handle. Should be in a text write mode, i.e.``open(filename, "w")``.
    data_format: Optional[str], default None
        Fortran IO format for A-EQDSK data. If not provided, uses ``(4e16.9)``.
    extended_sizes_format: Optional[str], default None
        Fortran IO format for the line specifying array lengths in the extended portion
        of the A-EQDSK file. If not provided, uses ``(4i4)``.
    time_format: Optional[str], default None
        Fortran IO format for time stamps. If not provided, uses ``(1e16.9)``.
    """
    # TODO Need proper header format

    if data_format is None:
        data_format = _data_format
    if extended_sizes_format is None:
        extended_sizes_format = _extended_sizes_format
    if time_format is None:
        time_format = _time_format
    data_writer = ff.FortranRecordWriter(data_format)
    extended_sizes_writer = ff.FortranRecordWriter(extended_sizes_format)
    time_writer = ff.FortranRecordWriter(time_format)

    # First line identification string
    # Default to date > 1997 since that format includes nsilop etc.
    fh.write("{0:11s}\n".format(data.get("header", " 26-OCT-98 09/07/98  ")))

    # Second line shot number
    fh.write(" {:d}               1\n".format(data.get("shot", 0)))

    # Third line time
    time = data.get("time", 0.0)
    write_array([time], fh, time_writer)

    # Fourth line
    # time(jj),jflag(jj),lflag,limloc(jj), mco2v,mco2r,qmflag
    #   jflag = 0 if error  (? Seems to contradict example)
    #   lflag > 0 if error  (? Seems to contradict example)
    #   limloc  IN/OUT/TOP/BOT: limiter inside/outside/top/bot SNT/SNB: single null
    #       top/bottom DN: double null
    #   mco2v   number of vertical CO2 density chords
    #   mco2r   number of radial CO2 density chords
    #   qmflag  axial q(0) flag, FIX if constrained and CLC for float
    fh.write(
        "*{:s}             {:d}                {:d} {:s}  {:d}   {:d} {:s}\n".format(
            time_writer.write([time]).strip(),
            data.get("jflag", 1),
            data.get("lflag", 0),
            data.get("limloc", "DN"),
            data.get("mco2v", 0),
            data.get("mco2r", 0),
            data.get("qmflag", "CLC"),
        )
    )

    # Output first block of general data
    write_array(
        [_field_value(field, data) for field in _general_block_1], fh, data_writer
    )

    # Output laser bits
    for field in _laser_block:
        write_array(_field_value(field, data), fh, data_writer)

    # Output second block of general data
    write_array(
        [_field_value(field, data) for field in _general_block_2], fh, data_writer
    )

    # Check if we need to write an extended section
    extended = False
    extended_blocks = [
        _extended_sizes,
        _extended_arrays_1,
        _extended_arrays_2,
        _extended_general,
    ]
    for field in itertools.chain.from_iterable(extended_blocks):
        if field.name in data:
            extended = True
            break

    if extended:
        # Write extended portion of the file
        write_array(
            [_field_value(field, data) for field in _extended_sizes],
            fh,
            extended_sizes_writer,
        )

        # First two arrays are joined because... reasons
        write_array(
            np.concat([_field_value(field, data) for field in _extended_arrays_1]),
            fh,
            data_writer,
        )

        # Next two arrays are arranged in a standard pattern
        for field in _extended_arrays_2:
            write_array(_field_value(field, data), fh, data_writer)

        # Write a final general block
        # Find the last field that is present within data. Write up to there and no
        # further, filling default values on the way
        last_field = -1
        for idx, field in enumerate(_extended_general):
            if field.name in data:
                last_field = idx
        write_array(
            [_field_value(field, data) for field in _extended_general[:last_field]],
            fh,
            data_writer,
        )


def read(
    fh: TextIO,
    data_format: Optional[str] = None,
    extended_sizes_format: Optional[str] = None,
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Read an A-EQDSK file, returning a dictionary of data.

    Parameters
    ----------
    fh: TextIO
        File handle to write to. Should be opened in a text read mode, i.e.
        ``open(filename, "r")``.
    data_format: Optional[str], default None
        Fortran IO format for A-EQDSK data. If not provided, uses ``(4e16.9)``.
    extended_sizes_format: Optional[str], default None
        Fortran IO format for the line specifying array lengths in the extended portion
        of the A-EQDSK file. If not provided, uses ``(4i4)``.

    Returns
    -------
    data: Dict[str, Union[float, int, np.ndarray]]
        Dict of A-EQDSK data.
    """
    if data_format is None:
        data_format = _data_format
    if extended_sizes_format is None:
        extended_sizes_format = _extended_sizes_format
    data_reader = ff.FortranRecordReader(data_format)
    extended_sizes_reader = ff.FortranRecordReader(extended_sizes_format)

    # First line label. Date.
    header = fh.readline()

    # Second line shot number
    shot = int(fh.readline().split()[0])

    # Third line time [ms]
    time = float(fh.readline())

    # Fourth line has (up to?) 9 entries
    # time(jj),jflag(jj),lflag,limloc(jj), mco2v,mco2r,qmflag
    words = fh.readline().split()

    # Dictionary to hold result
    data = {
        "header": header,
        "shot": shot,
        "time": time,
        "jflag": int(words[1]),
        "lflag": int(words[2]),
        "limloc": words[3],  # e.g. "SNB"
        "mco2v": int(words[4]),
        "mco2r": int(words[5]),
        "qmflag": words[6],  # e.g. "CLC"
    }

    # Read first block of data
    general_block_1_values = read_array(len(_general_block_1), fh, data_reader)
    for field, value in zip(_general_block_1, general_block_1_values):
        data[field.name] = value

    # Read laser bits
    for field in _laser_block:
        values = read_array(data[field.has_length], fh, data_reader)
        data[field.name] = values

    # Read next block of data
    general_block_2_values = read_array(len(_general_block_2), fh, data_reader)
    for field, value in zip(_general_block_2, general_block_2_values):
        data[field.name] = value

    # Try reading first line of extended section. If this fails, raise a warning and
    # return data without extended portion
    try:
        extended_sizes = read_array(len(_extended_sizes), fh, extended_sizes_reader)
    except Exception:
        warnings.warn("Failed to read A-EQDSK extended section, assuming old file type")
        return data

    for field, value in zip(_extended_sizes, extended_sizes):
        data[field.name] = value

    # The first two arrays are joined together
    joined_len = sum(data[field.has_length] for field in _extended_arrays_1)
    joined = read_array(joined_len, fh, data_reader)
    data[_extended_arrays_1[0].name] = joined[: _extended_arrays_1[0].has_length]
    data[_extended_arrays_1[1].name] = joined[_extended_arrays_1[0].has_length :]

    # The next two are normal
    for field in _extended_arrays_2:
        values = read_array(data[field.has_length], fh, data_reader)
        data[field.name] = values

    # Read in another general data block
    extended_values = read_array("all", fh, data_reader)
    if len(extended_values) > len(_extended_general):
        warnings.warn(
            "Encountered variables at the end of an A-EQDSK file that are not "
            "recognised by FreeQDSK. Please consider contributing to the project "
            "and letting us know what these are!"
        )
    # Zip only includes elements up to the shortest iterable provided. data will not
    # include elments that FreeQDSK doesn't recognise, nor will it store default values
    # for elements not in the file.
    for field, value in zip(_extended_general, extended_values):
        data[field.name] = value

    return data


# Dynamic docstring generation
# TODO: Documentation does not fully describe header rows in A-EQDSK file


def _table_entry(field: Field) -> str:
    name_str = f"   * - {field.name}"
    description_str = f"     - {field.description}"
    if field.default is not None:
        default_str = f"     - {field.default}"
    else:
        if field.has_length is not None:
            default_str = f"     - ``{field.has_length} * [0.0]``"
        elif field.length_of is not None:
            default_str = f"     - ``len({field.length_of})``"
        else:
            default_str = "     -"
    return "\n".join((name_str, description_str, default_str))


_docstrings = [
    dedent(
        """\
        .. note::
           This documentation is incomplete. If you have information that could be
           added, please get in touch or a raise a pull request!

        A-EQDSK files contain a wide variety of diagnostic data. It begins with a
        specially formatted header over the first 4 lines which contains:

        - header: First row of the file
        - shot: int, The shot number
        - time: int, The time in milliseconds
        - jflag: int, 0 if error
        - iflag: int, >0 if error
        - limloc: str, IN/OUT/TOP/BOT: limiter inside/outside/top/bot SNT/SNB: single
          null top/bottom DN: double null.
        - mco2v: int, number of vertical CO2 laser chords. Should match length of arrays
          rco2v and dco2v.
        - mco2r: int, number of radial CO2 laser chords. Should match length of arrays
          rco2r and dco2r.
        - qmflag, str: axial q(0) flag, FIX if constrained and CLC for float


        This is followed by a collection of variables expressed as floats, written 4 per
        line with the Fortran format ``(4e16.9)``:

        .. list-table:: Initial block
           :widths: 20 20 60
           :header-rows: 1

           * - Name
             - Description
             - Default Value
        """
    )
]

for field in _general_block_1:
    _docstrings.append(_table_entry(field))

_docstrings.append(
    dedent(
        """\

        The next data block describes data related to CO2 lasers in the form of 4 1D
        arrays. These are similarly expressed 4-floats-per-line, with blank spaces on
        the last line if the array length is not a multiple of 4:

        .. list-table:: CO2 lasers
           :widths: 20 20 60
           :header-rows: 1

           * - Name
             - Description
             - Default Value
        """
    )
)

for field in _laser_block:
    _docstrings.append(_table_entry(field))

_docstrings.append(
    dedent(
        """\

        This is followed by another general data block:

        .. list-table:: Second general block
           :widths: 20 20 60
           :header-rows: 1

           * - Name
             - Description
             - Default Value
        """
    )
)

for field in _general_block_2:
    _docstrings.append(_table_entry(field))

_docstrings.append(
    dedent(
        """\

        The following parts of an A-EQDSK file are not present in old versions of the
        file. The next line describes the lengths of 4 further arrays using 4 ints in
        the Fortran format '(4i4)':

        .. list-table:: Extended section sizes
           :widths: 20 20 60
           :header-rows: 1

           * - Name
             - Description
             - Default Value
        """
    )
)

for field in _extended_sizes:
    _docstrings.append(_table_entry(field))

_docstrings.append(
    dedent(
        """\

        The next two arrays are stored in a concatenated fashion, so there is no newline
        between them if the length of the first array is not a multiple of four:

        .. list-table:: Extended arrays 1
           :widths: 20 20 60
           :header-rows: 1

           * - Name
             - Description
             - Default Value
        """
    )
)

for field in _extended_arrays_1:
    _docstrings.append(_table_entry(field))

_docstrings.append(
    dedent(
        """\

        The following two arrays are stored similarly to the laser data:

        .. list-table:: Extended arrays 2
           :widths: 20 20 60
           :header-rows: 1

           * - Name
             - Description
             - Default Value
        """
    )
)

for field in _extended_arrays_2:
    _docstrings.append(_table_entry(field))

_docstrings.append(
    dedent(
        """\

        The rest of the file consists of a further general data block. The total amount
        of variables depends on the exact A-EQDSK version:

        .. list-table:: Extended general block
           :widths: 20 20 60
           :header-rows: 1

           * - Name
             - Description
             - Default Value
        """
    )
)

for field in _extended_general:
    _docstrings.append(_table_entry(field))

__doc__ = "\n".join(_docstrings)
