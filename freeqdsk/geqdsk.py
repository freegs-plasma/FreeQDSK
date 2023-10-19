r"""
G-EQDSK files describe Grad-Shafranov tokamak plasma equilibria. These may be generated
via a Grad-Shafranov solver, or by fitting to experimental measurements. Tools that
make use of G-EQDSK files include, but are not limited to:

- EFIT
- FreeGS
- SCENE
- TRANSP

G-EQDSK files begin with a header line containing the following information:

- ``comment`` A comment 48 characters in length. This normally includes information such as the
  software used to generate the file, the date of creation, a shot number, and the
  time frame within the shot. Unfortunately, the format of this line is not
  rigorously defined, so each code will tend to define it differently. FreeQDSK
  currently assumes a FreeGS-style comment, but this may be expanded to other
  comment styles in a future update.
- ``int`` A mysterious integer of unknown purpose, with a width of 4 characters.
- ``nx`` The number of points in the R direction, expressed as an integer with a width of
  four characters.
- ``ny`` The number of points in the Z direction, expressed as an integer with a width of
  four characters.


The Fortran format for the header can be expressed as ``(a48,3i4)``. This is followed
by 4 lines of floats describing a tokamak plasma equilibrium. Each line contains 5
floats, following the Fortran format ``(5e16.9)``. These floats are:

====== ====== ====== ====== ======
rdim   zdim   rcentr rleft  zmid
rmagx  zmagx  simagx sibdry bcentr
cpasma simagx        rmagx
zmagx         sibdry
====== ====== ====== ====== ======

The blank spaces are ignored, and are usually set to zero. Note that ``rmagx``,
``zmagx``, ``simagx``, and ``sibdry`` are duplicated. The meaning of these
floats are:

======= ========================================================================
rdim    Width of computational domain in R direction, float [meter]
zdim    Height of computational domain in Z direction, float [meter]
rcentr  Reference value of R, float [meter]
rleft   R at left (inner) boundary, float [meter]
zmid    Z at middle of domain, float [meter]
rmagx   R at magnetic axis (0-point), float [meter]
zmagx   Z at magnetic axis (0-point), float [meter]
simagx  Poloidal flux :math:`\psi` at magnetic axis, float [weber / radian]
sibdry  Poloidal flux :math:`\psi` at plasma boundary, float [weber / radian]
bcentr  Vacuum toroidal magnetic field at rcentr, float [tesla]
cpasma  Plasma current, float [ampere]
======= ========================================================================

This is then followed by a series of grids:

======= ========================================================================
fpol    Poloidal current function :math:`F(\psi)=RB_t`, 1D array [meter * tesla]
pres    Plasma pressure :math:`p(\psi)`, 1D array [pascal]
ffprime :math:`FF'(\psi)`, 1D array [meter**2 * tesla**2 * radian / weber]
pprime  :math:`p'(\psi)`, 1D array [pascal * radian / weber]
psi     Poloidal flux :math:`\psi`, 2D array [weber / radian]
qpsi    Safety factor :math:`q(\psi)`, 1D array [dimensionless]
======= ========================================================================

The 1D arrays are expressed on a linearly spaced :math:`\psi` grid which may be
generated using ``numpy.linspace(simagx, sibdry, nx)``. The 2D :math:`\psi` grid is
instead expressed on a linearly spaced  grid extending the range
``[rleft, rleft + rdim]`` in the R direction and ``[zmid - zdim/2, zmid + zdim/2]``
in the Z direction. Each grid is printed over multiple lines using the Fortran
format ``(5e16.9)``, with the final line containing some blank spaces if the total
grid size is not a multiple of 5. Note that the ``psi`` grid is expressed in a
flattened state using Fortran ordering, meaning it increments in the columns
direction first, then in rows.

The G-EQDSK file then gives information on the plasma boundary and the surrounding
limiter contour. The next line gives the dimensions of these grids in the format
``(2i5)``:

======= ========================================================================
nbdry   Number of points in the boundary grid, int
nlim    Number of points in the limiter grid, int
======= ========================================================================

Finally, the boundary and limiter grids are specified as lists of ``(R, Z)``
coordinates, again using the format ``(5e16.9)``:

======= ========================================================================
rbdry   R of boundary points, 1D array [meter]
zbdry   Z of boundary points, 1D array [meter]
rlim    R of limiter points, 1D array [meter]
zlim    Z of limiter points, 1D array [meter]
======= ========================================================================

Note that these grids are interleaved, so are expressed as:

============ ============ ============ ============ ============
``rbdry[0]`` ``zbdry[0]`` ``rbdry[1]`` ``zbdry[1]`` ``rbdry[2]``
``zbdry[2]`` ``rbdry[3]`` ``zbdry[3]`` ``rbdry[4]`` ``rbdry[4]``
...          ...          ...          ...          ...
============ ============ ============ ============ ============

++++++++++++++++++++++++++++++++
A note on coordinate conventions
++++++++++++++++++++++++++++++++

Various conventions regarding the orientation of the toroidal angle
:math:`\varphi` and the poloidal angle :math:`\theta` are in use.  For example,
if looking from the top :math:`\nabla\varphi = \frac{1}{R}\hat e_\varphi` can
either point clockwise or counter-clockwise depending on whether
:math:`(R,Z,\varphi)` or :math:`(R,\varphi,Z)` is used as cylindrical
coordinates.  This observations led to the formal definition of the COCOS
conventions in
`O. Sauter and S. Y Medvedev, "Tokamak Coordinate Conventions: COCOS", Comput.
Phys. Commun. 184 (2013) 293
<https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf>`_
From the paper the magnetic field can be generally expressed as

.. math::
   \vec B = F(\psi) \nabla \varphi + \sigma_{B_p} \frac{1}{(2\pi)^{e_{B_p}}}
   \nabla\varphi \times \nabla\psi

where :math:`\sigma_{B_P} = \pm 1` and :math:`e_{B_p} \in \{ 0,1\}` depends on
the convention in use.  The orientation of the poloidal angle :math:`\theta`
largely affects the sign of :math:`q` (if :math:`(\rho,\theta,\varphi)` is
right-handed then q is positive for right-handed field winding while if
:math:`(\rho,\theta,\varphi)` is left-handed then q is positive for left-handed
winding).  However, some tools define q as always positive so the paper warns
against using it as a consistency check.

Note the table of conventions in https://crppwww.epfl.ch/~sauter/cocos/

SPDX-FileCopyrightText: © 2016 Ben Dudson, University of York.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations  # noqa

import warnings
from datetime import date
from typing import Dict, Optional, TextIO, Union

import numpy as np
from numpy.typing import ArrayLike

from ._fileutils import read_array, read_line, write_array, write_line


#: Default header contains comment, unknown int, nx, ny
_header_fmt = "(a48,3i4)"

#: Default format for all float data
_data_fmt = "(5e16.9)"

#: Default format for line describing length of boundary and limiter arrays
_bdry_lim_fmt = "(2i5)"

#: The labels assigned to floats on the first 4 lines after the header.
#: 'None' indicates a blank space, which should be filled with 0.0.
_float_keys = (
    "rdim",
    "zdim",
    "rcentr",
    "rleft",
    "zmid",
    "rmagx",
    "zmagx",
    "simagx",
    "sibdry",
    "bcentr",
    "cpasma",
    "simagx",
    None,
    "rmagx",
    None,
    "zmagx",
    None,
    "sibdry",
    None,
    None,
)


def write(
    data: Dict[str, Union[int, float, ArrayLike]],
    fh: TextIO,
    label: Optional[str] = None,
    shot: int = 0,
    time: int = 0,
    header_fmt: Optional[str] = None,
    data_fmt: Optional[str] = None,
    bdry_lim_fmt: Optional[str] = None,
) -> None:
    r"""
    Write a G-EQDSK equilibrium file, given a dictionary of data.

    Parameters
    ----------
    data:
        G-EQDSK data to write. See the 'Notes' section below for info.
    fh:
        File handle to write to. Should be opened in a text write mode, i.e.
        ``open(filename, "w")``.
    label:
        Text label to put in the file. Defaults to 'FREEGS' if not provided.
    shot:
        Shot number to put in the file.
    time:
        Time in milliseconds to put in the file.
    header_fmt:
        Fortran IO format for G-EQDSK header line. If not provided, uses ``(a48,3i4)``,
        corresponding to a comment, a dummy int, nx, and ny.
    data_fmt:
        Fortran IO format for G-EQDSK data. If not provided, uses ``(5e16.9)``.
    bdry_lim_fmt:
        Fortran IO format specifying the lengths of the boundary/limiter grids. If
        not provided, defaults to ``(2i5)``.

    Raises
    ------
    KeyError
        If required information is missing from ``data``.
    ValueError
        If the provided arrays have incorrect dimensions, or if the data provided is
        of the wrong type.

    Notes
    -----
    ``ffprime`` and ``pprime`` may be excluded from ``data``, in which case they will
    be filled with zeros. Similarly, ``rbdry``, ``zbdry``, ``rlim`` and ``zlim`` may
    be excluded, in which case they aren't added to the end of the file. If ``nbdry``
    and ``nlim`` are excluded, they are inferred from ``rbdry`` and ``rlim``, and are
    set to zero if these aren't found. Similarly, if ``nx`` or ``ny`` are excluded, they
    are inferred from ``psi``, which should have the shape ``(nx, ny)``.
    """
    if header_fmt is None:
        header_fmt = _header_fmt
    if data_fmt is None:
        data_fmt = _data_fmt
    if bdry_lim_fmt is None:
        bdry_lim_fmt = _bdry_lim_fmt

    # Get dimensions and check data is correct
    nx = data.get("nx", np.shape(data["psi"])[0])
    ny = data.get("ny", np.shape(data["psi"])[1])
    nbdry = data.get("nbdry", len(data.get("rbdry", [])))
    nlim = data.get("nlim", len(data.get("rlim", [])))
    for grid in ("fpol", "pres", "ffprime", "pprime", "qpsi"):
        if grid not in data:
            if grid in ("ffprime", "pprime"):
                continue
            raise ValueError(f"Grid {grid} not in data")
        if np.shape(data[grid]) != (nx,):
            raise ValueError(f"Grid {grid} should have shape {(nx,)}")
    if "psi" not in data:
        raise ValueError("Grid psi not in data")
    if np.shape(data["psi"]) != (nx, ny):
        raise ValueError(f"Grid psi should have shape {(nx, ny)}")
    if nbdry > 0:
        if np.shape(data["rbdry"]) != (nbdry,):
            raise ValueError("rbdry should have length nbdry")
        if np.shape(data["rbdry"]) != np.shape(data["zbdry"]):
            raise ValueError("rbdry and zbdry should have the same length")
    if nlim > 0:
        if np.shape(data["rlim"]) != (nlim,):
            raise ValueError("rlim should have length nlim")
        if np.shape(data["rlim"]) != np.shape(data["zlim"]):
            raise ValueError("rlim and zlim should have the same length")

    # Write header
    # TODO There is no rigorous standard for GEQDSK headers. As FreeQDSK is derived from
    #      FreeGS, we currently use the same header specification. This should be
    #      generalised for other codes that use G-EQDSK, including but not limited to:
    #      - EFIT
    #      - SCENE
    #      - TRANSP
    #      - eqtools
    if label is None:
        label = "FREEGS"
    if len(label) > 11:
        warnings.warn(f"Label {label} too long, it will be shortened to {label[:11]}")
        label = label[:11]
    creation_date = date.today().strftime("%d/%m/%Y")
    shot_str = f"# {shot:d}"
    time_str = f"  {time:d}ms"
    comment = f"{label:11}{creation_date:10s}   {shot_str:>8s}{time_str:16s}"

    idum = 3  # No idea what the third-to-last int is, here it is set to 3
    write_line((comment, idum, nx, ny), fh, header_fmt)

    # The next four lines contain floats in the order specified by _float_keys
    # If an entry in _float_keys is None, that float is a dummy value and is set to 0.0
    floats = [(0.0 if k is None else data[k]) for k in _float_keys]
    write_array(floats, fh, data_fmt)

    # Write each grid
    write_array(data["fpol"], fh, data_fmt)
    write_array(data["pres"], fh, data_fmt)
    write_array(data.get("ffprime", np.zeros(nx)), fh, data_fmt)
    write_array(data.get("pprime", np.zeros(nx)), fh, data_fmt)
    write_array(data["psi"], fh, data_fmt)
    write_array(data["qpsi"], fh, data_fmt)

    # Boundary / limiters
    write_array((nbdry, nlim), fh, bdry_lim_fmt)
    if nbdry > 0:
        bdry = np.empty(2 * nbdry)
        bdry[0::2], bdry[1::2] = data["rbdry"], data["zbdry"]
        write_array(bdry, fh, data_fmt)
    if nlim > 0:
        lim = np.empty(2 * nlim)
        lim[0::2], lim[1::2] = data["rlim"], data["zlim"]
        write_array(lim, fh, data_fmt)


def read(
    fh: TextIO,
    cocos: int = 1,
    header_fmt: Optional[str] = None,
    data_fmt: Optional[str] = None,
    bdry_lim_fmt: Optional[str] = None,
) -> Dict[str, Union[int, float, np.ndarray]]:
    r"""
    Read a G-EQDSK formatted equilibrium file.
    The format is specified `here <https://fusion.gat.com/theory/Efitgeqdsk>`_.

    Parameters
    ----------

    fh:
        File handle to read from. Should be opened in a text read mode, i.e.
        ``open(filename, "r")``.
    cocos:
        COordinate COnventionS. This feature is not fully handled yet, and only
        determines whether psi is divided by :math:`2\pi` or not. If ``cocos >= 10``,
        :math:`\psi` is divided by :math:`2\pi`, and otherwise it is left unchanged.
        See `Sauter et al, 2013 <https://doi.org/10.1016/j.cpc.2012.09.010>`_.
    header_fmt:
        Fortran IO format for G-EQDSK header line. If not provided, uses ``(a48,3i4)``,
        corresponding to a comment, a dummy int, nx, and ny.
    data_fmt:
        Fortran IO format for G-EQDSK data. If not provided, uses ``(5e16.9)``.
    bdr_lim_fmt:
        Fortran IO format specifying the lengths of the boundary/limiter grids. If
        not provided, defaults to ``(2i5)``

    Warns
    -----
    UserWarning
        If any of the entries that are duplicated in the file do not match. These
        include 'rmagx', 'zmagx', 'simagx' and 'sibdry'. The value returned will be
        the last found.
    """
    if header_fmt is None:
        header_fmt = _header_fmt
    if data_fmt is None:
        data_fmt = _data_fmt
    if bdry_lim_fmt is None:
        bdry_lim_fmt = _bdry_lim_fmt

    # TODO Should try to extract shot/time data from header comment
    # MW: at least give it to a human to read
    comment, integer, nx, ny = read_line(fh, header_fmt)

    # Dictionary to hold result
    data = {"comment": comment, "int": integer, "nx": nx, "ny": ny}

    # Read first four lines
    floats = read_array(20, fh, data_fmt)
    for key, value in zip(_float_keys, floats):
        # Skip dummy values
        if key is None:
            continue
        # Check duplicates are actually duplicated. Warn otherwise.
        if key in data:
            if data[key] != value:
                warnings.warn(
                    f"The value of '{key}' should be duplicated. "
                    f"Found values {value} and {data[key]}"
                )
        data[key] = value

    # Read grids
    data["fpol"] = read_array(nx, fh, data_fmt)
    data["pres"] = read_array(nx, fh, data_fmt)
    data["ffprime"] = read_array(nx, fh, data_fmt)
    data["pprime"] = read_array(nx, fh, data_fmt)
    data["psi"] = read_array((nx, ny), fh, data_fmt)
    data["qpsi"] = read_array(nx, fh, data_fmt)

    # Ensure that psi is divided by 2pi
    if cocos > 10:
        # FIXME Should this not also affect ffprime and pprime?
        for var in ["psi", "simagx", "sibdry"]:
            data[var] /= 2 * np.pi

    # Get boundary/limiter dimensions
    nbdry, nlim = read_array(2, fh, bdry_lim_fmt)
    data["nbdry"], data["nlim"] = nbdry, nlim

    if nbdry > 0:
        bdry = read_array(2 * nbdry, fh, data_fmt)
        data["rbdry"], data["zbdry"] = bdry[0::2], bdry[1::2]

    if nlim > 0:
        lim = read_array(2 * nlim, fh, data_fmt)
        data["rlim"], data["zlim"] = lim[0::2], lim[1::2]

    return data
