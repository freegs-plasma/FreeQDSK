"""
Low level routines for reading and writing G-EQDSK files

SPDX-FileCopyrightText: © 2016 Ben Dudson, University of York.

SPDX-License-Identifier: MIT

"""
from __future__ import annotations  # noqa

from datetime import date
from typing import Dict, Optional, TextIO, Union

import numpy as np
from numpy.typing import ArrayLike

from ._fileutils import f2s, ChunkOutput, write_1d, write_2d, next_value


def write(
    data: Dict[str, Union[int, float, ArrayLike]],
    fh: TextIO,
    label: Optional[str] = None,
    shot: Optional[int] = None,
    time: Optional[int] = None,
) -> None:
    r"""
    Write a G-EQDSK equilibrium file, given a dictionary of data.

    Parameters
    ----------
    data: Dict[str, Union[int, float, ArrayLike]]
        G-EQDSK data to write. See the 'Notes' section below for info.
    fh: TextIO
        File handle to write to. Should be opened in a text write mode, i.e.
        ``open(filename, "w")``.
    label: Optional[str], default None
        Text label to put in the file. Defaults to 'FREEGS' if not provided.
    shot: Optional[int], default None
        Shot number to put in the file. Defaults to 0 if not provided.
    time: Optional[int], default None
        Time in milliseconds to put in the file. Defaults to 0 if not provided.

    Notes
    -----

    The data dictionary should contain the following:

    ======= ========================================================================
    nx      Number of points in the R direction, int
    ny      Number of points in the Z direction, int
    rcentr  Reference value of R, float [meter]
    bcentr  Vacuum toroidal magnetic field at rcentr, float [tesla]
    rleft   R at left (inner) boundary, float [meter]
    zmid    Z at middle of domain, float [meter]
    rmagx   R at magnetic axis (0-point), float [meter]
    zmagx   Z at magnetic axis (0-point), float [meter]
    simagx  Poloidal flux :math:`\psi` at magnetic axis, float [weber / radian]
    sibdry  Poloidal flux :math:`\psi` at plasma boundary, float [weber / radian]
    cpasma  Plasma current, float [ampere]
    fpol    Poloidal current function :math:`F(\psi)=RB_t`, 1D array [meter * tesla]
    pres    Plasma pressure :math:`p(\psi)`, 1D array [pascal]
    qpsi    Safety factor :math:`q(\psi)`, 1D array [dimensionless]
    psi     Poloidal flux :math:`\psi`, 2D array [weber / radian]
    ======= ========================================================================

    The 1D arrays should be the same length, and are defined on a linear :math:`\psi`
    grid given by ``np.linspace(simagx, sibdry, nx)``. The following additional data
    may also be included:

    ======= ========================================================================
    ffprime :math:`FF'(\psi)=RB_t`, 1D array [meter**2 * tesla**2 * radian / weber]
    pprime  :math:`p'(\psi)`, 1D array [pascal * radian / weber]
    rbdry   R of boundary points, 1D array [meter]
    zbdry   Z of boundary points, 1D array [meter]
    rlim    R of limiter point, 1D array [meter]
    zlim    Z of limiter point, 1D array [meter]
    ======= ========================================================================

    The arrays ``ffprime`` and ``pprime`` are defined on the same 1D grid as in the
    previous table. ``rbdry`` and ``zbdry`` may be a different length, but must be the
    same length as each other. Similarly, ``rlim`` and ``zlim`` must have the same
    length as each other.
    """

    nx = data["nx"]
    ny = data["ny"]

    if not label:
        label = "FREEGS"
    if len(label) > 11:
        label = label[0:12]
        print("WARNING: label too long, it will be shortened to {}".format(label))

    creation_date = date.today().strftime("%d/%m/%Y")

    if not shot:
        shot = 0

    if isinstance(shot, int):
        shot = "# {:d}".format(shot)

    if not time:
        time = 0

    if isinstance(time, int):
        time = "  {:d}ms".format(time)

    # I have no idea what idum is, here it is set to 3
    idum = 3
    header = "{0:11s}{1:10s}   {2:>8s}{3:16s}{4:4d}{5:4d}{6:4d}\n".format(
        label, creation_date, shot, time, idum, nx, ny
    )

    # First line: Identification string, followed by resolution
    fh.write(header)

    # Second line
    fh.write(
        f2s(data["rdim"])
        + f2s(data["zdim"])
        + f2s(data["rcentr"])
        + f2s(data["rleft"])
        + f2s(data["zmid"])
        + "\n"
    )

    # Third line
    fh.write(
        f2s(data["rmagx"])
        + f2s(data["zmagx"])
        + f2s(data["simagx"])
        + f2s(data["sibdry"])
        + f2s(data["bcentr"])
        + "\n"
    )

    # 4th line
    fh.write(
        f2s(data["cpasma"])
        + f2s(data["simagx"])
        + f2s(0.0)
        + f2s(data["rmagx"])
        + f2s(0.0)
        + "\n"
    )

    # 5th line
    fh.write(
        f2s(data["zmagx"]) + f2s(0.0) + f2s(data["sibdry"]) + f2s(0.0) + f2s(0.0) + "\n"
    )

    # SCENE has actual ff' and p' data so can use that
    # fill arrays
    # Lukas Kripner (16/10/2018): uncommenting this, since you left there
    # check for data existence bellow. This seems to as safer variant.
    workk = np.zeros([nx])

    # Write arrays
    co = ChunkOutput(fh)

    write_1d(data["fpol"], co)
    write_1d(data["pres"], co)
    if "ffprime" in data:
        write_1d(data["ffprime"], co)
    else:
        write_1d(workk, co)
    if "pprime" in data:
        write_1d(data["pprime"], co)
    else:
        write_1d(workk, co)

    write_2d(data["psi"], co)
    write_1d(data["qpsi"], co)

    # Boundary / limiters
    nbdry = 0
    nlim = 0
    if "rbdry" in data:
        nbdry = len(data["rbdry"])
    if "rlim" in data:
        nlim = len(data["rlim"])

    co.newline()
    fh.write("{0:5d}{1:5d}\n".format(nbdry, nlim))

    if nbdry > 0:
        for r, z in zip(data["rbdry"], data["zbdry"]):
            co.write(r)
            co.write(z)
        co.newline()

    if nlim > 0:
        for r, z in zip(data["rlim"], data["zlim"]):
            co.write(r)
            co.write(z)
        co.newline()


def read(fh: TextIO, cocos: int = 1) -> Dict[str, Union[int, float, np.ndarray]]:
    r"""
    Read a G-EQDSK formatted equilibrium file.
    The format is specified `here <https://fusion.gat.com/theory/Efitgeqdsk>`_.

    Parameters
    ----------

    fh: TextIO
        File handle to write to. Should be opened in a text read mode, i.e.
        ``open(filename, "r")``.
    cocos: int, default 1
        COordinate COnventionS. This feature is not fully handled yet, and only
        determines whether psi is divided by :math:`2\pi` or not. If ``cocos >= 10``,
        :math:`\psi` is divided by :math:`2\pi`, and otherwise it is left unchanged.
        See `Sauter et al, 2013 <https://doi.org/10.1016/j.cpc.2012.09.010>`_.

    Returns
    -------
    Dict[str, Union[int, float, np.ndarray]]
        See the Notes section for details.

    Notes
    -----

    The resulting dictionary contains the following:

    ======= ========================================================================
    nx      Number of points in the R direction, int
    ny      Number of points in the Z direction, int
    rcentr  Reference value of R, float [meter]
    bcentr  Vacuum toroidal magnetic field at rcentr, float [tesla]
    rleft   R at left (inner) boundary, float [meter]
    zmid    Z at middle of domain, float [meter]
    rmagx   R at magnetic axis (0-point), float [meter]
    zmagx   Z at magnetic axis (0-point), float [meter]
    simagx  Poloidal flux :math:`\psi` at magnetic axis, float [weber / radian]
    sibdry  Poloidal flux :math:`\psi` at plasma boundary, float [weber / radian]
    cpasma  Plasma current, float [ampere]
    fpol    Poloidal current function :math:`F(\psi)=RB_t`, 1D array [meter * tesla]
    pres    Plasma pressure :math:`p(\psi)`, 1D array [pascal]
    qpsi    Safety factor :math:`q(\psi)`, 1D array [dimensionless]
    psi     Poloidal flux :math:`\psi`, 2D array [weber / radian]
    ffprime :math:`FF'(\psi)=RB_t`, 1D array [meter**2 * tesla**2 * radian / weber]
    pprime  :math:`p'(\psi)`, 1D array [pascal * radian / weber]
    rbdry   R of boundary points, 1D array [meter]
    zbdry   Z of boundary points, 1D array [meter]
    rlim    R of limiter point, 1D array [meter]
    zlim    Z of limiter point, 1D array [meter]
    ======= ========================================================================
    """

    # Read the first line
    header = fh.readline()
    words = header.split()  # Split on whitespace
    if len(words) < 3:
        raise ValueError("Expecting at least 3 numbers on first line")

    nx = int(words[-2])
    ny = int(words[-1])

    print("  nx = {0}, ny = {1}".format(nx, ny))

    # Dictionary to hold result
    data = {"nx": nx, "ny": ny}

    # List of fields to read. None means discard value
    fields = [
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
    ]

    values = next_value(fh)

    for f in fields:
        val = next(values)
        if f:
            data[f] = val

    # Read arrays

    def read_1d(n):
        """
        Read a 1D array of length n
        """
        val = np.zeros(n)
        for i in range(n):
            val[i] = next(values)
        return val

    def read_2d(n, m):
        """
        Read a 2D (n,m) array in Fortran order
        """
        val = np.zeros([n, m])
        for y in range(m):
            for x in range(n):
                val[x, y] = next(values)
        return val

    data["fpol"] = read_1d(nx)
    data["pres"] = read_1d(nx)
    data["ffprime"] = read_1d(nx)
    data["pprime"] = read_1d(nx)

    data["psi"] = read_2d(nx, ny)

    data["qpsi"] = read_1d(nx)

    # Ensure that psi is divided by 2pi
    if cocos > 10:
        for var in ["psi", "simagx", "sibdry"]:
            data[var] /= 2 * np.pi

    nbdry = next(values)
    nlim = next(values)

    # print(nbdry, nlim)

    if nbdry > 0:
        # Read (R,Z) pairs
        print(nbdry)
        data["rbdry"] = np.zeros(nbdry)
        data["zbdry"] = np.zeros(nbdry)
        for i in range(nbdry):
            data["rbdry"][i] = next(values)
            data["zbdry"][i] = next(values)

    if nlim > 0:
        # Read (R,Z) pairs
        data["rlim"] = np.zeros(nlim)
        data["zlim"] = np.zeros(nlim)
        for i in range(nlim):
            data["rlim"][i] = next(values)
            data["zlim"][i] = next(values)

    return data
