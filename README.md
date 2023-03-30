# FreeQDSK

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://img.shields.io/badge/license-MIT-blue.svg)
[![py3comp](https://img.shields.io/badge/py3-compatible-brightgreen.svg)](https://img.shields.io/badge/py3-compatible-brightgreen.svg)

Read and write G-EQDSK, A-EQDSK, and P-EQDSK file formats, which are used to describe
the tokamak fusion devices.

## Installation

To install, you may need to update `pip` to the latest version:

```bash
$ python3 -m pip install --upgrade pip
```

You can then install using:

```bash
$ python3 -m pip install .
```

To run tests:

```bash
$ python3 -m install .[tests]
$ pytest -v
```

To build the docs:

```bash
$ python3 -m install .[docs]
$ cd docs
$ make html
```

## Usage

A G-EQDSK file may be read using the `geqdsk.read` function:

```python
from freeqdsk import geqdsk

with open(filename, "r") as f:
    data = geqdsk.read(f)
```

The result is a dict containing data from the G-EQDSK file. To write a file:

```python
with open(filename, "w") as f:
    geqdsk.write(data, f)
```

Similarly, for A-EQDSK files:

```python
from freeqdsk import aeqdsk

with open(filename, "r") as f:
    data = aeqdsk.read(f)

with open(filename, "w") as f:
    aeqdsk.write(data, f)
```

P-EQDSK files are read into a [namedtuple][namedtuple] containing the fields `profiles`,
`species`, and `units`. Both `profiles` and `species` are
[Pandas DataFrames](dataframe), while `units` is a dict containing units for the columns
in `profiles`.

```python
from freeqdsk import peqdsk

with open(filename, "r") as f:
    profiles, species, units = peqdsk.read(f)

with open(filename, "w") as f:
    peqdsk.write(profiles, species, units, f)

# Alternative syntax:

with open(filename, "r") as f:
    data = peqdsk.read(f)

with open(filename, "w") as f:
    peqdsk.write(*data, f)
```

[namedtuple]: https://docs.python.org/3/library/collections.html#collections.namedtuple
[dataframe]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
