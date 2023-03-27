# FreeQDSK

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://img.shields.io/badge/license-MIT-blue.svg)
[![py3comp](https://img.shields.io/badge/py3-compatible-brightgreen.svg)](https://img.shields.io/badge/py3-compatible-brightgreen.svg)

Read and write the popular "geqdsk" and "aeqdsk" tokamak equilibrium
file formats.

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

A GEQDSK file may be read using the `geqdsk.read` function:

```python
from freeqdsk import geqdsk
with open(filename, "r") as f:
    data = geqdsk.read(f)
```

The result is a dict containing data from the GEQDSK file. To write a file:

```python
with open(filename, "w") as f:
    geqdsk.write(data, f)
```

Similarly, for AEQDSK files:

```python
from freeqdsk import aeqdsk

with open(filename, "r") as f:
    data = aeqdsk.read(f)

with open(filename, "w") as f:
    aeqdsk.write(data, f)
```
