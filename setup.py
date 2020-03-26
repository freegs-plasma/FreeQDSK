from setuptools import setup
import os

with open("README.md", "r") as f:
    long_description = f.read()

version_dict = {}
with open("freeqdsk/_version.py") as f:
    exec(f.read(), version_dict)

name = "freeqdsk"
version = version_dict["__version__"]
release = version

setup(
    name=name,
    version=version,
    packages=[name],
    license="MIT",
    author="Ben Dudson",
    author_email="benjamin.dudson@york.ac.uk",
    # url="https://github.com/bendudson/freegs",
    description="GEQDSK and AEQDSK tokamak equilibrium file reader/writer",
    long_description=long_description,
    install_requires=["numpy>=1.8"],
    platforms="any",
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
