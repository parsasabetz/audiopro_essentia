"""
This is the setup script for the AudioPro package.

It uses setuptools to handle the packaging and distribution of the package.
The `setup()` function is called when this script is run as the main module.
"""

# imports
from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="audiopro",
        version="0.13.0",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
    )
