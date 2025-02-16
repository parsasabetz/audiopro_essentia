"""
This is the setup script for the AudioPro package.

It uses setuptools to handle the packaging and distribution of the package.
The `setup()` function is called when this script is run as the main module.
"""

# imports
from setuptools import setup, find_namespace_packages


if __name__ == "__main__":
    setup(
        name="audiopro",
        version="1.3.0",
        packages=find_namespace_packages(where="src", include=["audiopro*"]),
        package_dir={"": "src"},
    )
