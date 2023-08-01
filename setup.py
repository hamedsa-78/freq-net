"""
This is setup.py file for installing packages of this repo

use [pipreqs .] from command line for requirements.txt file generation

use [pip install .] from command line for installation and do not execute the script directly

"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="freq_net",
    version="0.0.1",
    install_requires=required_packages,
    packages=find_packages(),
)
