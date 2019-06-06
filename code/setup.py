#!/usr/bin/env python
import os
from setuptools import setup, find_packages
import sys

NAME = 'centuri_project'
DESCRIPTION = 'spike detection'
LICENSE = 'GNU General Public License v3 (GPLv3)'
INSTALL_REQUIRES = ['numpy', 'daiquiri', 'matplotlib', 'pandas', 'keras', 'docopt', 'scipy']  # TODO to be completed

PYTHON_REQUIRES = '>=3.5'

###############################################################################
if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'\n")

if sys.version_info[:2] < (3, 5):
    errmsg = '{} requires Python 3.5 or later ({[0]:d}.{[1]:d} detected).'
    print(errmsg.format(NAME, sys.version_info[:2]))
    sys.exit(-1)


def setup_package():
    """Setup function"""


    setup(name=NAME,
          version=0.1,
          description=DESCRIPTION,
          license=LICENSE,
          packages=find_packages(),
          install_requires=INSTALL_REQUIRES,
          python_requires=PYTHON_REQUIRES,
          )


if __name__ == "__main__":
    setup_package()
