"""Attempt at creating an automated molecular force field creator from QM calculations."""

# Add imports here
from .atom import *
from .molecule import *
from .collection import *
from .drivers import *

from ._version import __version__

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("I'm here")