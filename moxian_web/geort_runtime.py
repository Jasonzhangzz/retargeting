"""Compatibility import; implementation: retarget.baseline.geort.runtime."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.geort.runtime')
