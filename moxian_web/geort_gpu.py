"""Compatibility import; implementation: retarget.baseline.geort.gpu."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.geort.gpu')
