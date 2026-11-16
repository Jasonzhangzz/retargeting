"""Compatibility import; implementation: retarget.baseline.geort.workspace."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.geort.workspace')
