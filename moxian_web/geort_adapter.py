"""Compatibility import; implementation: retarget.baseline.geort.adapter."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.geort.adapter')
