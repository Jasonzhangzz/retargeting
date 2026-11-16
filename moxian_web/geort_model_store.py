"""Compatibility import; implementation: retarget.baseline.geort.model_store."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.geort.model_store')
