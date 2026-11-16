"""Compatibility import; implementation: retarget.baseline.card.geometry."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.card.geometry')
