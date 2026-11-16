"""Compatibility import; implementation: retarget.baseline.card.initializer."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.card.initializer')
