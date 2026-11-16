"""Compatibility import; implementation: retarget.baseline.card.math."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.card.math')
