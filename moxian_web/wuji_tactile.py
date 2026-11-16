"""Compatibility import; implementation: retarget.baseline.card.tactile."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.card.tactile')
