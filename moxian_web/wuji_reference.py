"""Compatibility import; implementation: retarget.baseline.wuji_retargeting.reference."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('retarget.baseline.wuji_retargeting.reference')
