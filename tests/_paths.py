"""Stable paths shared by all test groups."""
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
ROOT = TEST_ROOT.parent
FIXTURES = TEST_ROOT / "fixtures"
