"""Local asset locations replacing the examples' automatic asset downloads."""
from pathlib import Path
from types import SimpleNamespace

ASSETS = Path(__file__).resolve().parents[4] / 'data/contact/assets'
sharpa_hand = SimpleNamespace(MJCF_PATH_RIGHT=ASSETS / 'sharpa/right_hand.xml')
h1_mj_description = SimpleNamespace(MJCF_PATH=ASSETS / 'h1/h1.xml')
