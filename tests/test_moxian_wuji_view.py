"""Exercise the actual view update path with a small Three.js/DOM test double."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(not shutil.which('node'),reason='Node.js unavailable')
def test_wuji_view_frame_hold_and_contact_updates():
    scripts=[Path(__file__).with_name(name) for name in ('wuji_view.cjs','mano_view.cjs')]
    result=subprocess.run(['node','--test',*map(str,scripts)],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
