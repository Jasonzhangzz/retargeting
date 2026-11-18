#!/usr/bin/env python3
"""Run the unchanged remote EMF H5/NPZ CLI from this checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.baseline.wuji_retargeting.reference import EmfRetargetPipeline  # noqa: F401,E402
from emf_retarget.cli import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
