#!/usr/bin/env python3
"""Validate actual DexForge native 1/30-step gradients before full FA execution."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/humanoid_move_largebox_forceaware.yaml")
    parser.add_argument("--output", type=Path, default=ROOT/"output/dexforge_gradient_validation")
    args = parser.parse_args()
    from retarget.config.run import load
    from retarget.env.humanoid.dexforge_runner import validate_only

    report = validate_only(load(args.config), args.output)
    print(args.output.resolve()/"gradient_validation.json", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
