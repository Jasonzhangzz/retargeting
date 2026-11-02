#!/usr/bin/env python3
"""Evaluate one saved final grasp with the common native GraspADMM protocol."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from retarget.env.grasp.grasp_evaluation import evaluate_grasp, select_grasp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="Saved method case directory")
    parser.add_argument("--model", type=Path, help="Source ordinary-mesh MuJoCo scene")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--output", type=Path, required=True, help="New evaluation directory; source is untouched")
    parser.add_argument("--timeout", type=float, default=600.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    model = args.model or next((args.case/name for name in ("scene.xml", "model.xml") if (args.case/name).is_file()), None)
    if model is None:
        parser.error("No scene found; provide --model")
    qpos, selection = select_grasp(args.case, args.method, args.trajectory)
    report = evaluate_grasp(model, qpos, args.output, method=args.method, selection=selection, timeout=args.timeout)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
