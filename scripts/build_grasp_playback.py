#!/usr/bin/env python3
"""Generate a cached, unassisted gravity playback for a saved grasp."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retarget.grasp_playback import simulate_playback

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=3.)
    parser.add_argument('--saved-velocity', action='store_true')
    args = parser.parse_args()
    path, report = simulate_playback(args.case, duration=args.duration, saved_velocity=args.saved_velocity)
    print(json.dumps(dict(trajectory=str(path), report=report)), flush=True)
