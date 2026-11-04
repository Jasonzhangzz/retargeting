#!/usr/bin/env python3
"""Show final grasp results in viser: python visualize.py [--batch NAME]."""

import argparse
import os
from pathlib import Path

from retarget.config.run import ROOT, load
from retarget.viewer.runtime import web


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/web_graspadmm.yaml")
    parser.add_argument("--batch", help="Saved result directory name under output/")
    parser.add_argument("--port", type=int)
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind address (default: local only)")
    args = parser.parse_args()
    config = load(args.config)
    config.visualization["final_only"] = True
    config.visualization["host"] = args.host
    if args.batch:
        config.visualization["batch"] = args.batch
    if args.port is not None:
        if not 0 < args.port < 65536:
            parser.error("port must be between 1 and 65535")
        config.visualization["port"] = args.port
    batch = config.visualization.get("batch")
    if batch:
        if Path(batch).name != batch:
            parser.error('batch must be a directory name, not a path')
        active = ROOT / 'output' / batch
        if not (active/'manifest.json').is_file():
            cached = ROOT/'results'/batch
            if (cached/'manifest.json').is_file() and not active.exists() and not active.is_symlink():
                active.parent.mkdir(parents=True,exist_ok=True)
                active.symlink_to(os.path.relpath(cached,active.parent),target_is_directory=True)
            else:
                parser.error(f"Missing result manifest for batch: {batch}")
    web(config)


if __name__ == "__main__":
    main()
