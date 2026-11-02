#!/usr/bin/env python3
"""Show final grasp results in viser: python visualize.py [--batch NAME]."""

import argparse
from pathlib import Path

from retarget.config.run import ROOT, load
from retarget.viewer.runtime import web


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/web_grasp_template_mppi.yaml")
    parser.add_argument("--batch", help="Saved result directory name under output/")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = load(args.config)
    config.visualization["final_only"] = True
    if args.batch:
        config.visualization["batch"] = args.batch
    if args.port is not None:
        if not 0 < args.port < 65536:
            parser.error("port must be between 1 and 65535")
        config.visualization["port"] = args.port
    batch = config.visualization.get("batch")
    if batch and not (ROOT / "output" / batch / "manifest.json").is_file():
        parser.error(f"Missing result manifest for batch: {batch}")
    web(config)


if __name__ == "__main__":
    main()
