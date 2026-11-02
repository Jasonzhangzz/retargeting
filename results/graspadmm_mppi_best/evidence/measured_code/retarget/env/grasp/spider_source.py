"""Pinned SPIDER provenance for the dexterous sampling adapter."""

import hashlib
import json

from retarget.config.run import ROOT


def verify_sources():
    root = ROOT/"third_party/SPIDER/dex"
    manifest = json.loads((root/"UPSTREAM.json").read_text())
    if manifest["commit"] != "44717007de41cbef7565dff7ff9f4453557a2d3d":
        raise ValueError("Unexpected SPIDER source revision")
    for relative, expected in manifest["sha256"].items():
        path = root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"SPIDER source verification failed: {relative}")
    return dict(repository=manifest["repository"], commit=manifest["commit"],
        license=manifest["license"], verified_files=len(manifest["sha256"]),
        cost="original one-hand combined weighted configuration L2 + velocity L2 + masked point L2",
        sampling="original generic dense weighted-mean MPPI; finite-candidate and small-population guards",
        physics="native installed MuJoCo Warp; full WUJI mesh; no virtual object actuators")
