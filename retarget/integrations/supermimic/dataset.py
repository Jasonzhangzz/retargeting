"""Portable, split-aware grasp references; no success-based filtering."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

BODY_NAMES = ["right_palm_link"] + [f"right_finger{f}_link{k}" for f in range(1, 6) for k in range(1, 5)]
SCHEMA = "wuji2_grasp_hold:2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_pose(q):
    """A hand pose in the object frame -> object pose in the fixed palm frame."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (27,) or not np.isfinite(q).all() or np.linalg.norm(q[3:7]) < 1e-10:
        raise ValueError("Invalid cached xyz/wxyz/20-joint grasp")
    inv = Rotation.from_quat(q[3:7], scalar_first=True).inv()
    return np.r_[-inv.apply(q[:3]), inv.as_quat(scalar_first=True)]


class GraspDataset:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.is_dir():
            self.path /= "manifest.json"
        cache = os.environ.get("RETARGET_GRASP_DATA_CACHE")
        expected = os.environ.get("RETARGET_GRASP_EXPECTED_MANIFEST")
        if cache:
            if not expected:
                raise ValueError("A runtime data cache requires its expected manifest SHA-256")
            self.path = Path(cache).resolve() / "manifest.json"
        self.root = self.path.parent
        self.signature = sha(self.path)
        if cache and self.signature != expected:
            raise ValueError("Runtime cache manifest mismatch")
        self.manifest = json.loads(self.path.read_text())
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("Unsupported grasp dataset schema")
        if sha(self.root / "references.npz") != self.manifest["arrays_sha256"]:
            raise ValueError("Grasp reference array hash mismatch")
        self.cases, self.records = self.manifest["cases"], self.manifest["records"]
        self.ids = {c["case_id"]: i for i, c in enumerate(self.cases)}
        with np.load(self.root / "references.npz", allow_pickle=False) as values:
            self.arrays = {key: values[key].copy() for key in values.files}
        self.groups = np.zeros((len(self.cases), 6, len(BODY_NAMES)), np.float32)
        self.allowed = np.zeros((len(self.cases), 6), np.float32)
        for i, case in enumerate(self.cases):
            if sha(self.root / case["mesh"]) != case["mesh_sha256"]:
                raise ValueError(f"Mesh hash mismatch: {case['case_id']}")
            for j, group in enumerate(case["required_cbody"]):
                for body in group:
                    self.groups[i, j, BODY_NAMES.index(body)] = 1
                    finger = 5 if body == "right_palm_link" else int(body[len("right_finger")]) - 1
                    self.allowed[i, finger] = 1
        self.required = (self.groups.sum(-1) > 0).astype(np.float32)

    def pools(self, split):
        result = [[] for _ in self.cases]
        for i, record in enumerate(self.records):
            if record["split"] == split:
                result[self.ids[record["case_id"]]].append(i)
        if any(not pool for pool in result):
            raise ValueError(f"Missing {split} candidates for one or more cases")
        return result
