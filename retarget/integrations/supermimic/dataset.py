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


def export(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Export requires a new destination: {destination}")
    destination.mkdir(parents=True)
    (destination / "meshes").mkdir()
    bundle = root / "results/graspadmm_mppi_best"
    index = json.loads((bundle / "index.json").read_text())["cases"]
    values = np.load(bundle / "candidates.npz", allow_pickle=False)
    positions = {str(cid): i for i, cid in enumerate(values["case_ids"])}
    soft = root / "output/graspadmm_soft_center_mppi_200_pi0"
    recent = {r["id"]: r for r in json.loads((soft / "summary.json").read_text())["jobs"]} if soft.exists() else {}
    cases, records, qrefs, poses, raw_poses = {}, [], [], [], []
    for row in index:
        cid = row["case_id"]
        case = cid.split("_seed")[0]
        mesh = bundle / row["mesh"]
        required = row["required_cbody"]
        source = "portable_admm_mppi"
        raw = values
        choices = [(method, raw[f"{method}_qpos"][positions[cid]], bundle / "candidates.npz") for method in ("admm", "mppi")]
        if cid in recent:
            r = recent[cid]
            # Resolve the known local source explicitly, independent of old absolute prefixes.
            old = Path(r["source_summary"])
            marker = "output/graspadmm_soft_center_200_pi0/"
            a = root / "output/graspadmm_soft_center_200_pi0" / str(old).split(marker, 1)[1]
            a = a.parent
            b = soft / "cases" / cid
            if not all((p / "native_candidate.npz").is_file() for p in (a, b)):
                raise ValueError(f"Incomplete completed soft-center cache: {cid}")
            meta = json.loads((a / "native_candidate.json").read_text())
            mesh = a / "native_scene/object.obj"
            if not mesh.is_file():
                source_mesh = Path(meta["mesh_path"])
                mesh = root / str(source_mesh).split(str(root) + "/", 1)[-1]
            required = meta["required_cbody"]
            choices = [(method, np.load(p / "native_candidate.npz", allow_pickle=False)["grasp_qpos"], p / "native_candidate.npz")
                       for method, p in (("admm", a), ("mppi", b))]
            source = "soft_center_200_pi0"
        digest = sha(mesh)
        mesh_name = f"meshes/{digest}.obj"
        target = destination / mesh_name
        if not target.exists():
            shutil.copy2(mesh, target)
        case_record = dict(case_id=case, type_id=int(row["template"].split("_")[0]), template=row["template"],
                           primitive=row["primitive"], mesh=mesh_name, mesh_sha256=digest, required_cbody=required)
        if case in cases and cases[case] != case_record:
            raise ValueError(f"Case geometry or contact requirements changed between seeds: {case}")
        cases[case] = case_record
        for method, q, origin in choices:
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            valid = q.shape == (27,) and np.isfinite(q).all() and np.linalg.norm(q[3:7]) > 1e-10
            seed = int(row["seed"])
            if source == "soft_center_200_pi0":
                evaluation = origin.parent / ("graspadmm_evaluation" if method == "admm" else "evaluation") / "evaluation.json"
                result = json.loads(evaluation.read_text())
                provenance = dict(state=result["state"], task_success=bool(result["task_success"]),
                                  failure_stage=result.get("failure_stage"), protocol=result.get("evaluation_protocol", result.get("protocol")),
                                  metadata_path=str(evaluation.relative_to(root)), metadata_sha256=sha(evaluation))
            else:
                provenance = dict(state="recorded", task_success=bool(values[f"{method}_success"][positions[cid]]),
                                  failure_stage=None, protocol="graspadmm_native_v1",
                                  metadata_path=str((bundle / "index.json").relative_to(root)), metadata_sha256=sha(bundle / "index.json"),
                                  original_evaluation_sha256=row["source_evaluation_sha256"])
            records.append(dict(case_id=case, trial_id=cid, seed=seed, method=method, source=source,
                                source_path=str(origin.relative_to(root)), source_sha256=sha(origin), valid=bool(valid),
                                historical_evaluation=provenance,
                                split="train" if seed < 8 else "validation" if seed == 8 else "test"))
            raw_poses.append(q if q.shape == (27,) else np.full(27, np.nan))
            qrefs.append(q[7:] if valid else np.zeros(20))
            poses.append(relative_pose(q) if valid else np.array([0., 0., .1, 1., 0., 0., 0.]))
    ordered = sorted(cases.values(), key=lambda r: (r["type_id"], r["primitive"]["size_scale"]))
    if len(ordered) != 161 or len({r["type_id"] for r in ordered}) != 31 or len(records) != 3220:
        raise ValueError("Expected 31 types, 161 sizes and both candidates for all 1610 trials")
    np.savez_compressed(destination / "references.npz", qref=np.asarray(qrefs, np.float32), object_pose=np.asarray(poses, np.float32),
                        original_grasp_qpos=np.asarray(raw_poses, np.float64))
    manifest = dict(schema=SCHEMA, cases=ordered, records=records, arrays_sha256=sha(destination / "references.npz"),
                    success_filter=False, body_names=BODY_NAMES, source_root=str(root))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(cases=len(ordered), records=len(records), invalid=sum(not r["valid"] for r in records),
                soft_center_records=sum(r["source"] == "soft_center_200_pi0" for r in records), manifest_sha256=sha(destination / "manifest.json"))


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
