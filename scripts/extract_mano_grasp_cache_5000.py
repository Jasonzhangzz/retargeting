#!/usr/bin/env python3
"""Extract 5,000 rows/type from the frozen MANO personalized CEM cache."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "mano_personalized_cem_20261105"
DEFAULT_OUTPUT = Path("/mnt/cpfs/zbl-cpfs-new/dataset/jasonzhang/mano_grasp_cache_5000")
FIELDS = [
    "variables", "qpos", "mano_joints", "object_pose_world", "object_pose_after_rollout",
    "object_scale", "scene_scale", "initial_cost", "final_cost", "penetration", "soft_penetration",
    "penetration_depth_m", "mapped_hand_points", "mapped_hand_normals", "mapped_object_points",
    "mapped_object_normals", "mapped_distance", "projection_face_ids", "projection_barycentric",
    "collision_contact_count", "collision_depth_m", "triangle_object_depth_m", "contact_hand_points",
    "contact_object_points", "contact_normals", "contact_gap", "object_final_origin", "rollout_cost",
    "valid", "rollout_valid", "rollout_stable", "rollout_pose_cost", "rollout_velocity_cost",
    "rollout_linear_speed_m_s", "rollout_angular_speed_rad_s", "rollout_free_pose_cost",
    "rollout_free_velocity_cost", "rollout_free_cost", "rollout_displacement_m", "rollout_rotation_rad",
    "rollout_center_world", "rollout_rotation", "rollout_linear_velocity", "rollout_angular_velocity",
    "rollout_initial_contact_count", "rollout_final_contact_count", "rollout_max_row_force_n",
    "rollout_eligible", "rollout_tangential_displacement_m", "rollout_normal_force_n", "rollout_tangent_force_n",
    "rollout_implicit_converged", "rollout_newton_linear_solves", "anchor_distance_m", "object_template_points",
    "patch_hand_points", "patch_object_points", "patch_gap_m", "patch_anchor_offset_m", "patch_fitted",
    "patch_face_ids", "patch_coverage", "patch_gpu_overflow_count", "anchor_cost", "patch_fit_cost",
    "patch_fit_slot_error", "patch_fit_coverage_cost", "patch_fit_required_cost", "patch_group_fitted",
    "patch_all_required_fingers", "patch_geometry_accepted", "patch_joint_accepted", "total_cost_v3",
    "optimizer_cost_v3", "optimizer_evaluations", "optimizer_iterations", "optimizer_early_stopped",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_manifest(final_path: Path) -> Path:
    for parent in final_path.parents:
        candidate = parent / "MANIFEST.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"MANIFEST.json for {final_path}")


def collect_type(typ: str, limit: int):
    files = sorted((SOURCE / typ).glob("**/final.npz"))
    records = []
    loaded = {}
    for path in files:
        z = np.load(path, allow_pickle=False, mmap_mode="r")
        n = len(z["qpos"])
        manifest = json.loads(find_manifest(path).read_text())
        cases = manifest["source"]["cases"]
        if len(cases) != n:
            raise ValueError(f"case count mismatch: {path}")
        loaded[str(path)] = (path, z, cases)
        for row in range(n):
            valid = bool(z["valid"][row])
            stable = bool(z["rollout_stable"][row])
            geometry = bool(z["patch_geometry_accepted"][row])
            joint = bool(z["patch_joint_accepted"][row])
            # Stable + geometrically accepted rows are preferred, then valid
            # rows. Source path/row are deterministic tie-breakers.
            score = (int(stable) * 8 + int(geometry) * 4 + int(joint) * 2 + int(valid))
            records.append((score, str(path), row))
    records.sort(key=lambda x: (-x[0], x[1], x[2]))
    selected = records[:limit]
    if len(selected) != limit:
        raise ValueError(f"{typ}: only {len(selected)} rows")
    grouped = {}
    for score, path_text, row in selected:
        grouped.setdefault(path_text, []).append((score, row))
    arrays = {name: [] for name in FIELDS}
    source_case_name = []
    source_scene_path = []
    source_scene_scale = []
    source_shard = []
    source_row = []
    for path_text in sorted(grouped):
        path, z, cases = loaded[path_text]
        rows = np.asarray(sorted((row for _score, row in grouped[path_text])), dtype=np.int64)
        # Slice each shard in one operation. The previous row-by-row stack
        # multiplied Python object overhead and unnecessarily retained tens of
        # gigabytes for the same numeric payload.
        for name in FIELDS:
            if name in z:
                arrays[name].append(np.asarray(z[name][rows]))
        for row in rows.tolist():
            case = cases[row]
            source_case_name.append(str(case.get("name", f"{path.name}:{row}")))
            source_scene_path.append(str(case.get("scene_path", "")))
            source_scene_scale.append(float(case.get("scene_scale", z["scene_scale"][row])))
            source_shard.append(str(path))
            source_row.append(int(row))
    packed = {}
    for name, values in arrays.items():
        if values:
            packed[name] = np.concatenate(values, axis=0)
    packed.update(
        source_case_name=np.asarray(source_case_name, dtype="U128"),
        source_scene_path=np.asarray(source_scene_path, dtype="U256"),
        source_scene_scale=np.asarray(source_scene_scale, dtype=np.float32),
        source_shard=np.asarray(source_shard, dtype="U320"),
        source_row=np.asarray(source_row, dtype=np.int32),
    )
    return packed, dict(source_files=len(files), source_rows=len(records), selected_scores=[int(x[0]) for x in selected])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--limit-per-type", type=int, default=5000)
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    types = sorted([p.name for p in SOURCE.iterdir() if p.is_dir()], key=lambda x: int(x.split("_")[0]))
    global_manifest = dict(schema="mano_grasp_cache_v1", created_utc=time.time(), source=str(SOURCE),
                           selection="stable rollout + patch geometry/joint accepted rows prioritized; valid rows fill to limit",
                           limit_per_type=args.limit_per_type, types={})
    for typ in types:
        folder = out / typ; folder.mkdir(parents=True, exist_ok=True)
        packed, stats = collect_type(typ, args.limit_per_type)
        tmp = folder / ".poses.tmp"
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **packed)
        tmp.replace(folder / "poses.npz")
        report = dict(type=typ, samples=args.limit_per_type, **stats,
                      valid=int(packed["valid"].sum()), rollout_stable=int(packed["rollout_stable"].sum()),
                      patch_geometry_accepted=int(packed["patch_geometry_accepted"].sum()),
                      patch_joint_accepted=int(packed["patch_joint_accepted"].sum()),
                      files={"poses.npz": sha(folder / "poses.npz")})
        (folder / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        global_manifest["types"][typ] = report
        (out / "manifest.json").write_text(json.dumps(global_manifest, indent=2, ensure_ascii=False))
        print(json.dumps(dict(event="type_complete", **{k: v for k, v in report.items() if k != "files"}), ensure_ascii=False), flush=True)
    global_manifest["complete"] = len(global_manifest["types"]) == len(types) and all(v["samples"] == args.limit_per_type for v in global_manifest["types"].values())
    (out / "manifest.json").write_text(json.dumps(global_manifest, indent=2, ensure_ascii=False))
    print(json.dumps(dict(event="complete", output=str(out), types=len(types), samples=sum(v["samples"] for v in global_manifest["types"].values()), complete=global_manifest["complete"]), ensure_ascii=False))


if __name__ == "__main__": main()
