#!/usr/bin/env python3
"""Export frozen MANO grasps as a self-contained Moxian kinematic dataset.

The source campaign already contains MANO qpos for every Dexonomy grasp type.
This exporter performs the deterministic inverse anatomical map to the official
25-DOF Moxian/QnBot right-hand joint order, evaluates the copied URDF FK, and
stores 4,000 distinct object/size rows per type.  It does not change the
source MANO results or rerun a grasp optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CHAINS = ((13, 14, 15), (1, 2, 3), (4, 5, 6), (10, 11, 12), (7, 8, 9))
TIPS = (744, 320, 443, 554, 671)
NODE_ORDER = ["wrist"] + [n for f in ("thumb", "index", "middle", "ring", "little")
                            for n in ([f + "_metacarpal", f + "_proximal", f + "_intermediate",
                                       f + "_distal", f + "_tip"] if f != "thumb" else
                                      ["thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"])]
JOINT_ORDER = [
    "right_thumb_cmc_abduction", "right_thumb_cmc_flexion", "right_thumb_cmc_opposition",
    "right_thumb_mcp_abduction", "right_thumb_mcp_flexion", "right_thumb_ip_flexion",
    "right_index_mcp_abduction", "right_index_mcp_flexion", "right_index_pip_flexion", "right_index_dip_flexion",
    "right_middle_mcp_abduction", "right_middle_mcp_flexion", "right_middle_pip_flexion", "right_middle_dip_flexion",
    "right_ring_cmc_cup", "right_ring_mcp_abduction", "right_ring_mcp_flexion", "right_ring_pip_flexion", "right_ring_dip_flexion",
    "right_little_cmc_opposition", "right_little_cmc_cup", "right_little_mcp_abduction", "right_little_mcp_flexion",
    "right_little_pip_flexion", "right_little_dip_flexion",
]


def rotvec_to_matrix(v):
    v = np.asarray(v, dtype=np.float64)
    shape = v.shape[:-1]; theta = np.linalg.norm(v, axis=-1, keepdims=True)
    a = v / np.where(theta < 1e-12, 1.0, theta)
    x, y, z = a[..., 0], a[..., 1], a[..., 2]
    K = np.zeros((*shape, 3, 3), dtype=np.float64)
    K[..., 0, 1], K[..., 0, 2] = -z, y
    K[..., 1, 0], K[..., 1, 2] = z, -x
    K[..., 2, 0], K[..., 2, 1] = -y, x
    s = np.sin(theta)[..., None]; c = np.cos(theta)[..., None]
    I = np.broadcast_to(np.eye(3), (*shape, 3, 3))
    return I + s * K + (1.0 - c) * np.einsum("...ij,...jk->...ik", K, K)


def euler_xyz_to_matrix(a):
    a = np.asarray(a, dtype=np.float64); x, y, z = a[..., 0], a[..., 1], a[..., 2]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    R = np.empty((*a.shape[:-1], 3, 3), dtype=np.float64)
    R[..., 0, 0] = cz * cy; R[..., 0, 1] = cz * sy * sx - sz * cx; R[..., 0, 2] = cz * sy * cx + sz * sx
    R[..., 1, 0] = sz * cy; R[..., 1, 1] = sz * sy * sx + cz * cx; R[..., 1, 2] = sz * sy * cx - cz * sx
    R[..., 2, 0] = -sy; R[..., 2, 1] = cy * sx; R[..., 2, 2] = cy * cx
    return R


def matrix_to_euler_xyz(R):
    R = np.asarray(R, dtype=np.float64)
    y = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0)); cy = np.cos(y)
    x = np.arctan2(R[..., 2, 1], R[..., 2, 2]); z = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    gimbal = np.abs(cy) < 1e-8
    if np.any(gimbal):
        x = np.where(gimbal, np.arctan2(-R[..., 1, 2], R[..., 1, 1]), x); z = np.where(gimbal, 0.0, z)
    return np.stack((x, y, z), axis=-1)


def matrix_to_quat_xyzw(R):
    R = np.asarray(R, dtype=np.float64); q = np.empty((*R.shape[:-2], 4), dtype=np.float64)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q[..., 3] = 0.5 * np.sqrt(np.maximum(1.0 + tr, 1e-15))
    d = np.where(q[..., 3] > 1e-8, 4.0 * q[..., 3], 1.0)
    q[..., 0] = (R[..., 2, 1] - R[..., 1, 2]) / d; q[..., 1] = (R[..., 0, 2] - R[..., 2, 0]) / d; q[..., 2] = (R[..., 1, 0] - R[..., 0, 1]) / d
    # Stable diagonal branches for rotations close to pi.
    bad = q[..., 3] <= 1e-8
    if np.any(bad):
        for i, (j, k) in enumerate(((1, 2), (2, 0), (0, 1))):
            mask = bad & (R[..., i, i] >= np.maximum(R[..., j, j], R[..., k, k]))
            qi = 0.5 * np.sqrt(np.maximum(1.0 + R[..., i, i] - R[..., j, j] - R[..., k, k], 1e-15))
            q[..., i] = np.where(mask, qi, q[..., i]); denom = np.where(qi > 1e-8, 4.0 * qi, 1.0)
            q[..., 3] = np.where(mask, (R[..., k, j] - R[..., j, k]) / denom, q[..., 3])
            q[..., j] = np.where(mask, (R[..., j, i] + R[..., i, j]) / denom, q[..., j]); q[..., k] = np.where(mask, (R[..., k, i] + R[..., i, k]) / denom, q[..., k])
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def read_mano_rest(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    vertices = np.asarray(data["v_template"], dtype=np.float64)
    joints = np.asarray(data.get("retarget_rest_joints", np.asarray(data["J_regressor"]) @ vertices), dtype=np.float64)
    lateral = joints[1] - joints[7]
    forward = joints[4] - joints[0]
    normal = np.cross(lateral, forward)
    normal /= np.linalg.norm(normal)
    basis = np.zeros((15, 3, 3), dtype=np.float64)
    for finger, chain in enumerate(CHAINS):
        for k, j in enumerate(chain):
            end = joints[chain[k + 1]] if k < 2 else vertices[TIPS[finger]]
            twist = end - joints[j]
            twist /= np.linalg.norm(twist)
            flex = np.cross(twist, normal)
            flex /= np.linalg.norm(flex)
            if finger == 0:
                parent = chain[k - 1] if k else 0
                twist = joints[j] - joints[parent]
                twist /= np.linalg.norm(twist)
                flex = np.cross(-twist, [-1.0, 1.0, 1.0])
                flex /= np.linalg.norm(flex)
            spread = np.cross(twist, flex)
            basis[j - 1] = np.stack((flex, spread, twist), axis=1)
    neutral_pose = np.asarray(data.get("hands_mean", np.zeros(45)), dtype=np.float64).reshape(15, 3)
    neutral_rot = rotvec_to_matrix(neutral_pose)
    neutral_local = np.einsum("jki,jkl,jlm->jim", basis, neutral_rot, basis)
    neutral_angles = matrix_to_euler_xyz(neutral_local)
    return vertices, joints, basis, neutral_angles


def mano_angles(qpos: np.ndarray, basis: np.ndarray, neutral_angles: np.ndarray) -> np.ndarray:
    pose = np.asarray(qpos[:, 6:51], dtype=np.float64).reshape(-1, 15, 3)
    rot = rotvec_to_matrix(pose.reshape(-1, 3)).reshape(-1, 15, 3, 3)
    local = np.einsum("jki,njkl,jlm->njim", basis, rot, basis)
    return matrix_to_euler_xyz(local) - neutral_angles[None]


def map_to_moxian(qpos: np.ndarray, basis: np.ndarray, neutral_angles: np.ndarray, limits: dict[str, tuple[float, float]]):
    a = mano_angles(qpos, basis, neutral_angles)
    n = len(a)
    values = {}
    def put(name, joint, axis): values[name] = a[:, joint - 1, axis]
    # Moxian CMC has three explicit thumb axes; MANO CMC is decomposed in the
    # same anatomical basis (flex, spread, twist) used by the local runtime.
    put("right_thumb_cmc_abduction", 13, 1); put("right_thumb_cmc_flexion", 13, 0); put("right_thumb_cmc_opposition", 13, 2)
    put("right_thumb_mcp_abduction", 14, 1); put("right_thumb_mcp_flexion", 14, 0); put("right_thumb_ip_flexion", 15, 0)
    for prefix, base in (("index", 1), ("middle", 4), ("ring", 10), ("little", 7)):
        put(f"right_{prefix}_mcp_abduction", base, 1); put(f"right_{prefix}_mcp_flexion", base, 0)
        put(f"right_{prefix}_pip_flexion", base + 1, 0); put(f"right_{prefix}_dip_flexion", base + 2, 0)
    # The extra ulnar CMC channels have no independent MANO joint. Keep them
    # at the calibrated neutral pose; the remaining MCP channels are retained.
    values["right_ring_cmc_cup"] = np.zeros(n)
    values["right_little_cmc_opposition"] = np.zeros(n)
    values["right_little_cmc_cup"] = np.zeros(n)
    q = np.stack([values[name] for name in JOINT_ORDER], axis=1)
    for i, name in enumerate(JOINT_ORDER):
        lo, hi = limits[name]
        q[:, i] = np.clip(q[:, i], lo, hi)
    return q.astype(np.float32), a.astype(np.float32)


def parse_urdf(path: Path):
    root = ET.parse(path).getroot()
    joints = []
    limits = {}
    children = set()
    for node in root.findall("joint"):
        name, typ = node.get("name"), node.get("type")
        parent, child = node.find("parent").get("link"), node.find("child").get("link")
        children.add(child)
        origin = node.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        T = np.eye(4); T[:3, :3] = euler_xyz_to_matrix(rpy); T[:3, 3] = xyz
        axis_node = node.find("axis"); axis = np.fromstring(axis_node.get("xyz", "0 0 0"), sep=" ") if axis_node is not None else np.zeros(3)
        if typ == "revolute":
            lim = node.find("limit"); limits[name] = (float(lim.get("lower")), float(lim.get("upper")))
        joints.append((name, typ, parent, child, T, axis))
    roots = {x.get("name") for x in root.findall("link")} - children
    if len(roots) != 1: raise ValueError(f"URDF root is not unique: {roots}")
    pending = list(joints); ordered = []
    known = {next(iter(roots)): np.eye(4)}
    while pending:
        progress = False
        for joint in list(pending):
            if joint[2] not in known: continue
            pending.remove(joint); ordered.append(joint); progress = True
            name, typ, parent, child, T, axis = joint
            known[child] = known[parent] @ T
        if not progress: raise ValueError("URDF joint graph is disconnected")
    return ordered, limits


def moxian_fk(q: np.ndarray, joints, q_names):
    n = len(q); qmap = {name: q[:, i] for i, name in enumerate(q_names)}
    transforms = {joints[0][2]: np.broadcast_to(np.eye(4), (n, 4, 4)).copy()}
    for name, typ, parent, child, origin, axis in joints:
        base = np.einsum("nij,jk->nik", transforms[parent], origin)
        if typ == "revolute":
            rv = qmap[name][:, None] * axis[None, :]
            R = rotvec_to_matrix(rv)
            move = np.broadcast_to(np.eye(4), (n, 4, 4)).copy(); move[:, :3, :3] = R
            transforms[child] = np.einsum("nij,njk->nik", base, move)
        else:
            transforms[child] = base
    positions, quats = [], []
    for node in NODE_ORDER:
        link = "right_" + node
        T = transforms[link]
        positions.append(T[:, :3, 3]); quats.append(matrix_to_quat_xyzw(T[:, :3, :3]))
    return np.stack(positions, 1).astype(np.float32), np.stack(quats, 1).astype(np.float32)


def select_rows(root: Path, typ: str, limit: int):
    candidates = []
    for p in sorted(root.glob("mano*/" + typ + "/0000000/final.npz")):
        with np.load(p, allow_pickle=False, mmap_mode="r") as z:
            candidates.append((p, int(z["qpos"].shape[0])))
    if not candidates: raise RuntimeError(f"No MANO source with {limit} rows for {typ}")
    chosen = candidates[0][0]
    with np.load(chosen, allow_pickle=False) as z:
        count = len(z["qpos"]); rows = np.arange(min(count, limit), dtype=np.int64)
        # Prefer one row per recorded object/size case. If a source contains a
        # duplicate case, fill from the next frozen variant deterministically.
        key = z["source_case_name"] if "source_case_name" in z else np.arange(count)
        seen = set(); selected = []
        for i, value in enumerate(key.tolist()):
            token = str(value)
            if token in seen: continue
            seen.add(token); selected.append((chosen, i))
            if len(selected) == limit: break
    if len(selected) < limit:
        for p, _ in candidates[1:]:
            with np.load(p, allow_pickle=False) as z:
                key = z["source_case_name"] if "source_case_name" in z else np.arange(len(z["qpos"]))
                for i, value in enumerate(key.tolist()):
                    token = str(value)
                    if token in seen: continue
                    seen.add(token); selected.append((p, i))
                    if len(selected) == limit: break
            if len(selected) == limit: break
    if len(selected) != limit: raise RuntimeError(f"Only {len(selected)}/{limit} unique rows for {typ}")
    # Grouped loads keep output deterministic and avoid keeping all campaign
    # variants resident at once.
    grouped = {}
    for p, i in selected: grouped.setdefault(p, []).append(i)
    fields = ["qpos", "mano_joints", "object_pose_world", "object_scale", "scene_scale", "object_scene", "source_case_name", "valid", "rollout_valid", "rollout_stable"]
    out = {f: [] for f in fields}
    source_rows = []
    for p, ids in grouped.items():
        with np.load(p, allow_pickle=False) as z:
            for f in fields:
                if f in z: out[f].append(z[f][ids])
                elif f == "valid": out[f].append(np.ones(len(ids), dtype=bool))
                elif f in ("rollout_valid", "rollout_stable"): out[f].append(np.ones(len(ids), dtype=bool))
            source_rows.extend((str(p), int(i)) for i in ids)
    values = {f: np.concatenate(v, axis=0) for f, v in out.items()}
    values["source_rows"] = np.asarray(source_rows, dtype="U240")
    return values, chosen, candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mano-model", type=Path, required=True)
    ap.add_argument("--limit-per-type", type=int, default=4000)
    args = ap.parse_args(); repo = args.repo.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    source_root = repo / "results" / "mano50_wuji_success_20261108"
    mesh_root = repo / "data" / "grasp" / "assets" / "moxian"
    urdf = mesh_root / "resources" / "pinocchio" / "right_skeleton.urdf"
    if not urdf.is_file(): raise FileNotFoundError(urdf)
    vertices, joints, basis, neutral_angles = read_mano_rest(args.mano_model)
    joint_specs, limits = parse_urdf(urdf)
    limits = {name: limits[name] for name in JOINT_ORDER}
    primitive_src = repo / "data" / "grasp" / "templates" / "assets" / "hand" / "shadow" / "raw_anno"
    primitive_out = output / "grasp_primitives" / "raw_anno"; primitive_out.mkdir(parents=True, exist_ok=True)
    primitive_files = {}
    for p in sorted(primitive_src.glob("*.yaml")):
        dst = primitive_out / p.name; shutil.copy2(p, dst); primitive_files[p.name] = sha256(dst)
    for name in ("right.xml", "keypoint.yaml"):
        p = primitive_src.parent / name
        if p.is_file(): shutil.copy2(p, output / "grasp_primitives" / name); primitive_files[name] = sha256(output / "grasp_primitives" / name)
    mesh_out = output / "hand_assets" / "moxian"; shutil.copytree(mesh_root, mesh_out, dirs_exist_ok=True)
    global_manifest = output / "manifest.json"
    manifest = dict(schema="moxian_grasp_v1", created_utc=time.time(), hand="moxian_qnbot_right",
                    kinematic_map="MANO full axis-angle -> official QnBot 25-DOF anatomical channels; extra ring/little CMC channels fixed neutral",
                    source_campaign=str(source_root), limit_per_type=args.limit_per_type,
                    mano_model=str(args.mano_model), mano_model_sha256=sha256(args.mano_model),
                    urdf=str(urdf), urdf_sha256=sha256(urdf), joint_order=JOINT_ORDER,
                    node_order=NODE_ORDER, mesh_files=tree_sha(mesh_out), primitive_files=primitive_files, types={})
    types = sorted({p.parent.parent.name for p in source_root.glob("mano*/" + "*/0000000/final.npz")}, key=lambda x: int(x.split("_")[0]))
    for typ in types:
        folder = output / "dexonomy" / typ; folder.mkdir(parents=True, exist_ok=True)
        existing = folder / "manifest.json"
        if existing.is_file():
            try:
                old = json.loads(existing.read_text());
                if old.get("samples") == args.limit_per_type and (folder / "poses.npz").is_file():
                    manifest["types"][typ] = old; continue
            except Exception: pass
        values, chosen, candidates = select_rows(source_root, typ, args.limit_per_type)
        q, angles = map_to_moxian(values["qpos"], basis, neutral_angles, limits)
        positions, quats = moxian_fk(q, joint_specs, JOINT_ORDER)
        arrays = dict(moxian_qpos=np.concatenate((values["qpos"][:, :6].astype(np.float32), q), axis=1),
                      moxian_joint_qpos=q, moxian_joint_angles=angles, moxian_skeleton_positions=positions,
                      moxian_skeleton_quaternions_xyzw=quats, mano_qpos=values["qpos"].astype(np.float32),
                      mano_joints=values["mano_joints"].astype(np.float32), object_pose_world=values["object_pose_world"].astype(np.float32),
                      object_scale=values["object_scale"].astype(np.float32), scene_scale=values["scene_scale"].astype(np.float32),
                      source_case_name=values["source_case_name"], object_scene=values["object_scene"], source_rows=values["source_rows"],
                      valid=values["valid"].astype(bool), rollout_valid=values["rollout_valid"].astype(bool), rollout_stable=values["rollout_stable"].astype(bool))
        tmp = folder / ".poses.tmp"
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        tmp.replace(folder / "poses.npz")
        report = dict(type=typ, samples=len(q), source=str(chosen), source_candidates=len(candidates), unique_case_names=len(set(values["source_case_name"].tolist())),
                      finite=bool(all(np.isfinite(v).all() for v in (q, positions, quats, arrays["object_pose_world"]))),
                      q_min=q.min(0).tolist(), q_max=q.max(0).tolist(), object_sizes_unique=int(len(np.unique(values["object_scale"].round(7), axis=0))))
        (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        type_manifest = dict(**report, files={n: sha256(folder / n) for n in ("poses.npz", "report.json")})
        (folder / "manifest.json").write_text(json.dumps(type_manifest, indent=2, ensure_ascii=False))
        manifest["types"][typ] = type_manifest
        global_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(json.dumps(dict(event="type_complete", **report), ensure_ascii=False), flush=True)
    manifest["complete"] = len(manifest["types"]) == len(types) and all(v["samples"] == args.limit_per_type for v in manifest["types"].values())
    global_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(dict(event="complete", types=len(types), samples=sum(v["samples"] for v in manifest["types"].values()), output=str(output), complete=manifest["complete"]), ensure_ascii=False))


if __name__ == "__main__": main()
