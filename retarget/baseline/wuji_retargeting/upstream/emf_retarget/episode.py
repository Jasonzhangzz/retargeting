"""Independent recorded-episode adapter. No SDK or ego_fast_retarget imports.

Read decoded EMF poses in the local TX frame, run every native frame in order,
then select native results on the camera timeline. Angles: rad; positions: m.
"""
from pathlib import Path
import hashlib
import json
import time

import h5py
import ijson
import numpy as np

from . import EmfFrame, EmfRetargetPipeline
from .io import output_arrays

SIDES = ("left", "right")
METADATA_NAMES = {"instruction.json", "instruction_meta.json", "diversity.json",
                  "calibration.json", "replay_metadata.json", "diagnostics.json",
                  "wuji_glove_offline_manifest.json"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text(value):
    return value.decode("utf8") if isinstance(value, bytes) else str(value)


def episode_paths(source):
    source = Path(source).expanduser().resolve()
    root = source if source.is_dir() else source.parent
    h5 = source if source.suffix in (".h5", ".hdf5") else root / "wuji_glove_offline.h5"
    if not h5.is_file():
        raise FileNotFoundError(h5)
    named = root / (root.name + ".json")
    if source.is_file() and source.suffix == ".json":
        camera = source
    elif named.is_file():
        camera = named
    else:
        candidates = [p for p in sorted(root.glob("*.json"))
                      if p.name not in METADATA_NAMES and not p.name.endswith(".meta.json")]
        camera = candidates[0] if len(candidates) == 1 else None
    return h5, camera


def fingerprint(p, q, c):
    # Match recorded JSON/H5 before the SDK-compatible float32 input boundary.
    return np.concatenate([np.asarray(v).ravel() for v in (p, q, c)]).astype("<f8").tobytes()


def json_payload(message):
    poses = message.get("poses", [])
    if len(poses) != 5:
        raise ValueError("Expected five local EMF poses")
    def vector(value, fields):
        return [float(value[k]) for k in fields] if isinstance(value, dict) else list(value)
    return ([vector(p["pose"]["position"], "xyz") for p in poses],
            [vector(p["pose"]["orientation"], "xyzw") for p in poses],
            [float(p["confidence"]) for p in poses])


def load_stream(group, side):
    values = {k: group[k][:] for k in ("source_timestamp_ns", "source_seq", "action_timestamp_ns")}
    values["frame_id"] = np.asarray([text(x) for x in group["source_frame_id"][:]])
    for short, key in (("p", "position"), ("q", "orientation_xyzw"), ("c", "confidence")):
        values[short] = group["emf_poses/" + key][:]
    n = len(values["source_timestamp_ns"])
    if not n or any(len(v) != n for v in values.values()):
        raise ValueError(f"{side}: empty or inconsistent stream")
    for key, shape in (("p", (n, 5, 3)), ("q", (n, 5, 4)), ("c", (n, 5))):
        if values[key].shape != shape or not np.isfinite(values[key]).all():
            raise ValueError(f"{side}: invalid {key}")
    if np.any(np.linalg.norm(values["q"], axis=-1) < 1e-8):
        raise ValueError(f"{side}: zero EMF quaternion")
    stamps = values["source_timestamp_ns"]
    if np.any(stamps < 0) or np.any(stamps[1:] <= stamps[:-1]):
        raise ValueError(f"{side}: source timestamps must increase strictly")
    action = values["action_timestamp_ns"]
    if np.any(action[1:] < action[:-1]):
        raise ValueError(f"{side}: action timestamps must be sorted for nearest lookup")
    if np.any(values["frame_id"] != f"{side[0]}_hand_emf_tx"):
        raise ValueError(f"{side}: requires local TX-frame EMF poses")
    return values


def replay_joint_names(side, model):
    """Published joint-vector names; order is thumb->pinky, four DOFs each."""
    if model == "wuji1":
        return np.asarray([f"{side}_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)])
    names = []
    for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"):
        suffixes = ("cmc_flex", "cmc_abd", "mcp", "ip") if finger == "thumb" else ("mcp_flex", "mcp_abd", "pip", "dip")
        names.extend(f"{side[0]}_{finger}_{s}" for s in suffixes)
    return np.asarray(names)


def compute_native(source, models=("wuji1", "wuji2"), urdfs=None, progress=100):
    h5, _ = episode_paths(source)
    started = time.monotonic()
    arrays, lookups = {}, {}
    info = {"input_h5": str(h5), "input_sha256": sha256(h5), "models": list(models),
            "backend": "independent_python", "sides": {},
            "state_policy": "continuous_from_first_native_frame", "rx_corrections": None}
    with h5py.File(h5, "r", locking=False) as handle:
        if text(handle.attrs.get("schema", "")) != "rd001.wuji_offline_h5":
            raise ValueError("Unsupported HDF5 schema")
        for side in SIDES:
            stream = load_stream(handle[side], side)
            pipe = EmfRetargetPipeline(side, models, (urdfs or {}).get(side))
            results, lookup = [], {}
            n = len(stream["p"])
            for i in range(n):
                frame = EmfFrame(stream["p"][i], stream["q"][i], stream["c"][i],
                    seq=int(stream["source_seq"][i]),
                    timestamp_us=int(stream["source_timestamp_ns"][i]) // 1000,
                    frame_id=str(stream["frame_id"][i]))
                results.append(pipe.process(frame))
                lookup.setdefault(fingerprint(stream["p"][i], stream["q"][i], stream["c"][i]), []).append(i)
                if progress and ((i + 1) % progress == 0 or i + 1 == n):
                    print(f"[independent] {side} {i + 1}/{n}", flush=True)
            arrays.update(output_arrays(results, pipe))
            for field in ("source_timestamp_ns", "source_seq", "action_timestamp_ns"):
                arrays[f"{side}_{field}"] = stream[field]
            # Keep source URDF names too; published Hand1 names have side prefixes.
            for model in pipe.models:
                arrays[f"{side}_{model}_urdf_joint_names"] = arrays[f"{side}_{model}_joint_names"]
                arrays[f"{side}_{model}_joint_names"] = replay_joint_names(side, model)
            urdf = pipe.human.kinematics.urdf_path
            info["sides"][side] = {"frames": n, "human_urdf": str(urdf),
                "human_urdf_sha256": sha256(urdf),
                "source_start_ns": int(stream["source_timestamp_ns"][0]),
                "source_end_ns": int(stream["source_timestamp_ns"][-1])}
            lookups[side] = lookup
    info["seconds"] = time.monotonic() - started
    return arrays, info, lookups


def select_model(arrays, model):
    """Expose the joint arrays under the ego hand-stage data field names."""
    out = {"hand_retarget_backend": np.asarray("independent_python"),
           "hand_model": np.asarray(model), "hand_angle_unit": np.asarray("rad")}
    for side in SIDES:
        for name in ("skeleton_position", "skeleton_orientation_xyzw", "skeleton_confidence",
                     "source_timestamp_ns", "source_seq", "action_timestamp_ns"):
            out[f"{side}_{name}"] = arrays[f"{side}_{name}"]
        out[f"{side}_hand_q"] = arrays[f"{side}_{model}_q"]
        out[f"{side}_hand_q_names"] = arrays[f"{side}_{model}_joint_names"]
        out[f"{side}_skeleton_joint_names"] = arrays[f"{side}_skeleton_names"]
    return out


def align_camera(camera_json, native, lookups):
    matches = {s: [] for s in SIDES}
    missing = {s: [] for s in SIDES}
    nearest = {s: [] for s in SIDES}
    periods = {}
    for side in SIDES:
        gaps = np.diff(native[f"{side}_action_timestamp_ns"].astype(np.int64))
        gaps = gaps[gaps > 0]
        periods[side] = int(np.median(gaps)) if len(gaps) else None
    timestamps = []
    with Path(camera_json).open("rb") as stream:
        for row, frame in enumerate(ijson.items(stream, "data.item")):
            stamp = int(frame["timestamp"])
            timestamps.append(stamp)
            for side in SIDES:
                times = native[f"{side}_action_timestamp_ns"]
                message = frame.get(f"{side}_glove_emf_poses")
                candidates = lookups[side].get(fingerprint(*json_payload(message)), []) if isinstance(message, dict) else []
                if candidates:
                    index = min(candidates, key=lambda i: abs(int(times[i]) - stamp))
                else:
                    right = min(int(np.searchsorted(times, stamp, side="left")), len(times) - 1)
                    left = max(right - 1, 0)
                    index = min((left, right), key=lambda i: abs(int(times[i]) - stamp))
                    if periods[side] is not None and abs(int(times[index]) - stamp) > periods[side]:
                        index = -1
                        missing[side].append(row)
                    else:
                        nearest[side].append(row)
                matches[side].append(index)
    valid = (np.asarray(matches["left"]) >= 0) & (np.asarray(matches["right"]) >= 0)
    if not valid.any():
        raise ValueError("No paired camera frames within one native action period")
    rows = np.arange(len(timestamps), dtype=np.int64)
    stamps = np.asarray(timestamps, dtype=np.int64)
    out = {"frame_indices": rows[valid], "camera_timestamp_ns": stamps[valid],
           "timestamps": stamps[valid] / 1e9,
           "hand_retarget_backend": np.asarray("independent_python")}
    for side in SIDES:
        indices = np.asarray(matches[side], dtype=np.int64)[valid]
        out[f"{side}_native_frame_indices"] = indices
        for field in ("hand_q", "skeleton_position", "skeleton_orientation_xyzw", "skeleton_confidence", "source_timestamp_ns"):
            out[f"{side}_{field}"] = native[f"{side}_{field}"][indices]
        out[f"{side}_hand_q_names"] = native[f"{side}_hand_q_names"]
    out["wuji_hand_q"] = np.concatenate([out["right_hand_q"], out["left_hand_q"]], axis=1)
    out["wuji_hand_q_names"] = np.concatenate([out["right_hand_q_names"], out["left_hand_q_names"]])
    return out, {"camera_frames": len(timestamps), "paired_frames": int(valid.sum()),
        "dropped_frame_indices": rows[~valid].tolist(), "nearest_action_frame_indices": nearest,
        "unmatched_by_side": missing}


def save_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    temp.replace(path)


def run_episode(source, output, models=("wuji1", "wuji2"), urdfs=None, native_only=False):
    destination = Path(output).expanduser().resolve()
    _, camera_json = episode_paths(source)
    if not native_only and camera_json is None:
        raise FileNotFoundError("Camera JSON ambiguous/missing; use --native-only for native EMF")
    destination.mkdir(parents=True, exist_ok=False)
    arrays, info, lookups = compute_native(source, models, urdfs)
    np.savez_compressed(destination / "native_all.npz", **arrays)
    for model in models:
        native = select_model(arrays, model)
        np.savez_compressed(destination / f"{model}_native.npz", **native)
        if not native_only:
            camera, alignment = align_camera(camera_json, native, lookups)
            np.savez_compressed(destination / f"{model}_camera.npz", **camera)
            info["alignment"] = alignment
    if not native_only:
        info["camera_json"] = str(camera_json)
        info["camera_sha256"] = sha256(camera_json)
    info["output"] = str(destination)
    save_json(destination / "manifest.json", info)
    return info
