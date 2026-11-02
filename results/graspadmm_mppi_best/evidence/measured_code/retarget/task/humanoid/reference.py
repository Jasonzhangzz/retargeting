"""Derived sprint contact labels; the public reference arrays are preserved."""

from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco
from retarget.common.results import write_json


def generate_contacts(directory, threshold=0.035):
    directory = Path(directory)
    source = directory / "reference_raw.npz"
    raw = dict(np.load(source))
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    sites = [model.site(name).id for name in ("left_foot", "right_foot")]
    bodies = [model.site_bodyid[s] for s in sites]
    floor = next(
        i
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
    )
    geom_ids = []
    for body in bodies:
        ids = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == body
            and (
                model.geom_contype[g]
                or model.geom_conaffinity[g]
                or g in model.pair_geom1
                or g in model.pair_geom2
            )
        ]
        if not ids:
            raise ValueError(f"No collision geometry on foot body {body}")
        geom_ids.append(ids)
    gaps = np.zeros((len(raw["qpos"]), 2))
    positions = np.zeros((len(gaps), 2, 3))
    for t, q in enumerate(raw["qpos"]):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        positions[t] = data.site_xpos[sites]
        for foot, ids in enumerate(geom_ids):
            gaps[t, foot] = min(
                mujoco.mj_geomDistance(model, data, floor, g, 100.0, None) for g in ids
            )
    raw.update(
        contact=(gaps <= threshold).astype(np.float32),
        contact_pos=positions.astype(np.float32),
    )
    np.savez_compressed(directory / "reference.npz", **raw)
    info_path = directory / "task_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["contact_site_ids"] = sites
    write_json(info_path, info)
    write_json(
        directory / "contact_provenance.json",
        dict(
            method="reference FK; minimum signed collision geometry distance to floor",
            threshold_m=threshold,
            sites=["left_foot", "right_foot"],
            geom_ids=geom_ids,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            upstream_annotation=False,
            stance_frames=(gaps <= threshold).sum(0).tolist(),
            flight_frames=int((gaps > threshold).all(1).sum()),
        ),
    )
    np.savez_compressed(directory / "contact_gaps.npz", signed_gap_m=gaps)
