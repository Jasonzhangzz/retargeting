"""Measure initial geometry across recorded sizes; no rollout frame is used."""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.viewer.runtime import load_case


def bone_directions(model, data, q):
    data.qpos[:26] = q
    mujoco.mj_kinematics(model, data)
    palm = model.body("right_palm_link").id
    inverse = data.xmat[palm].reshape(3, 3).T
    directions = []
    for number, name in enumerate(("thumb", "index", "middle", "ring", "pinky"), 1):
        bodies = [model.body(f"right_finger{number}_link{link}").id for link in (2, 3, 4)]
        points = [data.xpos[body].copy() for body in bodies]
        points.append(data.xpos[bodies[-1]]+data.xmat[bodies[-1]].reshape(3,3)@model.site(f"right_{name}_tip").pos)
        vectors = np.diff(points, axis=0)@inverse.T
        directions.extend(vectors/np.linalg.norm(vectors, axis=1, keepdims=True))
    return np.asarray(directions)


def audit(batch):
    batch = Path(batch).resolve()
    cases = json.loads((batch/"manifest.json").read_text())["cases"]
    unique={}
    for case in cases:
        key=case.get('case_id',case['name'])
        if key not in unique or case.get('phase')=='hold':unique[key]=case
    cases=list(unique.values())
    jobs = {j["id"]: j for j in json.loads((batch/"summary.json").read_text())["jobs"]}
    rows = []
    for case in cases:
        if not case.get("has_hand_pose", True):
            continue
        model, qs = load_case(batch, case)
        if case.get('phase')=='acquisition':qs=qs[-1:]
        data = mujoco.MjData(model); data.qpos[:] = qs[0]; mujoco.mj_kinematics(model, data)
        palm, obj = model.body("right_palm_link").id, model.body("grasp_object").id
        inverse = data.xmat[palm].reshape(3,3).T
        position = inverse@(data.xpos[obj]-data.xpos[palm])
        rotation = inverse@data.xmat[obj].reshape(3,3)
        work = Path(jobs[case["case_id"]]["work"])
        files = list((work/"ca_seed").glob("**/initialization.json"))
        bone_error = None
        seed_archive=work/'spider_seed/spider_seed.npz'
        if seed_archive.is_file():
            with np.load(seed_archive) as archive:reference=archive['gesture_qpos'].copy()
            actual=bone_directions(model,data,qs[0,:26]);target=bone_directions(model,data,reference)
            bone_error=float(np.degrees(np.arccos(np.clip(np.sum(actual*target,axis=1),-1,1))).max())
        if files:
            seed = json.loads(files[0].read_text())[0]
            fit = seed["template_fit"]["numerical_seed_fit"]
            reference = np.asarray(fit["reference_qpos"])
            actual = bone_directions(model, data, qs[0,:26])
            target = bone_directions(model, data, reference)
            bone_error = float(np.degrees(np.arccos(np.clip(np.sum(actual*target,axis=1),-1,1))).max())
        rows.append(dict(id=case["case_id"], template=case["template"], scale=case["radius_scale"],
            kind=case["primitive"]["kind"], stage=case["stage"], object_center_palm_m=position.tolist(),
            rotation_palm=rotation.tolist(), bone_deviation_deg=bone_error))
    for row in rows:
        nominal = next((r for r in rows if r["template"]==row["template"] and r["scale"]==1),None)
        if nominal is None:
            continue
        a,b = np.asarray(row["rotation_palm"]), np.asarray(nominal["rotation_palm"])
        # Spheres have no observable orientation. Cylinder/disk/plate axes
        # are unsigned; a 180° equivalent symmetry is not a pose failure.
        row["axis_change_deg"] = (0. if row["kind"]=="sphere" else
            float(np.degrees(np.arccos(np.clip(abs(a[:,2]@b[:,2]),0,1)))))
        delta = np.asarray(row["object_center_palm_m"])-nominal["object_center_palm_m"]
        row["tangential_center_change_mm"] = float(np.linalg.norm(delta[[0,2]])*1000)
        row["flagged"] = bool(row["axis_change_deg"]>15 or row["tangential_center_change_mm"]>20
                              or (row["bone_deviation_deg"] or 0)>45)
    result = dict(frame="actual palm; initial saved pose only", thresholds=dict(axis_deg=15,
        tangential_center_mm=20, bone_deviation_deg=45), rows=rows,
        flagged_cases=[r["id"] for r in rows if r.get("flagged")])
    (batch/"size_consistency_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("batch",type=Path)
    result=audit(parser.parse_args().batch)
    print(json.dumps(dict(cases=len(result["rows"]), flagged=len(result["flagged_cases"]),
        flagged_types=sorted({int(r["template"].partition("_")[0]) for r in result["rows"] if r.get("flagged")}))))
