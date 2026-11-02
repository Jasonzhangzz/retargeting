"""One native Dexonomy/GraspADMM acceptance protocol for every grasp method.

Optimization and method-specific diagnostics stay in their original runners.
Only the final material hand/object configuration is transferred to this test;
method-specific preload, controls, proxy geometry and hold tests are not reused.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from retarget.common.results import write_json

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "graspadmm_native_v1"
CRITERION = dict(protocol=PROTOCOL, object_mass_kg=.1,
    translation_threshold_m=.05, rotation_threshold_deg=15.,
    directions=["+x", "-x", "+y", "-y", "+z", "-z"],
    evaluation_friction=[.6, .02], filter_qp_friction=[.6, 0.],
    skip_pregrasp=False, hard=False,
    source="Dexonomy grasp_admm filtering, squeeze/pregrasp generation and native eval_fc")


def evaluation_python():
    configured = os.environ.get("GRASPADMM_PYTHON")
    if configured:
        return str(Path(configured).expanduser().resolve())
    dedicated = ROOT / ".venv-graspadmm/bin/python"
    return str(dedicated if dedicated.is_file() else Path(sys.executable))



@lru_cache(maxsize=1)
def pinned_native_environment():
    """Permit direct evaluation only inside the exact dedicated native stack."""
    import importlib.metadata
    if Path(sys.prefix).resolve() != (ROOT/".venv-graspadmm").resolve():
        return False
    expected=dict(numpy="2.2.6",scipy="1.15.3",torch="2.5.1",
        mujoco="3.3.3",trimesh="4.5.3",qpsolvers="4.8.0",clarabel="0.11.1",
        rtree="1.4.1",transforms3d="0.4.2")
    expected["warp-lang"]="1.8.0"
    try:
        return all(importlib.metadata.version(name).split("+")[0]==version
                   for name,version in expected.items())
    except importlib.metadata.PackageNotFoundError:
        return False

def _arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def select_grasp(case_directory, method=None, trajectory=None):
    """Select an actual method output, never an accepted seed or best eval frame.

    Static sampling exports its optimized pose separately from the prediction.
    Dynamic methods contribute the final executed state. Failed MPPI acquisition
    can still contribute its genuinely optimized final candidate.
    """
    directory = Path(case_directory)
    static = directory / "grasp.npz"
    if static.is_file():
        values = _arrays(static)
        if "qpos" in values:
            return values["qpos"].reshape(-1), dict(kind="optimized_static_grasp", path=str(static))
    acquisition = directory / "acquisition/acquisition.json"
    if acquisition.is_file():
        details = json.loads(acquisition.read_text())
        if details.get("success") is False and "hand_qpos" in details:
            return np.asarray(details["hand_qpos"], float), dict(kind="optimized_mppi_acquisition", path=str(acquisition))
    if method == "contactaware":
        for path in [directory/"contactaware_reference.json", directory/"contactaware.json", directory/"summary.json"]:
            if not path.is_file():
                continue
            record=json.loads(path.read_text())
            records=record if isinstance(record,list) else [record]
            for item in records:
                if not isinstance(item,dict):
                    continue
                diagnostics=item.get("contactaware",item)
                diagnostics=diagnostics if isinstance(diagnostics,list) else [diagnostics]
                for value in diagnostics:
                    if isinstance(value,dict) and "reference_qpos" in value:
                        return np.asarray(value["reference_qpos"],float),dict(kind="optimized_contactaware_grasp",path=str(path))
    trajectory = Path(trajectory) if trajectory else directory / "trajectory.npz"
    if trajectory.is_file():
        values = _arrays(trajectory)
        if "qpos" in values and len(values["qpos"]):
            # A failure preview is not an output of a solver that never ran.
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            reason = summary.get("reason", summary.get("failed_stage"))
            if reason in ("initialization_failed", "contactaware_reference_failed", "no_feasible_candidate"):
                raise ValueError(f"No solver output: {reason}")
            return np.asarray(values["qpos"][-1], float), dict(kind="final_executed_state", path=str(trajectory))
    for filename in ("contactaware_reference.json", "contactaware.json"):
        path = directory / filename
        if path.is_file():
            records = json.loads(path.read_text())
            records = records if isinstance(records, list) else [records]
            for record in records:
                if "reference_qpos" in record:
                    return np.asarray(record["reference_qpos"], float), dict(kind="optimized_contactaware_grasp", path=str(path))
    raise ValueError(f"No final optimized grasp in {directory}")




def infer_template(model_path, selection=None):
    """Identify a saved taxonomy template from original case provenance."""
    selection=selection or {}
    if selection.get("template"):
        return selection["template"]
    names=[path.stem for path in (ROOT/"data/grasp/templates/assets/hand/shadow/raw_anno").glob("*.yaml")]
    texts=[str(model_path),json.dumps(selection)]
    directory=Path(model_path).parent
    for root in [directory,*list(directory.parents[:5])]:
        for filename in ("reference_manifest.json","summary.json","config.yaml"):
            path=root/filename
            if path.is_file() and path.stat().st_size<500000:
                texts.append(path.read_text())
    # Longer template names first: 1_Large_Diameter must not be matched as a
    # suffix of 11_...; exact full template strings carry their own identity.
    for text in texts:
        for name in names:
            if name in text:
                return name
    return None

def infer_required_contacts(model, model_path, selection):
    """Recover the annotated contact groups independently of active CA slots."""
    if "required_cbody" in selection:
        return selection["required_cbody"]
    from .graspadmm_scene import required_body_groups
    directory = Path(model_path).parent
    roots = [directory, *list(directory.parents[:5])]
    for root in roots:
        native_metadata=root/'native_candidate.json'
        if native_metadata.is_file():
            native=json.loads(native_metadata.read_text())
            if 'required_cbody' in native:
                return native['required_cbody']
        guides = [root/"contactaware/contact_guidance.npz",
                  root/"sequence/retarget/wuji_right/contactaware/contact_guidance.npz"]
        guides += sorted(root.glob("input_*/contactaware/contact_guidance.npz"))
        for guide in guides:
            if guide.is_file():
                values = _arrays(guide)
                if "hand_query_body_ids" in values:
                    ids = values["hand_query_body_ids"].reshape(-1)
                    if np.all((ids >= 0) & (ids < model.nbody)):
                        names = [model.body(int(body)).name for body in ids]
                        return required_body_groups(names)
        for filename in ("contactaware_reference.json", "contactaware.json"):
            path = root/filename
            if path.is_file():
                record = json.loads(path.read_text())
                record = record[0] if isinstance(record, list) and record else record
                if isinstance(record, dict) and "required_finger_contact" in record:
                    order = ("thumb", "index", "middle", "ring", "pinky")
                    return [[f"right_finger{order.index(name)+1}_link{k}" for k in range(1,5)]
                            if name in order else ["right_palm_link"]
                            for name in record["required_finger_contact"]]
        ref = root/"reference_manifest.json"
        if ref.is_file():
            info = json.loads(ref.read_text())
            source = info.get("source_experiment")
            if source and Path(source).is_dir() and Path(source).resolve() not in roots:
                # Original CA inputs retain all annotated slots, including
                # contacts released by CA; the common filter checks the groups.
                for guide in Path(source).glob("**/contact_guidance.npz"):
                    values = _arrays(guide)
                    if "hand_query_body_ids" in values:
                        ids=values["hand_query_body_ids"].reshape(-1)
                        if np.all((ids>=0)&(ids<model.nbody)):
                            return required_body_groups([model.body(int(body)).name for body in ids])
    return []


def record_native_evaluation(env, target):
    """Observe the original eval_fc calls without changing its stepping/checks."""
    from dexonomy.util.np_util import np_get_delta_pose
    directions = [dict(direction=name, tested=False, success=None, checks=[])
                  for name in CRITERION["directions"]]
    result = dict(qpos=[], ctrl=[], time=[], sim_time=[], direction_index=[], directions=directions)
    state = dict(direction=-1, elapsed=0.)
    original_step, original_reset = env.step_sim, env.reset_qpos
    original_force, original_pose = env.set_active_obj_ext_fdir, env.get_active_obj_pose

    def append():
        result["qpos"].append(env._data.qpos.copy())
        result["ctrl"].append(env._data.ctrl.copy())
        display_time=state["elapsed"]
        if result["time"] and display_time<=result["time"][-1]:
            display_time=float(np.nextafter(result["time"][-1],np.inf))
        result["time"].append(display_time)
        result["sim_time"].append(float(getattr(env._data,"time",state["elapsed"])))
        result["direction_index"].append(state["direction"])

    def reset(*args, **kwargs):
        state["direction"] = -1
        value = original_reset(*args, **kwargs)
        append()
        return value

    def step(substep):
        value = original_step(substep)
        state["elapsed"] += substep*env._model.opt.timestep
        append()
        return value

    def force(direction):
        value = original_force(direction)
        axis = np.asarray(direction)
        vectors=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        matches=np.flatnonzero(np.all(np.isclose(vectors,axis),axis=1))
        if len(matches):
            state["direction"]=int(matches[0])
            directions[state["direction"]]["tested"]=True
        return value

    def pose():
        value = original_pose()
        index = state["direction"]
        if index >= 0:
            translation, rotation = np_get_delta_pose(target, value)
            success=bool(translation<.05 and rotation<15.)
            directions[index]["checks"].append(dict(translation_m=float(translation), rotation_deg=float(rotation), success=success))
            directions[index]["success"]=success
        return value

    env.reset_qpos, env.step_sim = reset, step
    env.set_active_obj_ext_fdir, env.get_active_obj_pose = force, pose
    return result


def measure_primitive(vertices):
    """Measure stored benchmark geometry without treating cylinder rims as spheres."""
    vertices=np.asarray(vertices,float)
    extent=np.ptp(vertices,axis=0)
    radii=np.linalg.norm(vertices,axis=1)
    radial=np.linalg.norm(vertices[:,:2],axis=1)
    z_levels=len(np.unique(np.round(vertices[:,2],8)))
    if len(vertices)>8 and z_levels>4 and np.ptp(radii)<max(float(radii.max()),1e-9)*.01:
        kind="sphere"
    elif len(vertices)<=8:
        kind="plate" if extent[2]<min(extent[:2])*.6 else "cuboid"
    else:
        kind="disk" if extent[2]<max(extent[:2])*.6 else "cylinder"
    return dict(kind=kind,radius_m=round(float(radii.max() if kind=="sphere" else extent[0]/2 if kind in ("plate","cuboid") else radial.max()),9),
        height_m=round(float(extent[2]),9),extent_xyz_m=extent.tolist(),
        dimensions_source="actual source object-local mesh bounds", family_source="inferred from benchmark mesh")

def prepare_candidate(model_path, qpos, destination, *, method="unknown", selection=None):
    """Copy actual object-local triangles and the hand's relative pose to a case.

    Palm FK rather than fixed qpos offsets makes this bridge independent of the
    source root representation. Joint identity follows the official WUJI model.
    """
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation
    from retarget.env.geometry import ObjectMeshSurface
    from .hand import HAND_BODIES, model_hand

    model_path = Path(model_path).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model_hand(model) != "wuji2":
        raise ValueError("unsupported_hand: common GraspADMM evaluation currently supports wuji2")
    data = mujoco.MjData(model)
    values = np.asarray(qpos, float).reshape(-1)
    if values.shape == (26,) and model.nq == 33:
        data.qpos[:] = model.qpos0
        data.qpos[:26] = values
    elif values.shape == (model.nq,):
        data.qpos[:] = values
    else:
        raise ValueError(f"Candidate qpos {values.shape} does not match source nq={model.nq}")
    if not np.isfinite(data.qpos).all():
        raise ValueError("Final grasp has nonfinite coordinates")
    mujoco.mj_kinematics(model, data)
    palm = model.body("right_palm_link").id
    obj = model.body("grasp_object").id
    rotation = data.xmat[obj].reshape(3, 3)
    relative_position = rotation.T @ (data.xpos[palm] - data.xpos[obj])
    relative_rotation = rotation.T @ data.xmat[palm].reshape(3, 3)
    fingers = []
    for body_name in HAND_BODIES[1:]:
        body = model.body(body_name).id
        if model.body_jntnum[body] != 1:
            raise ValueError(f"Expected one WUJI hinge on {body_name}")
        fingers.append(data.qpos[model.jnt_qposadr[model.body_jntadr[body]]])
    grasp = np.r_[relative_position, Rotation.from_matrix(relative_rotation).as_quat(scalar_first=True), fingers]
    surface = ObjectMeshSurface.from_model(model, obj)
    vertices, faces = np.unique(surface.triangles.reshape(-1, 3), axis=0, return_inverse=True)
    actual_primitive = measure_primitive(vertices)
    mesh = destination / "object.obj"
    trimesh.Trimesh(vertices=vertices, faces=faces.reshape(-1, 3), process=False).export(mesh)
    candidate = destination / "candidate.npz"
    np.savez_compressed(candidate, grasp_qpos=grasp)
    identity = hashlib.sha256()
    for array in (grasp, vertices, faces):
        identity.update(np.asarray(array).tobytes())
    required = infer_required_contacts(model, model_path, selection or {})
    identity.update(json.dumps(required, sort_keys=True).encode())
    metadata = dict(version=1, protocol=PROTOCOL, method=method, hand="wuji2", required_cbody=required, actual_primitive=actual_primitive,
        candidate_sha256=identity.hexdigest(), source_model=str(model_path),
        source_qpos=values.tolist(), source_object_pose=np.r_[data.xpos[obj], data.xquat[obj]].tolist(),
        source_selection=selection or {}, template=infer_template(model_path,selection), mesh=str(mesh), candidate=str(candidate),
        pose_frame="final hand relative to final object; xyz+wxyz+20 official WUJI hinges",
        original_controls_reused=False, original_object_mass_kg=float(model.body_mass[obj]))
    write_json(destination / "candidate.json", metadata)
    return metadata



class NativeEvaluationWorker:
    """Reuse one isolated native interpreter across a historical sweep.

    Each item still constructs fresh native models/state and runs the same
    protocol. The subprocess boundary keeps pinned MuJoCo separate from the
    newer source-model reader. A timed-out/crashed worker is restarted.
    """
    def __init__(self, log_path, *, command=None):
        self.log_path=Path(log_path);self.command=command
        self.process=None;self.log=None;self.queue=None

    def _start(self):
        import queue
        import threading
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.log_path.open("a")
        self.queue=queue.Queue()
        command=self.command or [evaluation_python(),"-m",__name__,"--worker"]
        self.process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,bufsize=1)
        stream=self.process.stdout;messages=self.queue
        def read():
            for line in stream:
                messages.put(line)
            messages.put(None)
        threading.Thread(target=read,daemon=True).start()

    def evaluate(self,directory,*,timeout=600.):
        import queue
        directory=Path(directory).resolve();started=time.perf_counter()
        if self.process is None or self.process.poll() is not None:
            self.close();self._start()
        try:
            self.process.stdin.write(json.dumps(dict(candidate=str(directory)))+"\n")
            self.process.stdin.flush()
            while True:
                remaining=timeout-(time.perf_counter()-started)
                if remaining<=0:
                    raise TimeoutError("Persistent native evaluation timed out")
                line=self.queue.get(timeout=remaining)
                if line is None:
                    raise RuntimeError(f"Native worker exited {self.process.poll()}; see {self.log_path}")
                if line.startswith("GRASPADMM_RESULT "):
                    response=json.loads(line[len("GRASPADMM_RESULT "):])
                    if response.get("candidate")!=str(directory):
                        raise RuntimeError("Native worker returned a different candidate")
                    report=response["report"]
                    break
                self.log.write(line);self.log.flush()
        except (OSError,RuntimeError,TimeoutError,queue.Empty) as error:
            report=dict(protocol=PROTOCOL,evaluation_protocol=PROTOCOL,state="failed",success=False,
                task_success=False,failure_stage="evaluation_runtime",error=f"{type(error).__name__}: {error}",criterion=CRITERION)
            self.close()
        report["evaluation_wall_seconds"]=time.perf_counter()-started
        report["runtime_mode"]="persistent_native_worker"
        write_json(directory/"evaluation.json",report)
        (directory/"evaluation.log").write_text(f"Persistent native worker log: {self.log_path}\n")
        return report

    def close(self):
        process,self.process=self.process,None
        if process is not None:
            if process.stdin:
                try:process.stdin.close()
                except OSError:pass
            try:process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=2.)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            if process.stdout:process.stdout.close()
        if self.log:self.log.close();self.log=None

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


def native_worker():
    """Line protocol used only by NativeEvaluationWorker, with fresh state per job."""
    import gc
    import traceback
    for line in sys.stdin:
        directory=None
        try:
            request=json.loads(line);directory=Path(request["candidate"]).resolve()
            report=native_evaluate(directory)
        except BaseException as error:
            traceback.print_exc(file=sys.stderr)
            report=dict(protocol=PROTOCOL,evaluation_protocol=PROTOCOL,state="failed",success=False,
                task_success=False,failure_stage="evaluation_runtime",error=f"{type(error).__name__}: {error}",criterion=CRITERION)
            if directory is not None:write_json(directory/"evaluation.json",report)
        report["worker_pid"]=os.getpid()
        print("GRASPADMM_RESULT "+json.dumps(dict(candidate=str(directory),report=report)),flush=True)
        gc.collect()

def evaluate_candidate(candidate_directory, *, timeout=600.):
    directory = Path(candidate_directory).resolve()
    result_path = directory / "evaluation.json"
    started = time.perf_counter()
    command = [evaluation_python(), "-m", __name__, "--native", "--candidate", str(directory)]
    try:
        with (directory / "evaluation.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        if not result_path.is_file():
            raise RuntimeError(f"Native evaluator exited {result.returncode}; see {directory/'evaluation.log'}")
        report = json.loads(result_path.read_text())
        if result.returncode and report.get("state") != "failed":
            raise RuntimeError(f"Native evaluator exited {result.returncode}")
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        report = dict(protocol=PROTOCOL, evaluation_protocol=PROTOCOL, state="failed", success=False, task_success=False,
            failure_stage="evaluation_runtime", error=f"{type(error).__name__}: {error}",
            criterion=CRITERION)
        write_json(result_path, report)
    report["evaluation_wall_seconds"] = time.perf_counter() - started
    report["runtime_mode"]="isolated_subprocess"
    write_json(result_path, report)
    return report


def evaluate_grasp(model_path, qpos, destination, *, method="unknown", selection=None, timeout=600.):
    metadata = prepare_candidate(model_path, qpos, destination, method=method, selection=selection)
    report = evaluate_candidate(destination, timeout=timeout)
    report["candidate_sha256"] = metadata["candidate_sha256"]
    write_json(Path(destination)/"evaluation.json", report)
    return report



def evaluate_native_grasp(grasp_qpos, mesh_path, destination, *, method="graspadmm",
                          required_cbody=None, selection=None, timeout=600., worker=None):
    """Evaluate an already normalized free-root candidate from GraspADMM."""
    import shutil
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    grasp=np.asarray(grasp_qpos,float).reshape(27)
    if not np.isfinite(grasp).all() or np.linalg.norm(grasp[3:7])<1e-12:
        raise ValueError("Expected a finite xyz+wxyz+20-joint candidate")
    mesh=destination/"object.obj";shutil.copy2(Path(mesh_path),mesh)
    np.savez_compressed(destination/"candidate.npz",grasp_qpos=grasp)
    required=required_cbody or []
    digest=hashlib.sha256(grasp.tobytes()+mesh.read_bytes()+json.dumps(required,sort_keys=True).encode()).hexdigest()
    write_json(destination/"candidate.json",dict(version=1,protocol=PROTOCOL,method=method,hand="wuji2",
        required_cbody=required,candidate_sha256=digest,mesh=str(mesh),
        candidate=str(destination/"candidate.npz"),source_selection=selection or {},
        original_controls_reused=False,pose_frame="final hand relative to final object; xyz+wxyz+20 official WUJI hinges"))
    if worker is not None:
        report=worker.evaluate(destination,timeout=timeout)
    elif method=="graspadmm" and pinned_native_environment():
        import contextlib
        import traceback
        started=time.perf_counter()
        with (destination/"evaluation.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                report=native_evaluate(destination)
            except Exception as error:
                traceback.print_exc()
                report=dict(protocol=PROTOCOL,evaluation_protocol=PROTOCOL,state="failed",success=False,
                    task_success=False,failure_stage="evaluation_runtime",error=f"{type(error).__name__}: {error}",criterion=CRITERION)
        report["evaluation_wall_seconds"]=time.perf_counter()-started
        report["runtime_mode"]="in_process_pinned"
    else:
        report=evaluate_candidate(destination,timeout=timeout)
    report["candidate_sha256"]=digest
    write_json(destination/"evaluation.json",report)
    return report

def apply_evaluation(summary, model_path, qpos, case_directory, *, method, selection=None):
    """Make the common native protocol the sole final success verdict."""
    previous = dict(task_success=summary.get("task_success"), acceptance=summary.get("acceptance"))
    try:
        report = evaluate_grasp(model_path, qpos, Path(case_directory)/"graspadmm_evaluation",
                                method=method, selection=selection)
    except (ValueError, OSError) as error:
        report = dict(protocol=PROTOCOL, evaluation_protocol=PROTOCOL, state="failed",
            success=False, task_success=False, failure_stage="candidate_extraction", error=str(error), criterion=CRITERION)
        write_json(Path(case_directory)/"graspadmm_evaluation/evaluation.json",report)
    summary.update(method_original_acceptance=previous, evaluation_protocol=PROTOCOL,
        acceptance=CRITERION, graspadmm_evaluation=report,
        task_success=bool(report.get("success")),
        physical_hold_tested=bool(report.get("six_direction_tested", False)))
    return summary


def native_evaluate(candidate_directory):
    """Run only byte-verified upstream filtering/trajectory/six-axis evaluation."""
    from .graspadmm_scene import write_scene, native_finalize, WujiEvalEnv, require_upstream, normalize_grasp_data
    source = require_upstream()
    from dexonomy.sim import HandCfg, MuJoCo_EvalCfg
    from dexonomy.util.traj_util import get_planner

    directory = Path(candidate_directory).resolve()
    metadata = json.loads((directory/"candidate.json").read_text())
    grasp = _arrays(directory/"candidate.npz")["grasp_qpos"].reshape(27)
    scene = write_scene(directory/"native_scene", metadata["mesh"], object_mass=.1)
    grasp_data = normalize_grasp_data(grasp, scene, required_cbody=metadata.get("required_cbody", []))
    data, status, trace = native_finalize(grasp_data, scene, settings=None)
    report = dict(protocol=PROTOCOL, evaluation_protocol=PROTOCOL, criterion=CRITERION, generation=status,
                  upstream_commit=source.manifest["commit"],
                  generation_config=source.OmegaConf.to_container(source.grasp_config,resolve=False),
                  eval_config=source.OmegaConf.to_container(source.eval_config,resolve=False),
                  state="completed", success=False, task_success=False)
    if not status.get("success", False):
        report["failure_stage"] = status.get("failure_stage", "grasp_or_pregrasp_filter")
        write_json(directory/"evaluation.json", report)
        return report
    env = WujiEvalEnv(hand_cfg=HandCfg(str(scene.hand_xml), freejoint=True),
        scene_cfg=scene.scene_cfg, sim_cfg=MuJoCo_EvalCfg(miu_coef=[.6, .02]),
        debug_render=False, debug_view=False, hard=False)
    planner = get_planner(dict(scene.scene_cfg["task"]))
    ctrl, ctrl_type, ext_dir, target = planner.plan_trajectory(
        env.get_active_obj_pose().copy(), data["pregrasp_qpos"], data["grasp_qpos"],
        data["squeeze_qpos"], None)
    recorder = record_native_evaluation(env, target)
    success, _, _ = env.eval_fc(ctrl, ctrl_type, ext_dir, target, .05, 15.)
    states, controls = recorder["qpos"], recorder["ctrl"]
    report.update(success=bool(success), task_success=bool(success), six_direction_tested=True,
        directions=recorder["directions"],
        trajectory_time_scope="time is strictly increasing display time; reset frames advance by one float unit; sim_time is actual resettable MuJoCo time",
        failure_stage=None if success else "six_direction_force_test",
        scene=str(directory/"evaluation_scene.xml"), trajectory=str(directory/"evaluation_trajectory.npz"))
    (directory/"evaluation_scene.xml").write_text(env._spec.to_xml())
    np.savez_compressed(directory/"evaluation_trajectory.npz", qpos=np.asarray(states),
        ctrl=np.asarray(controls), time=np.asarray(recorder["time"]), sim_time=np.asarray(recorder["sim_time"]),
        direction_index=np.asarray(recorder["direction_index"]), grasp_qpos=data["grasp_qpos"],
        squeeze_qpos=data["squeeze_qpos"], pregrasp_qpos=data["pregrasp_qpos"])
    write_json(directory/"evaluation.json", report)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return native_worker()
    if not args.native or args.candidate is None:
        parser.error("Use scripts/evaluate_grasp.py for public input extraction")
    try:
        report = native_evaluate(args.candidate)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        report = dict(protocol=PROTOCOL, evaluation_protocol=PROTOCOL, criterion=CRITERION, state="failed", success=False,
            task_success=False, failure_stage="evaluation_runtime", error=f"{type(error).__name__}: {error}")
        write_json(args.candidate/"evaluation.json", report)
        raise
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
