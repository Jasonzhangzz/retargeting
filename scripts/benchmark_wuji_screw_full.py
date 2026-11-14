"""Finite full-trajectory EMF->q timing, one backend per process; no training."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", required=True)
    args = ap.parse_args()
    root, out, method = args.root, args.output, args.method
    out.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    data = dict(np.load(root / "inputs/trajectory_0_right.npz", allow_pickle=False))
    offsets = data["segment_offsets"].tolist()
    assert len(data["position"]) == 1133 and offsets == [0, 1133]
    records = json.loads((root / "selection.json").read_text())["records"]
    sn = records[0]["metadata"]["devices"]["right"]["sn"]
    np.random.seed(42)
    timings, cold, prediction, raws = [], None, None, None

    def allowed():
        parent = json.loads((root / "CONTROL.json").read_text())
        local = json.loads((out.parent / "CONTROL.json").read_text())
        if parent.get("stop_requested") or local.get("stop_requested"):
            raise InterruptedError("Stop requested")

    if method == "sdk":
        import sys
        sys.path.insert(0, str(root / "sdk831"))
        import wuji_sdk as sdk
        sdk.set_log_level("error")
        assert sdk.__version__ == "2026.8.31"
        names = ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip") + tuple(f+"_"+j for f in ("index_finger", "middle_finger", "ring_finger", "pinky") for j in ("mcp", "pip", "dip", "tip"))
        setup_s = time.perf_counter()-setup_start
        for rep in range(12):
            allowed()
            begin = time.perf_counter()
            pipeline = sdk.WujiGlove.offline_pipeline(sn=sn, hand_side="right")
            session = sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2, side=sdk.Handedness.Right)
            init_end = time.perf_counter()
            objects_s = human_s = robot_s = 0.
            qs = []
            for i in range(1133):
                a = time.perf_counter()
                frame = sdk.EmfPoseArray(header=sdk.FrameHeader(seq=int(data["source_seq"][i]), timestamp_us=int(data["timestamp_ns"][i]//1000), frame_id="r_hand_emf_tx"), poses=[sdk.EmfPose(pose=sdk.Pose(position=data["position"][i,j].tolist(), orientation=sdk.Quaternion(*data["quaternion"][i,j].tolist())), confidence=float(data["confidence"][i,j])) for j in range(5)])
                b = time.perf_counter()
                skeleton = pipeline.hand_skeleton().compute(frame)
                by = {v.name:v for v in skeleton.joints}
                points = np.array([by[k].pose.position for k in names], np.float32)
                c = time.perf_counter()
                qs.append(session.step(points))
                d = time.perf_counter()
                objects_s += b-a; human_s += c-b; robot_s += d-c
            prediction = np.asarray(qs)
            end = time.perf_counter()
            row = dict(total=end-begin, initialize=init_end-begin, emf_objects=objects_s, human_and_targets=human_s, robot_and_filter=robot_s)
            row["python_loop_and_output"] = row["total"]-sum(row[k] for k in ("initialize", "emf_objects", "human_and_targets", "robot_and_filter"))
            if rep == 0: cold = row
            if rep >= 2: timings.append(row)
            atomic(out / "STATUS.json", dict(status="running", method=method, repeat=rep+1, neural_optimizer_updates=0))
            print("REPEAT",method,rep,json.dumps(row),flush=True)
        expected = np.load(root / "sdk/builtin/trajectory_0_right.npz")["q_sdk_filtered"]
        drift = float(np.abs(prediction-expected).max())
        assert np.array_equal(prediction, expected), drift
        report = dict(method=method, frames=1133, setup_s=setup_s, cold_first=cold, timings=timings, output_max_vs_cached_rad=drift, complete=True, neural_optimizer_updates=0)
    else:
        import torch
        import warp as wp
        from retarget.learning.wuji_ik.training import load_inference
        from retarget.learning.wuji_ik.trajectory_pipeline import EMFTrajectorySolver, quaternion_matrix, CompiledSpatial
        from retarget.learning.wuji_ik.trajectory_oneway import CausalHumanRollout, FrozenHistoryRobot
        from retarget.learning.wuji_ik.batched_lm import RobotRetargetObjective, prepare_robot_targets
        from retarget.learning.wuji_ik.trajectory import TrajectoryObjective, robot_diagnostics
        torch.set_grad_enabled(False); torch.set_num_threads(2); torch.manual_seed(42)
        torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch._inductor.config.compile_threads = 2
        torch._dynamo.config.recompile_limit = 128
        wp.config.kernel_cache_dir = "/dev/shm/wuji_lm_runtime_2207_20261111/anchor_warp_cache"
        model, normalizer, state = load_inference(Path("/dev/shm/wuji_lm_runtime_2207_20261111/ik_distilled_715516_latest_lm.pt"), torch.device("cuda"), backend="flex")
        assert state["variant"] == "temporal_causal"
        solver = EMFTrajectorySolver(root/"reference", root/"assets/builtin_right.urdf", "right", device="cuda", model=model, normalizer=normalizer)
        pos, quat = data["position"], data["quaternion"]
        frozen_state = {}

        def solve_network_history():
            # Same equations and engine as the existing frozen_history_network
            # benchmark, now including fresh network and human reconstruction.
            solver.sync(); begin = time.perf_counter()
            initial = solver.network(pos, quat, offsets)
            solver.sync(); nn_end = time.perf_counter()
            hp = solver.tensor(pos).float().double()
            hr = quaternion_matrix(solver.tensor(quat).float().double())
            if "human" not in frozen_state:
                frozen_state["human"] = CausalHumanRollout(solver, hp, hr, offsets)
            hq = frozen_state["human"](hp, hr)
            target = prepare_robot_targets(solver.sfk.forward(hq)[0], solver.alignment)
            if "objects" not in frozen_state:
                objects = [TrajectoryObjective(CompiledSpatial(RobotRetargetObjective(solver.rfk,*target,stage=s), True), offsets, solver.tensor(solver.robot.bounds.mean(1)).expand(1,-1), .01) for s in (1,2)]
                frozen_state["objects"] = objects
                frozen_state["engine"] = FrozenHistoryRobot(objects,initial,solver.rfk.lower,solver.rfk.upper,seed_mode="network")
            else:
                for s,obj in enumerate(frozen_state["objects"],1):obj.spatial.raw.copy_targets_(RobotRetargetObjective(solver.rfk,*target,stage=s))
            solver.sync(); human_end = time.perf_counter()
            q, filtered = frozen_state["engine"](initial)
            solver.sync(); robot_end = time.perf_counter()
            robot_diagnostics(frozen_state["objects"][1].spatial,q,offsets,frozen_state["objects"][1].reference,causal=True,lower=solver.rfk.lower,upper=solver.rfk.upper)
            return q, filtered, dict(network=nn_end-begin,human_and_targets=human_end-nn_end,robot=robot_end-human_end,filter=0.,filter_fused_into_robot=True)

        solver.sync(); setup_s = time.perf_counter()-setup_start
        drift_repeat = 0.
        for rep in range(12):
            allowed(); solver.sync(); begin = time.perf_counter()
            if method == "network":
                q = filtered = solver.network(pos,quat,offsets)
                solver.sync(); internal = dict(network=time.perf_counter()-begin)
            elif method == "frozen_history_network":
                q, filtered, internal = solve_network_history()
                internal.pop("filter_fused_into_robot")
            else:
                backend = "banded_lm" if method == "joint_lm" else ("frozen_history" if method == "frozen_history_previous" else method)
                result = solver.solve(pos,quat,offsets,backend=backend,iterations=32,stage1_iterations=4)
                q, filtered, internal = result.q_raw, result.q_filtered, dict(result.timings)
                internal.pop("total",None)
            solver.sync(); core_end = time.perf_counter()
            prediction = filtered.cpu().numpy(); raws = q.cpu().numpy()
            end = time.perf_counter()
            row = dict(total=end-begin, output_to_cpu=end-core_end, **internal)
            row["api_overhead_and_diagnostics"] = core_end-begin-sum(internal.values())
            if rep == 0: cold = row
            if rep == 1: reference_q = prediction.copy()
            if rep >= 2:
                timings.append(row); drift_repeat=max(drift_repeat,float(np.abs(prediction-reference_q).max()))
            atomic(out / "STATUS.json", dict(status="running", method=method, repeat=rep+1, neural_optimizer_updates=0))
            print("REPEAT",method,rep,json.dumps(row),flush=True)
        c = np.load(root/"oneway_comparison/builtin_right/curves.npz")
        if method == "joint_lm":
            expected = np.load(root/"jobs/development_banded_lm_builtin_right/budget_32/trajectory_0_right.npz")["q_filtered"]
        else:
            key = "q_network" if method == "network" else "source_causal_human_from_emf__"+method+"__"+("10" if method.startswith("frozen_history") else "32")
            expected = c[key][:1133]
        drift = float(np.abs(prediction-expected).max())
        assert np.isfinite(prediction).all() and np.isfinite(raws).all()
        assert drift_repeat <= 1e-4, drift_repeat
        report = dict(method=method,frames=1133,setup_s=setup_s,cold_first=cold,timings=timings,
            output_max_vs_cached_rad=drift,repeat_max_drift_rad=drift_repeat,
            cached_output_within_1e4=drift<=1e-4,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name(0),
            checkpoint_step=state["step"],complete=True,neural_optimizer_updates=0,
            filter_fused_into_robot=method.startswith("frozen_history"))
    report["median_s"] = {k:float(np.median([t[k] for t in timings])) for k in timings[0]}
    report["minimum_s"] = float(min(t["total"] for t in timings))
    report["maximum_s"] = float(max(t["total"] for t in timings))
    report["throughput_frames_s"] = 1133/report["median_s"]["total"]
    np.savez_compressed(out / "output.npz",q_filtered=prediction,**({"q_raw":raws} if raws is not None else {}))
    atomic(out/"result.json",report)
    atomic(out/"STATUS.json",dict(status="complete",method=method,neural_optimizer_updates=0))
    print("COMPLETE",method,json.dumps(report["median_s"]),flush=True)


if __name__ == "__main__":
    main()
