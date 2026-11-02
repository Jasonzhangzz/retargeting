"""Aggregate saved grasp experiments and plot SPIDER diagnostics on CPU.

Run after the experiment writers finish::

    .venv-contact/bin/python scripts/analyze_spider_grasp.py

No simulator, Torch, Warp, or GPU is loaded. ForceAware contributes measured
summary metrics only; the figure contains recorded SPIDER trajectories/forces.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "little")
FINGER_COLORS = ("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#7B61A8")
RADII = (.02625, .035, .04375)
LOAD_THRESHOLD_N = 1e-6
COMPARISON_SCOPE = (
    "End-to-end MPPI/native MuJoCo Warp versus ForceAware/native ComFree; "
    "different dynamics engines and search budgets, not an isolated optimizer ablation. "
    "CPU mesh replay is reported separately from each native execution."
)


class Sources:
    """Hash the exact file bytes used for each result, including external runs."""

    def __init__(self):
        self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        entry = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                     bytes=len(content))
        previous = self.files.setdefault(str(path), entry)
        if previous != entry:
            raise RuntimeError(f"Input changed during aggregation: {path}")
        return content, entry

    def json(self, path):
        raw, entry = self.read(path)
        return json.loads(raw), entry

    def npz(self, path, names):
        raw, entry = self.read(path)
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            return {name: arrays[name] for name in names if name in arrays}, entry

    def optional(self, path):
        return self.read(path)[1] if Path(path).is_file() else None


@dataclass
class SpiderCurves:
    radius: float
    time_gpu: np.ndarray
    drift_gpu: np.ndarray
    time_cpu: np.ndarray
    drift_cpu: np.ndarray
    forces: np.ndarray | None
    fingers: tuple


def radius_from(*values):
    for value in values:
        match = re.search(r"(?:radius_|cylinder_r)([0-9]+(?:\.[0-9]+)?)", str(value))
        if match:
            return float(match.group(1))
    raise ValueError(f"Cannot identify primitive radius: {values}")


def stats(values, *, exclude_initial=False):
    values = np.asarray(values, dtype=float)
    mean_values = values[1:] if exclude_initial else values
    if not len(values) or not len(mean_values) or not np.isfinite(values).all():
        return dict(mean=None, max=None, terminal=None, finite=False)
    return dict(mean=float(mean_values.mean()), max=float(values.max()),
                terminal=float(values[-1]), finite=True)


def tracking(qpos, reference_object):
    """Object is the final free joint: XYZ followed by WXYZ, exactly as saved."""
    qpos = np.asarray(qpos, dtype=float)
    reference = np.asarray(reference_object, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] != 33 or len(qpos) < 2 or reference.shape != (7,):
        raise ValueError("Expected WUJI2 qpos[states,33] and static object reference[7]")
    drift = np.linalg.norm(qpos[:, -7:-4] - reference[:3], axis=1) * 1000.
    quat, target = qpos[:, -4:], reference[3:]
    denominator = np.linalg.norm(quat, axis=1) * np.linalg.norm(target)
    with np.errstate(divide="ignore", invalid="ignore"):
        dot = np.abs(np.sum(quat * target, axis=1) / denominator)
        rotation = np.rad2deg(2. * np.arccos(np.clip(dot, 0., 1.)))
    rotation[denominator <= 1e-12] = np.nan
    return drift, dict(position_mm=stats(drift, exclude_initial=True),
                       rotation_deg=stats(rotation, exclude_initial=True),
                       mean_excludes_initial_state=True)


def hand_deviation(qpos):
    delta = np.asarray(qpos[1:, :26], dtype=float) - qpos[0, :26]
    return dict(reference="trajectory qpos[0,:26], the accepted CA hand configuration",
        wrist_translation_l2_mm=stats(np.linalg.norm(delta[:, :3], axis=1)*1000.),
        wrist_three_scalar_angle_l2_deg=stats(np.rad2deg(np.linalg.norm(delta[:, 3:6], axis=1))),
        wrist_angle_convention="L2 of three scalar wrist-coordinate differences; not a physical rotation geodesic",
        finger_twenty_joint_l2_deg=stats(np.rad2deg(np.linalg.norm(delta[:, 6:26], axis=1))),
        finger_twenty_joint_max_abs_deg=stats(np.rad2deg(np.max(np.abs(delta[:, 6:26]), axis=1))))


def state_time(qpos, time, source):
    qpos, time = np.asarray(qpos), np.asarray(time, dtype=float)
    if (time.shape != (len(qpos),) or len(time) < 2
            or not np.isfinite(time).all() or not np.all(np.diff(time) > 0)):
        raise ValueError(f"Invalid increasing state time in {source}")
    return qpos, time


def loads(values, *, names=FINGERS, threshold=LOAD_THRESHOLD_N):
    if values is None:
        return dict(available=False, reason="No recorded actual finger-force array")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5 or not len(values) or len(names) != 5:
        raise ValueError("Expected actual finger normal loads[physics_steps,5]")
    finite = bool(np.isfinite(values).all())
    positive = np.isfinite(values) & (values > threshold)
    return dict(available=True, finite=finite, physics_steps=len(values),
                finger_names=list(names), positive_load_threshold_n=threshold,
                zero_load_definition="load <= threshold, including nonfinite observations",
                finger_normal_force_min_n=values.min(0).tolist() if finite else None,
                finger_normal_force_mean_n=values.mean(0).tolist() if finite else None,
                finger_normal_force_max_n=values.max(0).tolist() if finite else None,
                finger_zero_load_steps=(~positive).sum(0).tolist(),
                finger_positive_load_fraction=positive.mean(0).tolist(),
                finger_continuously_positive=positive.all(0).tolist(),
                all_fingers_continuously_positive=bool(finite and positive.all()))


def strict_backend(metrics, *, warnings_zero=True):
    """Common comparison criterion; original run acceptance is retained separately."""
    position, rotation = metrics.get("position_mm", {}), metrics.get("rotation_deg", {})
    if (position.get("max") is None or rotation.get("max") is None
            or not metrics.get("available", False)):
        return None
    return bool(position["max"] <= 2. and rotation["max"] <= 5.
                and metrics.get("finite", False)
                and metrics.get("all_fingers_continuously_positive", False)
                and warnings_zero)


def spider_cases(root, label, sources, warnings):
    root = Path(root).resolve()
    if not root.is_dir():
        warnings.append(f"Skipped absent {label} run: {root}")
        return dict(root=str(root), available=False, cases=[]), {}
    run_sources = {name: sources.optional(root / name)
                   for name in ("summary.json", "config.yaml", "manifest.json", "provenance.json")}
    cases, curves = [], {}
    candidates = [root] if (root / "trajectory.npz").is_file() else sorted(root.iterdir())
    for case in candidates:
        if not case.is_dir() or not (case / "reference_manifest.json").is_file():
            continue
        required = [case / "summary.json", case / "trajectory.npz", case / "replay/replay.npz"]
        if not all(path.is_file() for path in required):
            warnings.append(f"Skipped unfinished {label} case: {case}")
            continue
        report, report_source = sources.json(required[0])
        data, trajectory_source = sources.npz(required[1], ("qpos", "time", "finger_normal_force_n"))
        replay, replay_source = sources.npz(required[2], ("cpu_qpos", "time_cpu", "cpu_finger_normal_force_n"))
        gpu_qpos, gpu_time = state_time(data["qpos"], data["time"], required[1])
        cpu_qpos, cpu_time = state_time(replay["cpu_qpos"], replay["time_cpu"], required[2])
        reference = np.asarray(gpu_qpos[0, -7:], dtype=float)
        gpu_drift, gpu = tracking(gpu_qpos, reference)
        cpu_drift, cpu = tracking(cpu_qpos, reference)
        gpu["hand_reference_deviation"] = hand_deviation(gpu_qpos)
        cpu["hand_reference_deviation"] = hand_deviation(cpu_qpos)
        names = tuple(report.get("gpu", {}).get("finger_names", FINGERS))
        threshold = float(report.get("acceptance", {}).get("minimum_each_finger_normal_force_n", LOAD_THRESHOLD_N))
        force = data.get("finger_normal_force_n")
        cpu_force = replay.get("cpu_finger_normal_force_n")
        for values, times in ((force, gpu_time), (cpu_force, cpu_time)):
            if values is not None and len(values) != len(times) - 1:
                raise ValueError(f"Actual force count must match every physics step: {case}")
        gpu.update(loads(force, names=names, threshold=threshold))
        cpu.update(loads(cpu_force, names=tuple(report.get("cpu", {}).get("finger_names", names)), threshold=threshold))
        gpu["force_source"] = "Recorded native MuJoCo Warp contact_force, grouped by finger"
        cpu["force_source"] = "Independent replay mujoco.mj_contactForce, grouped by finger"
        gpu["strict_success"] = strict_backend(gpu)
        cpu["strict_success"] = strict_backend(cpu, warnings_zero=report.get("mujoco_warnings_zero", False))
        radius = radius_from(case.name, report.get("source_experiment", ""))
        if radius in curves:
            raise ValueError(f"Duplicate radius in {label}: {radius}")
        curves[radius] = SpiderCurves(radius, gpu_time, gpu_drift, cpu_time, cpu_drift, force, names)
        cases.append(dict(name=case.name, radius_m=radius, case=str(case),
            duration_seconds=float(gpu_time[-1] - gpu_time[0]), physics_steps=len(gpu_time)-1,
            reference_object_qpos=reference.tolist(),
            reference_definition="trajectory.npz qpos[0,-7:] (accepted static CA object pose)",
            timing=report.get("timing", {}), sampler=report.get("sampler", {}), profile=report.get("profile", {}),
            gpu=gpu, cpu=cpu, reported_task_success=report.get("task_success"),
            strict_success=bool(gpu["strict_success"] is True and cpu["strict_success"] is True
                                and report.get("finite", False)),
            hardware_tested=False, comparison_scope=report.get("comparison_scope", COMPARISON_SCOPE),
            sources=dict(summary=report_source, trajectory=trajectory_source, cpu_replay=replay_source,
                         replay_report=sources.optional(case / "replay/replay.json"),
                         reference_manifest=sources.optional(case / "reference_manifest.json"),
                         input_hashes=sources.optional(case / "ca_input_hashes.json"))))
    return dict(root=str(root), available=bool(cases), sources=run_sources, cases=cases), curves


def static_cases(root, sources, warnings):
    root = Path(root).resolve()
    if not (root / "summary.json").is_file():
        warnings.append(f"Skipped absent static CPU control baseline: {root}")
        return dict(root=str(root), available=False, cases=[]), {}
    aggregate, aggregate_source = sources.json(root / "summary.json")
    cases, curves = [], {}
    for item in aggregate.get("cases", []):
        case = root / item["case"]
        if not all((case / name).is_file() for name in ("summary.json", "trajectory.npz")):
            warnings.append(f"Skipped unfinished static CPU baseline: {case}")
            continue
        report, report_source = sources.json(case / "summary.json")
        data, trajectory_source = sources.npz(case / "trajectory.npz", ("qpos", "time", "finger_normal_force_n"))
        qpos, time = state_time(data["qpos"], data["time"], case)
        drift, cpu = tracking(qpos, np.asarray(qpos[0, -7:], dtype=float))
        cpu.update(loads(data.get("finger_normal_force_n"), names=tuple(report.get("cpu", {}).get("finger_names", FINGERS))))
        cpu["hand_reference_deviation"] = hand_deviation(qpos)
        cpu["force_source"] = "Actual native CPU mujoco.mj_contactForce, grouped by finger"
        cpu["strict_success"] = strict_backend(cpu, warnings_zero=not any(report.get("mujoco_warning_counts", [1])))
        radius = radius_from(case.name)
        curves[radius] = SpiderCurves(radius, time, drift, time, drift,
                                     data.get("finger_normal_force_n"), tuple(cpu["finger_names"]))
        cases.append(dict(name=case.name, radius_m=radius, case=str(case.resolve()),
            duration_seconds=float(time[-1]-time[0]), physics_steps=len(time)-1,
            optimization_performed=False, timing=dict(optimization_seconds=0.,
                source="No optimization: constant saved QP preload command"),
            cpu=cpu, strict_success=cpu["strict_success"], reported_task_success=report.get("task_success"),
            hardware_tested=False,
            sources=dict(aggregate_summary=aggregate_source, summary=report_source, trajectory=trajectory_source,
                model=sources.optional(report["model_xml"]), command=sources.optional(report["command_source"])) ))
    return dict(root=str(root), available=bool(cases), cases=cases), curves


def reported_force_metrics(audit):
    keys = ("finger_names", "positive_load_threshold_n", "finger_normal_force_min_n",
            "finger_normal_force_mean_n", "finger_normal_force_max_n", "finger_zero_load_steps",
            "finger_positive_load_fraction", "finger_continuously_positive", "all_fingers_continuously_positive")
    metrics = {name: audit[name] for name in keys if name in audit}
    metrics.update(available=bool(audit), finite=audit.get("finite", False))
    metrics["force_source"] = audit.get("force_source", "Actual mujoco.mj_contactForce in independent mesh replay")
    return metrics


def forceaware_cases(root, label, sources, warnings):
    root = Path(root).resolve()
    if not (root / "summary.json").is_file():
        warnings.append(f"Skipped absent {label} summary: {root}")
        return dict(root=str(root), available=False, cases=[])
    aggregate, aggregate_source = sources.json(root / "summary.json")
    cases = []
    for report in aggregate.get("cases", []):
        name = report["name"]
        # The FA200 report aggregates runs stored in multiple output roots.
        case = Path(report.get("source_case", root / name)).resolve()
        case_sources = dict(aggregate_summary=aggregate_source,
                            case_summary=sources.optional(case / "summary.json"))
        native_audit = report.get("physical_contact_audit", {})
        cpu_audit = report.get("independent_mujoco_contact_audit", {})
        native_path = case / "contact_audit/native_replay.npz"
        native = dict(available=bool(native_audit), finite=native_audit.get("finite", False))
        if native_path.is_file():
            data, case_sources["native_contact_replay"] = sources.npz(native_path, ("finger_normal_force_n",))
            native.update(loads(data.get("finger_normal_force_n"), names=tuple(native_audit.get("finger_names", FINGERS))))
        else:
            native.update(reported_force_metrics(native_audit))
            warnings.append(f"{label} native per-step force array absent; zero-load counts may be unavailable: {case}")
        native["force_source"] = native_audit.get("force_source", "Original native physical contact row pressure")
        native["position_mm"] = dict(max=native_audit.get("object_displacement_max_mm",
                                               report.get("native_physics_step_object_drift_max_mm")))
        native_rotation = native_audit.get("object_rotation_max_rad")
        native["rotation_deg"] = dict(max=float(np.rad2deg(native_rotation)) if native_rotation is not None
            else report.get("native_physics_step_object_rotation_max_deg"))
        cpu = reported_force_metrics(cpu_audit)
        cpu["position_mm"] = dict(max=cpu_audit.get("object_displacement_max_mm"))
        cpu_rotation = cpu_audit.get("object_rotation_max_rad")
        cpu["rotation_deg"] = dict(max=float(np.rad2deg(cpu_rotation)) if cpu_rotation is not None else None)
        native["strict_success"] = strict_backend(native)
        warning_counts = cpu_audit.get("mujoco_warning_counts")
        cpu["strict_success"] = strict_backend(cpu, warnings_zero=warning_counts is not None and not any(warning_counts))
        for relative in ("contact_audit/replay.json", "mesh_contact_audit/mesh_contact_audit.json"):
            case_sources[relative] = sources.optional(case / relative)
        cases.append(dict(name=name, radius_m=radius_from(name, case), case=str(case),
            duration_seconds=report.get("executed_duration_seconds"), timing=report.get("timing", {}),
            native=native, cpu=cpu, reported_task_success=report.get("task_success"),
            strict_success=bool(native["strict_success"] is True and cpu["strict_success"] is True),
            hardware_tested=False, curves_included=False, sources=case_sources,
            metric_scope="Native and CPU maxima from actual every-physics-step contact audits, not sparse control knots"))
    run_sources = {name: sources.optional(root / name)
                   for name in ("config.yaml", "source_runs.json", "provenance.json", "manifest.json")}
    return dict(root=str(root), available=bool(cases), sources=run_sources, cases=cases)


def aggregate_timing(run):
    values = [case.get("timing", {}).get("optimization_seconds") for case in run["cases"]]
    measured = [float(value) for value in values if value is not None and np.isfinite(value)]
    return dict(cases_measured=len(measured), mean_optimization_seconds=float(np.mean(measured)) if measured else None,
                timing_scope="Recorded optimization wall time; setup/output/replay excluded where source timing states so",
                missing_timing_cases=len(values)-len(measured))


def aggregate_metrics(run):
    result = dict(completed_cases=len(run["cases"]), strict_successes=sum(case["strict_success"] is True for case in run["cases"]),
                  reported_task_successes=sum(case.get("reported_task_success") is True for case in run["cases"]))
    for backend in ("gpu", "native", "cpu"):
        values = [case[backend] for case in run["cases"] if backend in case]
        if values:
            position = [item["position_mm"]["max"] for item in values if item["position_mm"].get("max") is not None]
            rotation = [item["rotation_deg"]["max"] for item in values if item["rotation_deg"].get("max") is not None]
            result[backend] = dict(mean_case_position_max_mm=float(np.mean(position)) if position else None,
                mean_case_rotation_max_deg=float(np.mean(rotation)) if rotation else None,
                strict_successes=sum(item.get("strict_success") is True for item in values))
    return result


def plot(curves, output, *, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "axes.titlesize": 10, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(3, 2, figsize=(12.5, 11.2), squeeze=False)
        figure.suptitle("SPIDER MPPI / WUJI2: 0.4 s stationary-object simulation", fontsize=14, y=.985)
        for row, radius in enumerate(RADII):
            left, right = axes[row]
            left.set_title(f"Radius {radius * 1000:g} mm: object drift")
            force_label = "tracking" if radius in curves["tracking"] else "fine"
            right.set_title(f"Radius {radius * 1000:g} mm: {force_label} native GPU finger loads")
            for label, color in (("coarse", "#0072B2"), ("fine", "#D55E00"),
                                 ("tracking", "#7B61A8"), ("quality_probe", "#009E73")):
                if label == "quality_probe" and not np.isclose(radius, RADII[1]):
                    continue
                case = curves[label].get(radius)
                if case is None:
                    continue
                display = label.replace("_", " ")
                left.plot(case.time_gpu, case.drift_gpu, color=color, lw=1.7, label=f"{display} GPU")
                left.plot(case.time_cpu, case.drift_cpu, color=color, lw=1.4, ls="--", label=f"{display} CPU replay")
            baseline = curves["static_cpu"].get(radius)
            if baseline is not None:
                left.plot(baseline.time_cpu, baseline.drift_cpu, color="#777777", lw=1.3,
                          ls="-.", label="Constant QP control CPU")
            left.axhline(2., color="#555555", lw=1.1, ls=":", label="2 mm limit")
            force_case = curves[force_label].get(radius)
            if force_case is not None and force_case.forces is not None:
                for finger, name in enumerate(force_case.fingers):
                    # The actual force belongs to the solver step ending at time[1:].
                    right.plot(force_case.time_gpu[1:], force_case.forces[:, finger], color=FINGER_COLORS[finger],
                               lw=1.1, label=name.capitalize())
            else:
                right.text(.5, .5, "No completed native force trace", transform=right.transAxes, ha="center")
            right.axhline(0., color="#444444", lw=.8, label="Zero load")
            for axis in (left, right):
                axis.set_xlabel("Step-end time (s)")
                axis.set_xlim(0., .4)
                axis.grid(True, alpha=.23, lw=.6)
                axis.spines[["top", "right"]].set_visible(False)
                axis.legend(loc="best", fontsize=8, framealpha=.9, ncol=2 if axis is right else 1)
            left.set_ylim(bottom=0.)
            left.set_ylabel("Position error from initial target (mm)")
            right.set_ylabel("Actual normal load (N)")
        figure.text(.5, .018,
            "Simulation only; no hardware test. GPU and independent CPU replay are distinct trajectories.\n"
            "Loads: actual native contact rows per physics step, shown at time[1:]; solver contacts precede post-step state.\n"
            "ForceAware metrics are JSON-only. Comparison changes both dynamics engine and search budget.",
            ha="center", va="bottom", fontsize=8, color="#444444")
        figure.tight_layout(rect=(0., .085, 1., .96), h_pad=1.8, w_pad=2.)
        paths = [output / f"spider_diagnostics.{extension}" for extension in ("png", "svg")]
        for path in paths:
            figure.savefig(path, dpi=dpi, facecolor="white")
        plt.close(figure)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (("coarse", "grasp_contactaware_spider_mppi"),
                          ("fine", "grasp_contactaware_spider_fine"),
                          ("quality-probe", "grasp_contactaware_spider_quality_probe"),
                          ("tracking", "grasp_contactaware_spider_tracking"),
                          ("static-cpu", "grasp_spider_control_baseline"),
                          ("fa40", "grasp_contactaware_forceaware_fast"),
                          ("fa200", "grasp_forceaware_fine_report")):
        parser.add_argument(f"--{flag}", type=Path, default=Path("output") / default)
    parser.add_argument("--output", type=Path, default=Path("output/grasp_spider_comparison"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    sources, warnings, runs, curves = Sources(), [], {}, {}
    for label, root in (("coarse", args.coarse), ("fine", args.fine), ("quality_probe", args.quality_probe),
                        ("tracking", args.tracking)):
        runs[label], curves[label] = spider_cases(root, label, sources, warnings)
    runs["static_cpu"], curves["static_cpu"] = static_cases(args.static_cpu, sources, warnings)
    for label, root in (("fa40", args.fa40), ("fa200", args.fa200)):
        runs[label] = forceaware_cases(root, label, sources, warnings)
    for run in runs.values():
        run["timing_aggregate"] = aggregate_timing(run)
        run["metrics_aggregate"] = aggregate_metrics(run)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = plot(curves, args.output, dpi=args.dpi)
    summary = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
        aggregation_only=True, hardware_tested=False, comparison_scope=COMPARISON_SCOPE,
        metric_conventions=dict(object_reference="Recorded initial object XYZ/WXYZ; static accepted CA target",
            position_unit="mm", rotation_unit="degree", force_unit="N", means_exclude_initial_state=True,
            maxima="Each backend's own full physics-step trace; GPU/native and CPU remain separate",
            common_strict_acceptance=dict(position_max_mm=2., rotation_max_deg=5.,
                each_finger_load_greater_than_n=LOAD_THRESHOLD_N, every_physics_step=True,
                cpu_mujoco_warnings_must_be_zero=True),
            reported_task_success="Original experiment's acceptance retained independently of common criterion"),
        runs=runs, warnings=warnings,
        plots=[dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
        provenance=dict(script=sources.optional(__file__), inputs=list(sources.files.values())))
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for warning in warnings:
        print(warning, file=sys.stderr)
    print(destination)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
