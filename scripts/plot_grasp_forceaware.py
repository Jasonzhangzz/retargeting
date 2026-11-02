"""Plot saved ForceAware grasp diagnostics without loading a simulator.

Usage::

    .venv-contact/bin/python scripts/plot_grasp_forceaware.py output/experiment

Each completed case contributes one column. The top row compares native and
independent mesh-replay object drift; the bottom row shows the five fingers'
actual native normal forces. PNG and PDF figures are saved in the input root.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
FINGER_COLORS = ("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#7B61A8")


@dataclass
class CaseData:
    path: Path
    summary: dict
    time_ms: np.ndarray
    native_drift_mm: np.ndarray
    mesh_time_ms: np.ndarray
    mesh_drift_mm: np.ndarray
    force_time_ms: np.ndarray
    force_n: np.ndarray
    finger_names: tuple
    pre_integration_forces: bool


def read_json(path):
    return json.loads(path.read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def state_time(data, path):
    qpos, time = np.asarray(data["qpos"]), np.asarray(data["time"], dtype=float)
    if qpos.ndim != 2 or time.shape != (len(qpos),) or len(time) < 2:
        raise ValueError(f"Invalid qpos/time shapes in {path}")
    if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
        raise ValueError(f"Time must be finite and increasing in {path}")
    return qpos, time


def find_mesh_replay(case, summary):
    # Prefer portable output-local paths over the original absolute output path.
    matches = sorted(case.glob("sequence/retarget/*/forceaware/replay/mujoco_cpu/replay.npz"))
    if not matches and summary.get("original_outputs"):
        matches = [Path(summary["original_outputs"]) / "replay/mujoco_cpu/replay.npz"]
        matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Expected one independent mesh replay for {case}; found {len(matches)}")
    return matches[0]


def load_case(case):
    summary = read_json(case / "summary.json")
    trajectory = read_npz(case / "trajectory.npz")
    qpos, time = state_time(trajectory, case / "trajectory.npz")
    reference = np.asarray(trajectory["object_reference_qpos"], dtype=float)
    if reference.shape == (7,):
        reference = np.broadcast_to(reference, (len(time), 7))
    if reference.shape != (len(time), 7):
        raise ValueError(f"Object reference must be [states,7] or [7] in {case}")

    manifest_path = case / "reference_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    # Prepared grasp scenes place the single object's free joint after the hand.
    object_qadr = int(manifest.get("qpos_dimension", qpos.shape[1] - 7))
    if object_qadr < 0 or qpos.shape[1] != object_qadr + 7:
        raise ValueError(f"Expected hand qpos followed by one object free joint in {case}")
    native_drift = np.linalg.norm(qpos[:, object_qadr:object_qadr + 3] - reference[:, :3], axis=1) * 1000.

    mesh_path = find_mesh_replay(case, summary)
    mesh_qpos, mesh_time = state_time(read_npz(mesh_path), mesh_path)
    if mesh_qpos.shape[1] != qpos.shape[1]:
        raise ValueError(f"Native and mesh qpos dimensions differ in {case}")
    if mesh_time[0] < time[0] - 1e-7 or mesh_time[-1] > time[-1] + 1e-7:
        raise ValueError(f"Mesh replay extends beyond the reference time range in {case}")
    mesh_reference = np.column_stack([np.interp(mesh_time, time, reference[:, axis]) for axis in range(3)])
    mesh_drift = np.linalg.norm(mesh_qpos[:, object_qadr:object_qadr + 3] - mesh_reference, axis=1) * 1000.

    audit = read_npz(case / "contact_audit/native_replay.npz")
    audit_report_path = case / "contact_audit/replay.json"
    audit_report = (read_json(audit_report_path) if audit_report_path.is_file()
                    else summary.get("physical_contact_audit", {}))
    force = np.asarray(audit["finger_normal_force_n"], dtype=float)
    force_time = np.asarray(audit["time"], dtype=float)
    names = tuple(audit_report.get("finger_names", FINGER_NAMES))
    if force.shape != (len(force_time), 5) or len(names) != 5:
        raise ValueError(f"Expected five actual finger normal forces per audit step in {case}")
    if not np.isfinite(force_time).all() or not np.all(np.diff(force_time) > 0):
        raise ValueError(f"Invalid force audit time in {case}")
    pre_integration = str(audit_report.get("collision_stage", "")).startswith("pre_integration")
    if pre_integration:
        # Audit time belongs to the post-step state; collision forces precede it.
        dt = float(audit_report.get("dt", np.median(np.diff(time))))
        force_time = force_time - dt
    return CaseData(case, summary, time * 1000., native_drift, mesh_time * 1000.,
                    mesh_drift, force_time * 1000., force, names, pre_integration)


def discover_cases(root, selected=()):
    if (root / "trajectory.npz").is_file() and (root / "summary.json").is_file():
        candidates = [root]
    else:
        candidates = sorted(path for path in root.iterdir() if path.is_dir()
                            and ((path / "reference_manifest.json").is_file()
                                 or (path / "trajectory.npz").is_file()))
    if selected:
        found = {path.name for path in candidates}
        missing = set(selected) - found
        if missing:
            raise ValueError(f"Unknown case names: {', '.join(sorted(missing))}")
        candidates = [path for path in candidates if path.name in selected]
    completed = []
    for case in candidates:
        if not all((case / name).is_file() for name in ("summary.json", "trajectory.npz")):
            print(f"Skipping unfinished case: {case.name}", file=sys.stderr)
            continue
        completed.append(case)
    if not completed:
        raise ValueError(f"No completed ForceAware cases found in {root}")
    return completed


def case_title(case):
    match = re.search(r"cylinder_r([\d.]+)_h([\d.]+)", case.path.name)
    if match:
        radius, height = (1000. * float(value) for value in match.groups())
        title = f"Cylinder: radius {radius:g} mm, height {height:g} mm"
    else:
        title = textwrap.fill(case.path.name.replace("_", " "), width=44)
    state = "complete" if case.summary.get("solver_completed") else "incomplete"
    result = "passed" if case.summary.get("task_success") else "failed"
    return f"{title}\nSolver {state}; grasp {result}"


def plot_cases(cases, root, *, name="forceaware_diagnostics", dpi=200):
    with plt.rc_context({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(2, len(cases), squeeze=False,
                                    figsize=(5.0 * len(cases), 6.6))
        figure.suptitle("ForceAware grasp diagnostics", fontsize=13, y=.99)
        for column, case in enumerate(cases):
            top, bottom = axes[:, column]
            if column:
                top.sharey(axes[0, 0])
                top.tick_params(labelleft=False)
            top.plot(case.time_ms, case.native_drift_mm, color="#0072B2", lw=1.8, label="Native GS/SDF")
            top.plot(case.mesh_time_ms, case.mesh_drift_mm, color="#D55E00", lw=1.6,
                     ls="--", label="MuJoCo mesh replay")
            limit = float(case.summary.get("acceptance", {}).get("max_object_drift_mm", 2.))
            top.axhline(limit, color="#666666", ls=":", lw=1.2, label=f"{limit:g} mm limit")
            top.set_title(case_title(case), pad=10)
            top.legend(loc="upper left", fontsize=8, framealpha=.9)
            for finger, (finger_name, color) in enumerate(zip(case.finger_names, FINGER_COLORS)):
                bottom.plot(case.force_time_ms, case.force_n[:, finger], color=color,
                            lw=1.05, label=finger_name.capitalize())
            bottom.axhline(0., color="#777777", lw=.5)
            bottom.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=5,
                          fontsize=8, frameon=False, handlelength=1.2,
                          columnspacing=.9, borderaxespad=0.)
            for axis in (top, bottom):
                axis.set_xlabel("Time (ms)")
                axis.set_xlim(0., max(case.time_ms[-1], case.mesh_time_ms[-1]))
                axis.set_ylim(bottom=0.)
                axis.grid(True, alpha=.22, lw=.6)
                axis.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel("Object drift from target (mm)")
        axes[1, 0].set_ylabel("Actual normal force (N)")
        note = "Forces: physical native contact rows, summed per finger; no time averaging."
        if all(case.pre_integration_forces for case in cases):
            note += " Force times are before integration."
        figure.text(.5, .018, note, ha="center", va="bottom", fontsize=8, color="#444444", wrap=True)
        figure.tight_layout(rect=(0., .06, 1., .94), h_pad=2.)
        paths = [root / f"{name}.{extension}" for extension in ("png", "pdf")]
        for path in paths:
            figure.savefig(path, dpi=dpi, facecolor="white")
        plt.close(figure)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rootoutput", type=Path, help="ForceAware experiment root, or one completed case")
    parser.add_argument("--case", action="append", default=[], help="Plot this case directory name; may repeat")
    parser.add_argument("--name", default="forceaware_diagnostics", help="Output file basename (PNG and PDF)")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution; default: 200")
    args = parser.parse_args(argv)
    if Path(args.name).name != args.name or not args.name or args.dpi <= 0:
        parser.error("--name must be a file basename and --dpi must be positive")
    root = args.rootoutput.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Root output directory does not exist: {root}")
    try:
        cases = [load_case(path) for path in discover_cases(root, args.case)]
        paths = plot_cases(cases, root, name=args.name, dpi=args.dpi)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Plotted {len(cases)} completed case(s)")
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
