"""Compare saved SPIDER tracking and local contact-keeping runs on CPU.

Run only after both experiment writers have finished::

    .venv-contact/bin/python scripts/analyze_spider_contact_keep.py

The input runs are read-only. The local added loss uses actual native solved
contact forces; it is not original SPIDER or DexForge ForceAware optimization.
Executed-trace rescoring is an offline diagnostic, not a reconstruction of the
MPPI prediction-window objective. No simulator, Torch, Warp, or GPU is loaded.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from analyze_spider_grasp import FINGERS, RADII, Sources, radius_from, state_time, tracking


POSITIVE_THRESHOLD_N = 1e-6
LOCAL_KEYS = {"contact_keep_rew_scale", "contact_min_force_n"}


def force_metrics(force, required, *, minimum, weight, terminal_scale, baseline):
    force = np.asarray(force, dtype=float)
    required = np.asarray(required, dtype=bool)
    if force.ndim != 2 or force.shape[1] != 5 or not len(force) or required.shape != (5,) or not required.any():
        raise ValueError("Expected measured force[steps,5] and at least one deduplicated CA finger")
    if not np.isfinite(force).all() or not np.isfinite(minimum) or minimum <= 0 or not np.isfinite(weight) or weight < 0:
        raise ValueError("Force observations and nonnegative weight must be finite; target must be positive")
    positive = force > POSITIVE_THRESHOLD_N
    target_loaded = force >= minimum
    deficit_squared = np.maximum(1. - np.maximum(force, 0.) / minimum, 0.) ** 2
    sum_required = deficit_squared[:, required].sum(axis=1)
    weighted = weight * sum_required
    temporal = np.ones(len(force)); temporal[-1] = terminal_scale
    return dict(finger_names=list(FINGERS), physics_steps=len(force),
        required_finger_mask=required.tolist(), required_finger_names=[name for name, active in zip(FINGERS, required) if active],
        positive_load_threshold_n=POSITIVE_THRESHOLD_N, minimum_target_force_n=minimum,
        finger_zero_load_steps=(~positive).sum(0).tolist(),
        finger_under_target_steps=(~target_loaded).sum(0).tolist(),
        finger_normal_force_min_n=force.min(0).tolist(), finger_normal_force_mean_n=force.mean(0).tolist(),
        finger_normal_force_max_n=force.max(0).tolist(),
        all_ca_fingers_positive_fraction=float(positive[:, required].all(1).mean()),
        all_ca_fingers_at_target_fraction=float(target_loaded[:, required].all(1).mean()),
        all_five_fingers_positive_fraction=float(positive.all(1).mean()),
        normalized_deficit_squared_per_finger_mean=deficit_squared.mean(0).tolist(),
        normalized_deficit_squared_required_mean=float(deficit_squared[:, required].mean()),
        normalized_deficit_squared_required_sum_mean=float(sum_required.mean()),
        offline_rescoring=dict(only_offline=True,
            baseline_offline_only=baseline, weight_from_new_run=weight,
            target_from_new_run_n=minimum, mean_weighted_keep_score=float(weighted.mean()),
            terminal_scale_from_new_run=terminal_scale,
            whole_execution_terminal_scaled_mean=float((weighted * temporal).mean()),
            formula="weight * mean_steps(sum_required_fingers(relu(1-max(actual_Fn,0)/target_N)^2))",
            scope="Executed physics-step trace diagnostic; not the optimized MPC prediction-window score"))


def yaml_input(path, sources):
    if not path.is_file():
        return {}, None
    raw, entry = sources.read(path)
    return yaml.safe_load(raw), entry


def discover(root, sources, warnings, label):
    root = Path(root).resolve()
    if not root.is_dir():
        warnings.append(f"Missing {label} run: {root}")
        return {}, dict(root=str(root), available=False)
    config, config_source = yaml_input(root / "config.yaml", sources)
    run_sources = dict(config=config_source)
    for name in ("summary.json", "manifest.json", "provenance.json", "status.json"):
        run_sources[name] = sources.optional(root / name)
    cases = {}
    for case in sorted(root.iterdir()):
        if not case.is_dir() or not (case / "reference_manifest.json").is_file():
            continue
        needed = ("summary.json", "trajectory.npz", "replay/replay.npz", "ca_input_hashes.json")
        if not all((case / name).is_file() for name in needed):
            warnings.append(f"Skipped unfinished {label} case: {case}")
            continue
        report, report_source = sources.json(case / "summary.json")
        reference, reference_source = sources.json(case / "reference_manifest.json")
        hashes, hashes_source = sources.json(case / "ca_input_hashes.json")
        initial, initial_source = sources.json(case / "control_initialization.json")
        trajectory, trajectory_source = sources.npz(case / "trajectory.npz", ("qpos", "time", "finger_normal_force_n"))
        replay, replay_source = sources.npz(case / "replay/replay.npz", ("cpu_qpos", "time_cpu", "cpu_finger_normal_force_n"))
        retention, retention_source = ({}, None)
        if (case / "contact_retention.json").is_file():
            retention, retention_source = sources.json(case / "contact_retention.json")
        declared = []
        for path, digest in hashes.items():
            entry = sources.optional(path)
            matches = entry is not None and entry["sha256"] == digest
            declared.append(dict(path=path, recorded_sha256=digest, currently_matches=matches))
            if not matches:
                warnings.append(f"Recorded CA input is absent or changed for {label}: {path}")
        record = dict(path=case, report=report, reference=reference, hashes=hashes, initial=initial,
            retention=retention, trajectory=trajectory, replay=replay, declared_inputs=declared,
            sources=dict(summary=report_source, reference_manifest=reference_source, ca_input_hashes=hashes_source,
                control_initialization=initial_source, trajectory=trajectory_source, cpu_replay=replay_source,
                contact_retention=retention_source, replay_report=sources.optional(case / "replay/replay.json"),
                model=sources.optional(case / "model.xml")))
        radius = radius_from(case.name, report.get("source_experiment", ""))
        if radius in cases:
            raise ValueError(f"Multiple completed {label} cases for radius {radius}")
        cases[radius] = record
    return cases, dict(root=str(root), available=bool(cases), config=config, sources=run_sources,
                       completed_radii_m=sorted(cases))


def body_fingers(names):
    """WUJI2 name convention used only to cross-check the saved physics mask."""
    import re
    required = np.zeros(5, dtype=bool)
    for name in names:
        match = re.search(r"(?:^|_)finger([1-5])(?:_|$)", name)
        if not match:
            raise ValueError(f"Cannot map accepted WUJI2 contact body to a finger: {name}")
        required[int(match.group(1))-1] = True
    return required


def required_mask(before, after):
    raw = np.asarray(after["retention"].get("required_finger_mask"))
    if raw.shape != (5,) or not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError("New run must record a valid five-entry deduplicated required_finger_mask")
    required = raw.astype(bool)
    if not required.any():
        raise ValueError("At least one CA finger must be required")
    for record in (before, after):
        names = record["initial"].get("active_bodies", [])
        if not names or not np.array_equal(body_fingers(names), required):
            raise ValueError(f"Saved CA bodies disagree with the physical finger mask: {record['path']}")
    slots = after["retention"].get("active_ca_slots")
    if slots != after["reference"]["active_slot_indices"]:
        raise ValueError("New contact-retention CA slots disagree with its reference manifest")
    return required


def differences(first, second):
    return {key: dict(before=first.get(key), after=second.get(key))
            for key in sorted(set(first) | set(second)) if first.get(key) != second.get(key)}


def controls_check(before, after, before_run, after_run):
    before_report, after_report = before["report"], after["report"]
    reference_fields = ("hand_reference_qpos", "object_reference_qpos", "active_slot_indices",
                        "active_contact_count", "hand_contact_local", "object_contact_local")
    same_reference = {name: before["reference"].get(name) == after["reference"].get(name) for name in reference_fields}
    before_cost = {k: v for k, v in before_report["cost"].items() if k not in LOCAL_KEYS}
    after_cost = {k: v for k, v in after_report["cost"].items() if k not in LOCAL_KEYS}
    profiles = [{k: v for k, v in record["report"].get("profile", {}).items() if k != "cost"}
                for record in (before, after)]
    shared = set(before["hashes"]) & set(after["hashes"])
    shared_identical = all(before["hashes"][path] == after["hashes"][path] for path in shared)
    before_config, after_config = before_run["config"], after_run["config"]
    cost_changes = differences(before_cost, after_cost)
    profile_changes = differences(*profiles)
    sampler_changes = differences(before_config.get("solver", {}), after_config.get("solver", {}))
    ca_unchanged = bool(all(same_reference.values()) and before["hashes"] == after["hashes"]
        and all(item["currently_matches"] for record in (before, after) for item in record["declared_inputs"])
        and before_report.get("contactaware_changed") is False and after_report.get("contactaware_changed") is False)
    return dict(contactaware_unchanged=ca_unchanged, reference_fields_identical=same_reference,
        declared_ca_hash_maps_identical=before["hashes"] == after["hashes"],
        shared_declared_input_hashes_identical=shared_identical, shared_declared_input_count=len(shared),
        original_tracking_weight_differences=cost_changes, dynamics_and_horizon_profile_differences=profile_changes,
        requested_solver_setting_differences=sampler_changes,
        seed_before=before_config.get("seed"), seed_after=after_config.get("seed"),
        same_ca_requested_budgets_and_tracking_weights=bool(ca_unchanged and not cost_changes and not profile_changes
            and not sampler_changes and before_config.get("seed") == after_config.get("seed")),
        actual_iterations_may_differ="Realized iterations and runtime may change because early stopping depends on cost",
        forceaware_optimizer_used=bool(before_report.get("forceaware_optimizer_used") or after_report.get("forceaware_optimizer_used")),
        forceaware_cost_used=bool(before_report.get("forceaware_cost_used") or after_report.get("forceaware_cost_used")))


def case_metrics(record, reference, required, *, minimum, weight, terminal_scale, baseline):
    report, trajectory, replay = record["report"], record["trajectory"], record["replay"]
    result = dict(case=str(record["path"]), source_experiment=report.get("source_experiment"),
        timing=report.get("timing", {}), sampler=report.get("sampler", {}), cost=report["cost"],
        optimized_keep_cost_weight=report["cost"].get("contact_keep_rew_scale", 0.),
        reported_task_success=report.get("task_success"), finite=report.get("finite"),
        hardware_tested=False, sources=record["sources"], declared_ca_inputs=record["declared_inputs"])
    curves = {}
    for backend, qpos, time, force in (
        ("gpu", trajectory["qpos"], trajectory["time"], trajectory["finger_normal_force_n"]),
        ("cpu", replay["cpu_qpos"], replay["time_cpu"], replay["cpu_finger_normal_force_n"])):
        qpos, time = state_time(qpos, time, record["path"])
        if len(force) != len(time)-1:
            raise ValueError(f"Missing physics-step force observations: {record['path']}")
        drift, backend_result = tracking(qpos, reference)
        backend_result.update(force_metrics(force, required, minimum=minimum, weight=weight,
            terminal_scale=terminal_scale, baseline=baseline))
        backend_result["strict_success"] = bool(backend_result["position_mm"]["max"] <= 2.
            and backend_result["rotation_deg"]["max"] <= 5.
            and backend_result["all_ca_fingers_positive_fraction"] == 1.
            and np.isfinite(qpos).all()
            and (backend != "cpu" or report.get("mujoco_warnings_zero") is True))
        result[backend] = backend_result
        curves[backend] = dict(time=time, drift=drift, force=force)
    result["strict_success"] = bool(result["gpu"]["strict_success"] and result["cpu"]["strict_success"]
                                    and report.get("finite"))
    return result, curves


def draw(comparisons, output, *, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#0072B2", "#D55E00", "#75B8DD", "#F3AE78")
    with plt.rc_context({"font.size": 10, "axes.titlesize": 10, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(3, 2, figsize=(12.5, 10.6))
        figure.suptitle("WUJI2: adding the local CA-finger contact-keeping cost", fontsize=14, y=.985)
        for row, radius in enumerate(RADII):
            left, right = axes[row]
            left.set_title(f"Radius {1000.*radius:g} mm: object drift")
            right.set_title(f"Radius {1000.*radius:g} mm: actual zero-load step counts")
            if radius in comparisons:
                comparison = comparisons[radius]
                x = np.arange(5)
                for index, (phase, backend, label) in enumerate((("before", "gpu", "Before GPU"),
                        ("after", "gpu", "After +keep GPU"), ("before", "cpu", "Before CPU replay"),
                        ("after", "cpu", "After +keep CPU replay"))):
                    trace = comparison["curves"][phase][backend]
                    left.plot(trace["time"], trace["drift"], color=colors[index % 2],
                              lw=1.6, ls="--" if backend == "cpu" else "-", label=label)
                    values = comparison["summary"][phase][backend]["finger_zero_load_steps"]
                    right.bar(x+(index-1.5)*.19, values, .19, color=colors[index],
                              hatch="//" if backend == "cpu" else None, label=label)
                right.set_xticks(x, [name.capitalize() for name in FINGERS])
                steps = max(comparison["summary"][phase][backend]["physics_steps"]
                            for phase in ("before", "after") for backend in ("gpu", "cpu"))
                right.set_ylim(0., max(200, steps))
            else:
                for axis in (left, right):
                    axis.text(.5, .5, "Paired completed runs unavailable", ha="center", transform=axis.transAxes)
                right.set_ylim(0., 200.)
            left.axhline(2., color="#666666", ls=":", lw=1., label="2 mm limit")
            left.set_xlabel("Step-end time (s)"); left.set_ylabel("Object position error (mm)")
            left.set_xlim(0., .4); left.set_ylim(bottom=0.)
            right.set_ylabel("Steps with normal load <= 1e-6 N")
            for axis in (left, right):
                axis.grid(True, axis="y", alpha=.23, lw=.6)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
                if radius in comparisons:
                    axis.legend(loc="best", fontsize=8, ncol=2, framealpha=.9)
        figure.text(.5, .016,
            "0.4 s simulation; actual solved forces at every 2 ms physics step. No hardware test.\n"
            "Before: original saved tracking run. After: same tracking objective plus the LOCAL force-deficit term.\n"
            "GPU execution and independent CPU replay are distinct trajectories; offline before scores were never optimized.",
            ha="center", va="bottom", fontsize=8, color="#444444")
        figure.tight_layout(rect=(0., .085, 1., .96), h_pad=1.7, w_pad=2.)
        paths = [output / f"contact_keep_comparison.{extension}" for extension in ("png", "svg")]
        for path in paths:
            figure.savefig(path, dpi=dpi, facecolor="white")
        plt.close(figure)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=Path("output/grasp_contactaware_spider_tracking"))
    parser.add_argument("--after", type=Path, default=Path("output/grasp_contactaware_spider_contact_keep"))
    parser.add_argument("--output", type=Path, default=Path("output/grasp_spider_contact_keep_comparison"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.before.resolve(), args.after.resolve()):
        raise ValueError("Comparison output must be separate from both source runs")
    sources, warnings = Sources(), []
    before_cases, before_run = discover(args.before, sources, warnings, "before")
    after_cases, after_run = discover(args.after, sources, warnings, "after")
    comparisons = {}
    for radius in RADII:
        if radius not in before_cases or radius not in after_cases:
            warnings.append(f"Missing paired completed radius {radius:g} m")
            continue
        before, after = before_cases[radius], after_cases[radius]
        required = required_mask(before, after)
        cost = after["report"]["cost"]
        minimum, weight = float(cost["contact_min_force_n"]), float(cost["contact_keep_rew_scale"])
        terminal_scale = float(cost["terminal_rew_scale"])
        reference = np.asarray(before["trajectory"]["qpos"][0, -7:], dtype=float)
        if not np.array_equal(reference, after["trajectory"]["qpos"][0, -7:]):
            raise ValueError(f"Before/after object reference changed for radius {radius}")
        if before["report"]["cost"].get("contact_keep_rew_scale", 0.) != 0. or weight <= 0.:
            raise ValueError("Expected an unmodified tracking baseline and an enabled added local cost")
        before_metrics, before_curves = case_metrics(before, reference, required, minimum=minimum,
            weight=weight, terminal_scale=terminal_scale, baseline=True)
        after_metrics, after_curves = case_metrics(after, reference, required, minimum=minimum,
            weight=weight, terminal_scale=terminal_scale, baseline=False)
        control_checks = controls_check(before, after, before_run, after_run)
        if not control_checks["same_ca_requested_budgets_and_tracking_weights"]:
            warnings.append(f"Before/after controls differ beyond the added loss for radius {radius:g}; inspect controls_check")
        summary = dict(radius_m=radius, required_finger_mask=required.tolist(),
            active_ca_slots=after["reference"]["active_slot_indices"],
            required_mask_source="New run contact_retention.json, cross-checked against unique accepted WUJI2 finger bodies in both runs",
            mask_is_subset_and_deduplicated=True, controls_check=control_checks,
            before=before_metrics, after=after_metrics,
            added_local_cost=dict(contact_keep_rew_scale=weight, contact_min_force_n=minimum,
                origin="LOCAL SPIDER-MPPI extension; not original SPIDER; no DexForge ForceAware optimizer/engine",
                actual_force_source="Solved native object-finger contact rows; CPU replay uses mj_contactForce"))
        comparisons[radius] = dict(summary=summary, curves=dict(before=before_curves, after=after_curves))
    args.output.mkdir(parents=True, exist_ok=True)
    plot_paths = draw(comparisons, args.output, dpi=args.dpi)
    sources.optional(__file__)
    sources.optional(Path(__file__).with_name("analyze_spider_grasp.py"))
    summary = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(),
        aggregation_only=True, hardware_tested=False,
        scope="Paired saved MPPI/native MuJoCo-Warp runs; local actual-force deficit penalty added to tracking",
        before_run=before_run, after_run=after_run,
        conventions=dict(positive_load_n=POSITIVE_THRESHOLD_N, zero_load="load <= positive_load_n",
            under_target="load < new run contact_min_force_n", normalized_deficit="relu(1-max(actual_Fn,0)/target_N)^2",
            required_deficit_mean="Mean over required fingers and all physics steps",
            required_deficit_sum_mean="Mean over steps after summing required fingers, matching added-cost formula",
            offline_rescoring="Both executed traces use the NEW weight/target for comparison; baseline scores were never optimized",
            metric_maxima="GPU and CPU maxima reported separately at every physics step",
            force_stage="Native solver contacts precede the accompanying post-integration qpos; no extra forward calls"),
        paired_cases=len(comparisons), cases=[item["summary"] for item in comparisons.values()], warnings=warnings,
        provenance=dict(inputs=list(sources.files.values())),
        plots=[dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in plot_paths])
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    artifacts = dict(summary=dict(path=str(destination.resolve()), sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),
                     plots=summary["plots"], inputs=summary["provenance"]["inputs"])
    (args.output / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    for warning in warnings:
        print(warning, file=sys.stderr)
    print(destination)
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
