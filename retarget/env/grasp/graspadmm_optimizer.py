"""Record the pinned operation, with optional local geometry/soft-force adapters."""

import copy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .graspadmm_source import require_upstream
from .graspadmm_scene import WujiOptEnv, project_to_hand, filter_diagnostics


def operation_config(scene, directory, settings=None):
    source = require_upstream()
    values = {} if settings is None else dict(settings)
    op = copy.deepcopy(source.grasp_config)
    for key in ("step", "substep", "step_obj", "step_hand", "rho_admm", "lr_obj",
                "square_res", "cdist", "fix_hand_local_cp", "opt_miu_coef", "device"):
        if key in values:
            op.grasp[key] = values[key]
    directory = Path(directory).resolve()
    return source.OmegaConf.create(dict(
        legacy_api=False, hand_name="wuji2", hand={"xml_path": str(scene.hand_xml)},
        op=source.OmegaConf.to_container(op, resolve=False),
        init_dir=str(directory / "init"), grasp_dir=str(directory / "grasp"),
        debug_dir=str(directory / "debug"), debug_render=False, debug_view=False,
        tmpl_upd_mode="disabled", n_worker=1,
    ))


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def optimize(grasp_data, scene, directory, settings=None):
    """Execute the pinned _single_grasp function with explicit optional guidance.

    Python line tracing observes complete outer iterations and return values.
    The original QP, gradient updates, dual reset, fallback and finalization
    remain byte-for-byte in third_party/Dexonomy. A supplied whole-hand
    reference adds soft potential forces to the optimization environment only.
    """
    source = require_upstream()
    cfg = operation_config(scene, directory, settings)
    if cfg.op.grasp.device != "cpu":
        raise ValueError("Pinned GraspADMM supports CPU optimization; initialization uses CUDA")
    input_path = Path(cfg.init_dir) / "candidate.npy"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(input_path, grasp_data)
    core = source.core
    function = core._single_grasp.__wrapped__
    lines, first = inspect.getsourcelines(function)
    iteration_line = next(first + i for i, line in enumerate(lines) if "for iter in range(grasp_cfg.step):" in line)
    end_line = next(first + i for i, line in enumerate(lines) if "if terminated: # fallback" in line)
    cost_line = next(first + i for i, line in enumerate(lines) if "res.backward()" in line)
    trace, captured, filters, costs, simulation = [], {}, {}, {}, []

    def recording_env(*args, **kwargs):
        env = WujiOptEnv(*args, **kwargs)
        env.contact_regions = copy.deepcopy(grasp_data.get("contact_regions"))
        env.gesture_reference=copy.deepcopy(grasp_data.get('gesture_reference'))
        if grasp_data.get("gesture_reference") is not None:
            from .graspadmm_gesture import install_soft_reference
            install_soft_reference(env,grasp_data["gesture_reference"])
        if grasp_data.get("reference_orientation_wxyz") is not None:
            from .contact_regions import lock_root_orientation
            lock_root_orientation(env, grasp_data["reference_orientation_wxyz"])
        original_step = env.step_sim
        elapsed = 0.

        def step(substep):
            nonlocal elapsed
            caller = inspect.currentframe().f_back
            state = caller.f_locals if caller.f_code is function.__code__ else {}
            stage = "pregrasp" if "pregrasp_lst" in state else "fallback" if state.get("terminated", False) else "admm_hand"
            original_step(substep)
            elapsed += substep * env._model.opt.timestep
            simulation.append(dict(time=elapsed, stage=stage, qpos=env.get_hand_qpos().copy()))

        env.step_sim = step
        return env

    def record(locals_):
        if "iter" not in locals_ or "cp_h" not in locals_:
            return
        iteration = int(locals_["iter"])
        if trace and trace[-1]["iteration"] == iteration:
            return
        env = locals_["sim_env"]
        cp_o, cp_h, lam = (_array(locals_[name]) for name in ("cp_o", "cp_h", "lam"))
        object_cost = costs.get(iteration, dict(grasp_metric_cost=np.nan, admm_penalty_cost=np.nan))
        trace.append(dict(
            iteration=iteration, cp_o=cp_o, cp_h=cp_h, dual=lam,
            primal_residual_m=float(np.linalg.norm(cp_h - cp_o)),
            object_metric_available=iteration in costs,
            **object_cost,
            object_subproblem_cost=object_cost["grasp_metric_cost"] + object_cost["admm_penalty_cost"],
            hand_qpos=env.get_hand_qpos().copy(),
            hand_cpn_b=_array(locals_["hand_cpn_b"]),
        ))

    def tracer(frame, event, arg):
        if frame.f_code is not function.__code__:
            return None
        if event == "line" and frame.f_lineno == iteration_line:
            record(frame.f_locals)
        elif event == "line" and frame.f_lineno == end_line:
            record(frame.f_locals)
        elif event == "line" and frame.f_lineno == cost_line:
            state = frame.f_locals
            delta = _array(state["cp"] - state["cp_h"] - state["lam"])
            costs[int(state["iter"])] = dict(
                grasp_metric_cost=float(_array(state["res"])),
                admm_penalty_cost=float(.5 * cfg.op.grasp.rho_admm * np.sum(delta ** 2)),
            )
        elif event == "return":
            captured.update(frame.f_locals)
        return tracer

    original_filter = core.GraspFilter

    class RecordedFilter(original_filter):
        def forward(self, info):
            success, contacts = super().forward(info)
            stage = "pregrasp_filter" if self.log_name.endswith(" pregrasp") else "grasp_filter"
            filters[stage] = dict(filter_diagnostics(self.sim_env, self.cfg, info), success=bool(success))
            return success, contacts

    previous_trace = sys.gettrace()
    original_env, original_projection = core.MuJoCo_OptEnv, core.project_to_hand
    try:
        core.MuJoCo_OptEnv = recording_env
        core.project_to_hand = project_to_hand
        core.GraspFilter = RecordedFilter
        sys.settrace(tracer)
        function((str(input_path), cfg))
    finally:
        sys.settrace(previous_trace)
        core.MuJoCo_OptEnv, core.project_to_hand = original_env, original_projection
        core.GraspFilter = original_filter
    data = captured.get("grasp_data", copy.deepcopy(grasp_data))
    if grasp_data.get("gesture_reference") is not None:
        data["gesture_reference"]=copy.deepcopy(grasp_data["gesture_reference"])
    status = dict(
        success=bool(filters.get("grasp_filter", {}).get("success") and filters.get("pregrasp_filter", {}).get("success")),
        failure_stage=None, fallback_used=bool(captured.get("terminated", False)),
        outer_iterations=len(trace), **filters,
    )
    if not filters.get("grasp_filter", {}).get("success"):
        status["failure_stage"] = "grasp_filter"
    elif not filters.get("pregrasp_filter", {}).get("success"):
        status["failure_stage"] = "pregrasp_filter"
    output = Path(directory)
    from .contact_regions import contact_region_diagnostics, orientation_error_deg
    region = grasp_data.get("contact_regions")
    if region is not None:
        names = grasp_data["hand_cbody"]
        contacts = data.get("ho_c", {})
        actual = np.c_[np.asarray(contacts.get("pos", [])).reshape(-1, 3),
                       np.asarray(contacts.get("normal", [])).reshape(-1, 3)]
        diagnostics = dict(
            initial_anchors=contact_region_diagnostics(grasp_data["grasp_qpos"][0], names,
                                                       grasp_data["hand_cpn_w"], region),
            actual_contacts=contact_region_diagnostics(data["grasp_qpos"][0], contacts.get("bn1", []), actual, region))
        if trace:
            # Reconstruct the last optimized anchors independently of the
            # upstream finalization that replaces anchors with real contacts.
            env = captured["sim_env"]
            env.reset_qpos(trace[-1]["hand_qpos"])
            anchors = env.transform_cpn_b2w(names, trace[-1]["hand_cpn_b"])
            diagnostics["optimized_anchors"] = contact_region_diagnostics(trace[-1]["hand_qpos"], names, anchors, region)
        (output / "contact_region_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
        status["contact_region_diagnostics"] = diagnostics
    reference = grasp_data.get("reference_orientation_wxyz")
    if reference is not None:
        status["reference_orientation_error_deg"] = orientation_error_deg(data["grasp_qpos"][0], reference)
        status["max_trace_orientation_error_deg"] = max(
            (orientation_error_deg(row["hand_qpos"], reference) for row in trace), default=0.)
    if grasp_data.get("gesture_reference") is not None:
        from .graspadmm_gesture import reference_diagnostics
        status["gesture_diagnostics"]=reference_diagnostics(data["grasp_qpos"][0],grasp_data["gesture_reference"])
    np.save(output / "graspadmm_data.npy", data)
    if simulation:
        np.savez_compressed(output / "simulation_trace.npz",
                            **{key: np.asarray([row[key] for row in simulation]) for key in simulation[0]})
    if trace:
        np.savez_compressed(output / "admm_trace.npz", **{
            key: np.asarray([row[key] for row in trace])
            for key in trace[0]
        })
    return data, status, trace
