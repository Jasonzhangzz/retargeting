"""Audit a saved WUJI FA native control VJP without changing the optimizer."""

import argparse
from dataclasses import fields
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(root, output, *, horizon=1, seed=177, servo_integration=False):
    from retarget.dynamics.dexforge_upstream import require_environment, activate_upstream
    require_environment(); activate_upstream()
    profile = "original_native"
    integration_metadata = None
    if servo_integration:
        from retarget.dynamics.dexforge_servo_integration import activate_servo_integration
        integration_metadata = activate_servo_integration()
        profile = integration_metadata["profile"]
        output = output / profile  # Preserve the independent original-native evidence.
    import warp as wp
    import comfree_warp
    from forceaware.config import load_retarget_config
    from forceaware.workflow import _load_runtime_models
    from retarget.env.grasp.forceaware_reference import load_reference, register_wuji2_hand
    from retarget.env.humanoid.dexforge_runner import _contact_signature
    from retarget.common.results import write_json
    from comfree_warp.native_adjoint.dynamics import DynamicsInput
    from comfree_warp.native_adjoint.fast_step import compile_fast_step, clear_contact_status, check_contact_status
    from comfree_warp.native_adjoint.fast_step_adjoint import BackwardCall, StepCotangent, allocate_workspace, record, forward, backward
    from comfree_warp.native_adjoint.runtime import StepInput

    wp.init(); wp.set_device("cuda:0"); register_wuji2_hand()
    cfg = load_retarget_config(root / "sequence", root / "forceaware_model.yaml")
    models = _load_runtime_models(cfg, comfree_warp)
    cpu = models.planner_cpu
    reference = load_reference(root, cpu, models.planner_device, cfg)
    compiled = compile_fast_step(cpu, models.planner_device)
    assert not compiled.sparse_contact, "This instrumented audit targets the actual dense WUJI solve"
    device = compiled.base.device
    matrix = lambda value: wp.array(np.asarray(value, np.float32), dtype=float, device=device, requires_grad=True, retain_grad=True)
    qbase, vbase = reference.qpos.astype(np.float32), reference.qvel.astype(np.float32)
    q, v, t = matrix(qbase[None]), matrix(vbase[None]), matrix(np.zeros(1))
    initial_q, initial_v, initial_t = q, v, t
    ubase = np.zeros((horizon, cpu.nu), np.float32)
    ubase[:, reference.index.hand_ctrl] = qbase[reference.index.hand_qpos]
    stages = []
    for step in range(horizon):
        dynamics = DynamicsInput()
        dynamics.qpos, dynamics.qvel = q, v
        dynamics.ctrl, dynamics.qfrc_applied = matrix(ubase[step:step+1]), matrix(np.zeros((1, cpu.nv)))
        inputs = StepInput(dynamics, t, matrix(np.zeros(1)))
        workspace = allocate_workspace(compiled, 1)
        recorded = record(compiled, inputs, workspace, freeze_frame_vjp=False)
        z = lambda shape, dtype=float: wp.zeros(shape, dtype=dtype, device=device)
        c = compiled.base.collision.contact_count
        cotangent = StepCotangent(z((1, cpu.nq)), z((1, cpu.nv)), z((1, cpu.nv)), z(1), z((1, cpu.nv)),
            z((1, cpu.nbody), wp.vec3), z((1, cpu.nbody), wp.mat33), z((1, c)), z((1, c), wp.vec3), z((1, c), wp.mat33))
        stages.append((inputs, workspace, recorded, cotangent))
        q, v, t = recorded.result.qpos, recorded.result.qvel, recorded.result.time

    rng = np.random.default_rng(seed)
    qseed = rng.choice([-1., 1.], cpu.nq).astype(np.float32)
    vseed = (.125*rng.choice([-1., 1.], cpu.nv)).astype(np.float32)
    du = rng.standard_normal(ubase.shape); du /= np.linalg.norm(du)
    seeds = {"mixed": (qseed, vseed), "qpos_only": (qseed, np.zeros(cpu.nv)),
        "qvel_only": (np.zeros(cpu.nq), vseed),
        "object_position": (np.r_[np.zeros(reference.index.object_qpos), qseed[reference.index.object_qpos:reference.index.object_qpos+3], np.zeros(4)], np.zeros(cpu.nv)),
        "hand_velocity": (np.zeros(cpu.nq), np.r_[vseed[:26], np.zeros(cpu.nv-26)]),
        "object_velocity": (np.zeros(cpu.nq), np.r_[np.zeros(26), vseed[26:]])}

    def evaluate(controls):
        initial_q.assign(qbase[None]); initial_v.assign(vbase[None]); initial_t.zero_()
        masks, raw_masks, geometric, pressures, state_q, state_v, clamp = [], [], [], [], [], [], []
        for step, (inputs, workspace, recorded, _) in enumerate(stages):
            u = np.asarray(controls[step:step+1], np.float32)
            inputs.dynamics.ctrl.assign(u)
            clear_contact_status(compiled, workspace.forward)
            result = forward(compiled, inputs, workspace)
            check_contact_status(compiled, workspace.forward)
            geometric.append(_contact_signature(compiled, workspace)[0])
            contact = workspace.forward.contact
            masks.append(np.stack([s.rows.slope.numpy()[0] for s in contact.states]).astype(bool))
            raw = contact.final.rows.free_force.numpy()[0] - contact.final.rows.velocity_weight.numpy()[0]*contact.final.response.numpy()[0]
            raw_masks.append(raw > 0.)
            pressures.append(contact.output.force.numpy()[0])
            state_q.append(result.qpos.numpy()[0]); state_v.append(result.qvel.numpy()[0])
            # Native direct scalar actuators have unit gear in this scene.
            qin, vin = inputs.dynamics.qpos.numpy()[0], inputs.dynamics.qvel.numpy()[0]
            qadr = cpu.jnt_qposadr[cpu.actuator_trnid[:,0]]
            dadr = cpu.jnt_dofadr[cpu.actuator_trnid[:,0]]
            uc = np.clip(u[0], cpu.actuator_ctrlrange[:,0], cpu.actuator_ctrlrange[:,1])
            raw_force = cpu.actuator_gainprm[:,0]*uc + cpu.actuator_biasprm[:,0] + cpu.actuator_biasprm[:,1]*qin[qadr] + cpu.actuator_biasprm[:,2]*vin[dadr]
            clamp.append(np.stack((u[0] <= cpu.actuator_ctrlrange[:,0], u[0] >= cpu.actuator_ctrlrange[:,1],
                raw_force <= cpu.actuator_forcerange[:,0], raw_force >= cpu.actuator_forcerange[:,1])))
        return dict(qpos=np.asarray(state_q), qvel=np.asarray(state_v), slopes=np.asarray(masks),
            projection=np.asarray(raw_masks), geometry=geometric, force=np.asarray(pressures), clamps=np.asarray(clamp))

    base = evaluate(ubase)
    repeated = [evaluate(ubase) for _ in range(2)]
    gradients, adjoints = {}, {}
    for name, (sq, sv) in seeds.items():
        evaluate(ubase)
        for _, _, _, cotangent in stages:
            for field in fields(cotangent):
                getattr(cotangent, field.name).zero_()
        stages[-1][3].qpos.assign(np.asarray(sq, np.float32)[None]); stages[-1][3].qvel.assign(np.asarray(sv, np.float32)[None])
        gradient = []
        for step in reversed(range(horizon)):
            inputs, workspace, recorded, cotangent = stages[step]
            g = backward(BackwardCall(compiled, inputs, workspace, recorded, cotangent))
            gradient.insert(0, g.ctrl.numpy()[0].copy())
            if step:
                wp.copy(stages[step-1][3].qpos, g.qpos); wp.copy(stages[step-1][3].qvel, g.qvel)
        gradients[name] = np.asarray(gradient)
        adjoints[name] = float(np.sum(gradients[name].astype(np.float64)*du))
    epsilons = (.03, .01, .003, .001, .0003, .0001, .00003, .00001, .000003, .000001, .0000003)
    report = dict(case=str(root.resolve()), horizon=horizon, dt=cfg.simulator.time_grid.mpc_dt,
        native_step_profile=profile, integration_metadata=integration_metadata,
        initial_qpos_fixed=True, optimizer_modified=False,
        original_seed=seed, adjoints=adjoints,
        base_repeats_max_qpos_difference=max(float(np.max(np.abs(base["qpos"]-other["qpos"]))) for other in repeated),
        base_repeats_max_qvel_difference=max(float(np.max(np.abs(base["qvel"]-other["qvel"]))) for other in repeated),
        gradient_linearity_mixed_qpos_plus_qvel_abs_max=float(np.max(np.abs(gradients["mixed"]-gradients["qpos_only"]-gradients["qvel_only"]))),
        trials=[])
    saved = dict(qbase=qbase, vbase=vbase, control_base=ubase, direction=du, qseed=qseed, vseed=vseed,
        base_qpos=base["qpos"], base_qvel=base["qvel"], base_slopes=base["slopes"], base_projection=base["projection"], base_force=base["force"])
    for epsilon in epsilons:
        controls_low, controls_high = (ubase-epsilon*du).astype(np.float32), (ubase+epsilon*du).astype(np.float32)
        low, high = evaluate(controls_low), evaluate(controls_high)
        effective = (controls_high.astype(np.float64)-controls_low.astype(np.float64))/(2*epsilon)
        dq = (high["qpos"][-1].astype(np.float64)-low["qpos"][-1].astype(np.float64))/(2*epsilon)
        dv = (high["qvel"][-1].astype(np.float64)-low["qvel"][-1].astype(np.float64))/(2*epsilon)
        trial = dict(epsilon=epsilon, control_direction_error_norm=float(np.linalg.norm(effective-du)),
            low_slopes_changed=int(np.count_nonzero(low["slopes"]!=base["slopes"])), high_slopes_changed=int(np.count_nonzero(high["slopes"]!=base["slopes"])),
            low_final_projection_changed=int(np.count_nonzero(low["projection"]!=base["projection"])), high_final_projection_changed=int(np.count_nonzero(high["projection"]!=base["projection"])),
            low_clamp_changed=int(np.count_nonzero(low["clamps"]!=base["clamps"])), high_clamp_changed=int(np.count_nonzero(high["clamps"]!=base["clamps"])),
            same_geometry=low["geometry"]==base["geometry"] and high["geometry"]==base["geometry"], finite_differences={}, quantized_direction_adjoint={})
        for name, (sq, sv) in seeds.items():
            trial["finite_differences"][name] = float(np.dot(dq,sq)+np.dot(dv,sv))
            trial["quantized_direction_adjoint"][name] = float(np.sum(gradients[name].astype(np.float64)*effective))
        report["trials"].append(trial)
        for prefix, values in (("low",low),("high",high)):
            for name in ("qpos","qvel","slopes","projection","force"):
                saved[f"eps_{epsilon}_{prefix}_{name}"] = values[name]
        print(f"eps {epsilon:g}: FDmixed={trial['finite_differences']['mixed']:.9g}, Newton changes {trial['low_slopes_changed']}/{trial['high_slopes_changed']}, final {trial['low_final_projection_changed']}/{trial['high_final_projection_changed']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"native_horizon_{horizon}.npz", **saved, **{f"grad_{k}":v for k,v in gradients.items()})
    write_json(output / f"native_horizon_{horizon}.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/grasp_forceaware_gradient_audit"))
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--servo-integration", action="store_true", help="Explicitly audit the local PD integration extension in a separate output subdirectory")
    args = parser.parse_args()
    run(args.root,args.output,horizon=args.horizon,servo_integration=args.servo_integration)
