"""Small forward checks; no full reference sequence is optimized."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest, torch, warp as wp, mujoco
from retarget.config.run import ROOT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_spider_rollout_aggregate_reference_indices_and_restore():
    from retarget.env.humanoid.runtime import HumanoidEnvironment
    from retarget.task.humanoid import cost

    t = HumanoidEnvironment(
        "move_largebox", ROOT / "data/humanoid/move_largebox", batch=4
    )
    c = t.config
    with torch.no_grad(), torch.cuda.stream(t.stream):
        values = torch.zeros((3, t.knots * c.nu), device=t.device)
        values[:, 0] = torch.tensor([0.0, 0.1, 0.2], device=t.device)
        before = {
            k: getattr(t.env.data_wp, k).numpy().copy()
            for k in ("qpos", "qvel", "qacc_warmstart", "ctrl", "time")
        }
        margin = t.env.model_wp.pair_margin.numpy().copy()
        scored = t.score(values)
        rews = []
        for step in range(c.horizon_steps):
            env = SimpleNamespace(
                data_wp=SimpleNamespace(
                    qpos=wp.from_torch(t.q[step + 1]), qvel=wp.from_torch(t.v[step + 1])
                )
            )
            reference = [r[step + 1] for r in t.refs]
            rew, _ = (
                cost.get_reward
                if step < c.horizon_steps - 1
                else cost.get_terminal_reward
            )(c, env, reference)
            rews.append(rew)
        torch.testing.assert_close(
            scored, (-torch.stack(rews).mean(0)[:3]).double(), rtol=1e-6, atol=1e-6
        )
        for key, expected in before.items():
            np.testing.assert_array_equal(getattr(t.env.data_wp, key).numpy(), expected)
        np.testing.assert_array_equal(t.env.model_wp.pair_margin.numpy(), margin)
        # Splitting candidate count from world count is only a dispatch change.
        many = t.score(torch.cat((values, values, values[:1])))
        # MJWarp floating-point contact reductions vary with world placement.
        error = float((many[:3] - many[3:6]).abs().max())
        assert error < 0.002
        q = torch.zeros((1, 8, 29), device=t.device)
        q[:, 0, 0] = 1
        from retarget.env.humanoid.interp import interp

        torch.testing.assert_close(
            t.controls(q.flatten(1)), t.nominal + interp(q * 0.05, c.knot_steps)
        )
    torch.cuda.synchronize(t.device)


def test_both_hand_models_load_with_same_control_shape():
    from retarget.env.grasp.hand import hand_tree
    from retarget.env.mjwarp import load_model

    for generation in ("wuji", "wuji2"):
        path = ROOT / "data/grasp/assets" / generation / "right.xml"
        device, model = load_model(str(path), device="cuda:0")
        assert model.nu == 26 and model.nq == 26
        assert model.body("right_palm_link").id >= 0
        assert np.isfinite(model.actuator_gainprm).all()


def test_full_grasp_batch_and_independent_servo_offsets():
    from retarget.config.run import load
    from retarget.env.grasp.runner import arguments
    from retarget.env.grasp.runtime import GraspRuntime
    from retarget.env.grasp.rollout import GraspProblem
    from retarget.env.grasp.inputs import StaticHand
    from retarget.env.geometry import ObjectMeshSurface
    from retarget.env.grasp.hand import HAND_BODIES

    folder = ROOT / "output/smoke_grasp_cem"
    inputs = list(folder.glob("**/input_00/card_input.npz"))
    if not inputs:
        pytest.skip("Run the grasp smoke config to create an initialized environment")
    experiment = inputs[0].parent
    args = arguments(load(folder / "config.yaml"))
    args.worlds = 4
    runtime = GraspRuntime(experiment, args)
    model = runtime.model
    raw = np.load(experiment / "contactaware/contact_guidance.npz")
    # The full hand samples and semantic contact data define the same task.
    hand = StaticHand(
        model,
        ObjectMeshSurface.from_model(model, runtime.index.object_body),
        raw["hand_query_body_ids"][0],
        raw["hand_query_local_pos"][0],
        np.zeros_like(raw["hand_query_local_pos"][0]),
    )
    import trimesh

    mesh_data = trimesh.load(experiment / "object.obj", process=False)
    mesh = wp.Mesh(
        points=wp.array(mesh_data.vertices, dtype=wp.vec3, device="cuda:0"),
        indices=wp.array(mesh_data.faces.reshape(-1), dtype=int, device="cuda:0"),
    )
    runtime.audit_context = (SimpleNamespace(mesh=mesh), hand, args)
    with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        p = GraspProblem(runtime)
        x = p.mean.repeat(3, 1)
        assert x.shape == (3, 46)
        x[1, 26:] += 0.1
        e = p.evaluate(x)
        torch.cuda.synchronize()
        assert torch.isfinite(wp.to_torch(p.explicit.cost)[:3]).all()
        np.testing.assert_allclose(p.state.x.numpy()[1, 26:], 0.1, atol=1e-7)
        q = p.state.qpos.numpy()
        np.testing.assert_array_equal(q[0], q[1])
        assert not np.array_equal(
            p.explicit.hold.numpy()[0], p.explicit.hold.numpy()[1]
        )
        assert np.isfinite(p.explicit.qpos[-1].numpy()[:3]).all()


def test_comfree_uses_mjwarp_collision_and_integrates(monkeypatch):
    from mujoco_warp._src import collision_driver
    import mujoco_warp as mjw
    from retarget.env.mjwarp import ComFreeEnvironment

    cpu = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".01" jacobian="dense"/>
        <worldbody><geom type="plane" size="1 1 .1"/>
        <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1"/></body>
        </worldbody></mujoco>""")
    wp.set_device("cuda:0")
    model = mjw.put_model(cpu)
    env = ComFreeEnvironment(cpu, model, 2)
    calls = []
    original = collision_driver.collision

    def collision(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(collision_driver, "collision", collision)
    before = env.data.qpos.numpy().copy()
    env.evaluate_dynamics()
    assert calls and (env.data.nefc.numpy() > 0).all()
    assert (env.data.qfrc_constraint.numpy()[:, 2] > 0).all()
    assert (model.geom_solimp.numpy()[..., 2] == np.float32(0.01)).all()
    env.integrate()
    assert np.isfinite(env.data.qpos.numpy()).all()
    assert not np.array_equal(env.data.qpos.numpy(), before)
    q = before.copy()
    q[:, 2] = 0.5
    env.data.qpos.assign(q)
    env.data.qvel.zero_()
    env.evaluate_dynamics()
    assert (env.data.nefc.numpy() == 0).all()
    np.testing.assert_array_equal(env.data.qfrc_constraint.numpy(), 0.0)


def test_mjwarp_fk_matches_cpu_for_both_hands():
    from retarget.env.mjwarp import (
        load_model,
        KinematicModel,
        allocate_output,
        kinematics,
    )

    for generation in ("wuji", "wuji2"):
        model, cpu = load_model(
            ROOT / "data/grasp/assets" / generation / "right.xml", device="cuda:0"
        )
        fk = KinematicModel(cpu, model)
        pose = allocate_output(fk, 3)
        data = mujoco.MjData(cpu)
        data.qpos[:] += np.random.default_rng(7).normal(0, 0.05, cpu.nq)
        q = np.tile(data.qpos, (3, 1)).astype(np.float32)
        kinematics(fk, wp.array(q, device="cuda:0"), pose)
        data.qpos[:] = q[0]
        mujoco.mj_kinematics(cpu, data)
        np.testing.assert_allclose(pose.body_position.numpy()[0], data.xpos, atol=2e-7)
        np.testing.assert_allclose(
            pose.body_matrix.numpy()[0], data.xmat.reshape(-1, 3, 3), atol=1e-6
        )
        np.testing.assert_allclose(
            pose.joint_anchor.numpy()[0], data.xanchor, atol=2e-7
        )
        np.testing.assert_allclose(pose.joint_axis.numpy()[0], data.xaxis, atol=1e-6)
