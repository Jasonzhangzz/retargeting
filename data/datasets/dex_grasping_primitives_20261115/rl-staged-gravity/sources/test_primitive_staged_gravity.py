"""Stability-gate checks independent of GPU simulation and PPO training."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts'
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT))
spec = importlib.util.spec_from_file_location('staged_gravity', SCRIPT / 'refine_primitives_staged_gravity.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_controller(mode='disturbance'):
    n = 12
    root = torch.zeros(n, 13)
    root[:, 3] = 1
    robot = NS(joint_pos=torch.zeros(n, 20), joint_vel=torch.zeros(n, 20),
               default_joint_pos=torch.zeros(n, 20))
    obj = NS(default_root_state=root, root_link_pose_w=root[:, :7].clone(),
             root_link_pos_w=root[:, :3].clone(), root_link_lin_vel_w=torch.zeros(n, 3),
             root_link_ang_vel_w=torch.zeros(n, 3))
    contacts = NS(found=torch.ones(n, 3), dist=torch.zeros(n, 3))
    scene = dict(robot=NS(data=robot), object=NS(data=obj, indexing=NS(body_ids=[0])),
                 grasp_contacts=NS(data=contacts, primary_names=['right_finger1_link3',
                                    'right_finger2_link3', 'right_finger2_tip_sensor'],
                                   cfg=NS(num_slots=1)))
    class Scene(dict):
        pass
    scene = Scene(scene)
    scene.env_origins = torch.zeros(n, 3)
    sim = NS(data=NS(xfrc_applied=torch.zeros(n, 1, 6)), step=lambda: None)
    env = NS(num_envs=n, device='cpu', scene=scene, sim=sim, step_dt=.02,
             primitive_radius=.04, primitive_mass=.04,
             episode_length_buf=torch.zeros(n, dtype=torch.long))
    args = NS(stable_s=.3, min_prepare_s=.5, max_prepare_s=3., ramp_s=.5,
              probe_s=2., pulse_s=.4, mass=.04, disturbance_g=1.,
              gravity_mode=mode, joint_speed=.3)
    return env, module.StabilityGravityController(env, args)


def test_release_requires_continuous_stability_and_no_support_during_probe():
    env, controller = make_controller()
    for step in range(25):
        env.episode_length_buf.fill_(step)
        controller.update_gate()
        assert (controller.release_step < 0).all()
    env.episode_length_buf.fill_(25)
    controller.update_gate()
    assert (controller.release_step == 25).all()
    controller.physics_step()
    assert torch.allclose(env.sim.data.xfrc_applied[:, 0, 2], torch.full((12,), .04 * 9.81))
    env.episode_length_buf.fill_(73)  # .96s after release: support gone, pulse near peak.
    controller.update_gate()
    controller.physics_step()
    assert (controller.support_weight == 0).all()
    wrench = env.sim.data.xfrc_applied[:, 0].clone()
    wrench[:, 2] -= .04 * 9.81  # remove known gravity compensation
    assert (wrench[:6, :3].norm(dim=-1) > .38).all()
    assert (wrench[6:, 3:].norm(dim=-1) > .007).all()
    assert (wrench[:6, 3:] == 0).all()
    assert torch.allclose(wrench[6:, :3], torch.zeros(6, 3), atol=1e-7)
    controller.reset(None)
    assert (controller.release_step == -1).all()
    assert (controller.peak_disturbance_force == 0).all()


def test_lost_contact_restarts_gate_and_timeout_never_forces_release():
    env, controller = make_controller()
    for step in range(20):
        env.episode_length_buf.fill_(step)
        controller.update_gate()
    env.scene['grasp_contacts'].data.found[:, 2] = 0
    env.episode_length_buf.fill_(20)
    controller.update_gate()
    assert (controller.stable_steps == 0).all()
    for step in range(21, 170):
        env.episode_length_buf.fill_(step)
        controller.update_gate()
    assert controller.timed_out.all()
    assert (controller.release_step == -1).all()
    env.scene['grasp_contacts'].data.found.fill_(1)
    for step in range(170, 200):
        env.episode_length_buf.fill_(step)
        controller.update_gate()
    assert (controller.release_step == -1).all()


def test_staged_mode_removes_gravity_compensation_only_after_release():
    env, controller = make_controller('staged')
    for step in range(26):
        env.episode_length_buf.fill_(step)
        controller.update_gate()
    env.episode_length_buf.fill_(60)
    controller.update_gate()
    controller.physics_step()
    assert torch.equal(env.sim.data.xfrc_applied, torch.zeros_like(env.sim.data.xfrc_applied))


def test_native_mesh_restores_original_adjacent_link_exclusions():
    from dex_supermimic.assets.wuji_hand2 import _get_spec, wuji_hand2_xml_path
    from dex_supermimic.tasks.dex_reorient.config.wuji_hand2.hand2_constants import (
        apply_hand2_reorient_collision_overlay,
    )
    from refine_primitives_supermimic import native_collision_spec
    native = _get_spec(wuji_hand2_xml_path())
    expected = {(x.bodyname1, x.bodyname2) for x in native.excludes}
    restored = native_collision_spec(lambda: apply_hand2_reorient_collision_overlay(native))
    actual = {(x.bodyname1, x.bodyname2) for x in restored.excludes}
    assert expected and actual == expected
    assert all(g.contype == 0 for g in restored.geoms if g.name.endswith('_capsule_disabled'))


def test_virtual_mode_applies_no_actual_probe_and_gravity_lp_is_not_closure():
    env, controller = make_controller('virtual')
    env.gravity_controller = controller
    for step in range(26):
        env.episode_length_buf.fill_(step)
        controller.update_gate()
    env.episode_length_buf.fill_(73)
    controller.update_gate()
    controller.physics_step()
    expected = torch.zeros_like(env.sim.data.xfrc_applied)
    expected[:, 0, 2] = .04 * 9.81  # only cancels physical gravity
    assert torch.allclose(env.sim.data.xfrc_applied, expected)
    assert (controller.peak_disturbance_force == 0).all()
    contact = env.scene['grasp_contacts'].data
    contact.normal = torch.tensor([0., 0., 1.]).expand(12, 3, 3)
    contact.tangent = torch.tensor([1., 0., 0.]).expand(12, 3, 3)
    contact.pos = torch.zeros(12, 3, 3)
    contact.force = torch.tensor([.5, 0., 0.]).expand(12, 3, 3)
    feasible = module.virtual_wrench_lp(env)
    assert feasible[:, 0].all()  # can support virtual gravity
    assert not feasible.all(-1).any()  # cannot resist any torque
    assert torch.isfinite(module.virtual_wrench_proxy(env)).all()
    contact.force = torch.zeros(12, 3, 3)
    assert not module.virtual_wrench_lp(env)[:, 0].any()  # no preload -> no load capacity


if __name__ == '__main__':
    test_release_requires_continuous_stability_and_no_support_during_probe()
    test_lost_contact_restarts_gate_and_timeout_never_forces_release()
    test_staged_mode_removes_gravity_compensation_only_after_release()
    test_native_mesh_restores_original_adjacent_link_exclusions()
    test_virtual_mode_applies_no_actual_probe_and_gravity_lp_is_not_closure()
    print('5 stability/gravity, virtual wrench and native collision checks passed')
