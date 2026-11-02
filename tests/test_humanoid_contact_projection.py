"""Dataset contact seeds are surface projections and leave execution untouched."""

from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp

from retarget.env.humanoid.contact_projection import ContactProjector, build_contact_candidates


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    wp.init()
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    return request.param


def _environment(device, *, rotation="1 0 0 0", child=False,
                 hand_position="0 0 0", hand_radius=.02, site_position="0 0 0"):
    box = f'<geom name="box" type="box" size=".1 .2 .3" quat="{rotation}" contype="0" conaffinity="0"/>'
    if child:
        box = f'<body name="fixed_child" pos="0 .1 0">{box}</body>'
    cpu = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <worldbody>
        <body name="robot" pos="0 0 0"><freejoint/>
          <geom name="hand" type="sphere" size="{hand_radius}" pos="{hand_position}"/>
          <site name="left_palm" pos="{site_position}"/>
        </body>
        <body name="object" pos=".4 0 0"><freejoint/>
          <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
          {box}
        </body>
      </worldbody>
      <contact><pair geom1="hand" geom2="box"/></contact>
    </mujoco>""")
    data = mujoco.MjData(cpu)
    mujoco.mj_forward(cpu, data)
    with wp.ScopedDevice(device):
        model_wp = mjw.put_model(cpu)
        data_wp = mjw.put_data(cpu, data, nconmax=1, njmax=1)
    env = SimpleNamespace(model_cpu=cpu, model_wp=model_wp, data_wp=data_wp)
    config = SimpleNamespace(device=device)
    return config, env


def test_reference_fk_projection_neighbors_and_live_state(device):
    config, env = _environment(device)
    before = {name: getattr(env.data_wp, name).numpy().copy() for name in ("qpos", "qvel", "ctrl", "time", "site_xpos")}
    reference = wp.to_torch(env.data_wp.qpos).repeat(3, 1)
    reference[1, :3] += torch.tensor([0.4, 0.0, 0.8], device=device)
    reference[1, -7:-4] += torch.tensor([0.4, 0.0, 0.8], device=device)
    # Change the hand relative to the box in the final reference frame.
    reference[2, 1] += 0.35
    result = build_contact_candidates(config, env, reference, [0], radius=0.025, seed=91)
    assert result.local_points.shape == (3, 1, 11, 3)
    assert result.object_qpos_adr == 7
    assert result.object_dof_adr == 6
    assert result.hand_site_ids == (0,)
    torch.testing.assert_close(result.local_points[:2, 0, 0], torch.tensor([[-.1, 0, 0], [-.1, 0, 0]], device=device), atol=2e-6, rtol=0)
    torch.testing.assert_close(result.local_normals[:2, 0, 0], torch.tensor([[-1., 0, 0], [-1., 0, 0]], device=device), atol=1e-6, rtol=0)
    assert float(result.reference_hand_positions[1, 0, 2]) == pytest.approx(0.8, abs=1e-6)
    limits = torch.tensor([.1, .2, .3], device=device)
    point = result.local_points.abs()
    assert bool((point <= limits + 1e-6).all())
    assert bool(((point - limits).abs().amin(-1) < 2e-6).all())
    torch.testing.assert_close(result.local_normals.norm(dim=-1), torch.ones((3, 1, 11), device=device))
    assert bool(((result.local_points[..., 1:, :] - result.local_points[..., :1, :]).norm(dim=-1) <= .02501).all())
    again = build_contact_candidates(config, env, reference, [0], radius=.025, seed=91)
    torch.testing.assert_close(again.local_points, result.local_points, atol=0, rtol=0)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env.data_wp, name).numpy(), value)


def test_rotated_geom_and_fixed_descendant_transform(device):
    _, env = _environment(device, rotation=".707106781 0 0 .707106781", child=True)
    projector = ContactProjector(env.model_cpu, env.model_cpu.body("object").id, device=device)
    query = torch.tensor([[-1., .1, 0], [-.2, .1, 0]], device=device)
    point, normal = projector.project(query)
    # A 90-degree z rotation maps the box's y half-width .2 onto world x.
    torch.testing.assert_close(point, torch.tensor([[-.2, .1, 0], [-.2, .1, 0]], device=device), atol=2e-6, rtol=0)
    torch.testing.assert_close(normal, torch.tensor([[-1., 0, 0], [-1., 0, 0]], device=device), atol=2e-6, rtol=0)


def test_projection_rejects_invalid_reference(device):
    config, env = _environment(device)
    qpos = wp.to_torch(env.data_wp.qpos).clone()
    with pytest.raises(ValueError, match="hand_site_ids"):
        build_contact_candidates(config, env, qpos, [])
    qpos[0, 0] = torch.nan
    with pytest.raises(ValueError, match="reference_qpos"):
        build_contact_candidates(config, env, qpos, [0])


def test_mesh_surface_with_inward_source_winding(device):
    # Deliberately reversed tetrahedron winding; repaired normals must point
    # away from its interior independently of the hand's approach direction.
    cpu = mujoco.MjModel.from_xml_string("""<mujoco>
      <asset><mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="0 1 2  0 3 1  0 2 3  1 3 2"/></asset>
      <worldbody><body name="object"><freejoint/>
        <geom type="mesh" mesh="tetra"/>
      </body></worldbody>
    </mujoco>""")
    projector = ContactProjector(cpu, cpu.body("object").id, device=device)
    point, normal = projector.project(torch.tensor([[.2, .2, -.2], [.2, .2, 0]], device=device))
    torch.testing.assert_close(point, torch.tensor([[.2, .2, 0], [.2, .2, 0]], device=device), atol=2e-6, rtol=0)
    torch.testing.assert_close(normal, torch.tensor([[0., 0, -1], [0., 0, -1]], device=device), atol=2e-6, rtol=0)


def test_reference_candidate_ids_reuse_offsets_over_time(device):
    config, env = _environment(device)
    reference = wp.to_torch(env.data_wp.qpos).repeat(4, 1)
    result = build_contact_candidates(config, env, reference, [0], seed=32)
    torch.testing.assert_close(result.local_points, result.local_points[:1].expand_as(result.local_points), atol=0, rtol=0)
    # The standalone sampler retains independently sampled queries by default.
    projector = ContactProjector(env.model_cpu, env.model_cpu.body("object").id, device=device)
    query = torch.tensor([[[-.4, 0, 0]]], device=device).expand(4, -1, -1)
    independent, _ = projector.candidates(query, seed=32)
    assert not torch.equal(independent[0, :, 1:], independent[1, :, 1:])
    shared, _ = projector.candidates(query, seed=32, shared_over_time=True)
    torch.testing.assert_close(shared, shared[:1].expand_as(shared), atol=0, rtol=0)


def test_hand_collision_surface_is_used_instead_of_displaced_palm_site(device):
    config, env = _environment(device, hand_position=".1 0 0", hand_radius=.05,
                               site_position=".08 0 0")
    reference = wp.to_torch(env.data_wp.qpos).repeat(2, 1)
    # Rotate and move the whole pair in the second frame. Robot body-local
    # contact coordinates must remain the same although world positions move.
    reference[1, 3:7] = reference.new_tensor([2**-.5, 0, 0, 2**-.5])
    reference[1, -7:-4] = reference.new_tensor([0, .4, 0])
    reference[1, -4:] = reference[1, 3:7]
    surface = build_contact_candidates(config, env, reference, [0])
    torch.testing.assert_close(surface.robot_local_points[:, 0],
                               reference.new_tensor([[.15, 0, 0], [.15, 0, 0]]), atol=2e-6, rtol=0)
    torch.testing.assert_close(surface.reference_hand_positions[:, 0],
                               reference.new_tensor([[.15, 0, 0], [0, .15, 0]]), atol=2e-6, rtol=0)
    torch.testing.assert_close(surface.local_points[:, 0, 0],
                               reference.new_tensor([[-.1, 0, 0], [-.1, 0, 0]]), atol=2e-6, rtol=0)
    radial = surface.robot_local_points[:, 0] - reference.new_tensor([.1, 0, 0])
    torch.testing.assert_close(radial.norm(dim=-1), reference.new_full((2,), .05), atol=2e-6, rtol=0)
    site = build_contact_candidates(config, env, reference, [0], use_hand_surface=False)
    torch.testing.assert_close(site.robot_local_points[:, 0],
                               reference.new_tensor([[.08, 0, 0], [.08, 0, 0]]), atol=2e-6, rtol=0)
    assert bool((surface.reference_hand_positions-site.reference_hand_positions).norm(dim=-1).gt(.069).all())


@pytest.mark.parametrize("kind", ["box", "mesh", "multiple"])
def test_hand_box_mesh_and_multiple_collision_geoms(device, kind):
    asset = ""
    if kind == "mesh":
        import trimesh
        mesh = trimesh.creation.box(extents=[.1, .2, .2])
        vertices = " ".join(str(value) for value in mesh.vertices.flatten())
        faces = " ".join(str(value) for value in mesh.faces.flatten())
        asset = f'<asset><mesh name="hand_mesh" vertex="{vertices}" face="{faces}"/></asset>'
        hand = '<geom type="mesh" mesh="hand_mesh" pos=".1 0 0" mass="1"/>'
    elif kind == "box":
        hand = '<geom type="box" size=".05 .1 .1" pos=".1 0 0" mass="1"/>'
    else:
        hand = '<geom type="sphere" size=".02" pos="-.1 0 0" mass="1"/><geom type="sphere" size=".05" pos=".1 0 0" mass="1"/>'
    cpu = mujoco.MjModel.from_xml_string(f"""<mujoco>{asset}<worldbody>
      <body name="robot"><freejoint/>{hand}<site name="palm" pos=".08 0 0"/></body>
      <body name="object" pos=".4 0 0"><freejoint/><geom type="box" size=".1 .2 .3" mass="1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(cpu)
    with wp.ScopedDevice(device):
        env = SimpleNamespace(model_cpu=cpu, model_wp=mjw.put_model(cpu), data_wp=mjw.put_data(cpu, data, nconmax=1, njmax=1))
    qpos = wp.to_torch(env.data_wp.qpos).clone()
    result = build_contact_candidates(SimpleNamespace(device=device), env, qpos, [0])
    torch.testing.assert_close(result.robot_local_points[0, 0], qpos.new_tensor([.15, 0, 0]), atol=2e-6, rtol=0)
    torch.testing.assert_close(result.local_points[0, 0, 0], qpos.new_tensor([-.1, 0, 0]), atol=2e-6, rtol=0)
