import numpy as np
import pytest
import torch

from retarget.env.grasp.template_mppi_size_ik import (
    SizeIKSettings, transport_contacts, bounded_lm, CylinderSideProjection,
)
from retarget.env.grasp.template_mppi import TemplateMPPISettings
from retarget.env.grasp.audit import GraspAudit
from retarget.env.grasp.primitives import primitive_geometry
from retarget.env.geometry import ObjectMeshSurface


def test_transport_uses_physical_dimensions_and_does_not_scale_hand_coordinates():
    p = np.array([[.01, -.02, .03]])
    canonical = dict(kind='cylinder', radius_m=.02, height_m=.1)
    target = dict(kind='cylinder', radius_m=.04, height_m=.05)
    np.testing.assert_allclose(transport_contacts(p, canonical, target), [[.02, -.04, .015]])
    np.testing.assert_array_equal(p, [[.01, -.02, .03]])
    assert TemplateMPPISettings().pose_weight == 0.


def test_bounded_lm_accepts_only_descent_and_can_open_beyond_seed_angles():
    def residual(q):
        return q-torch.tensor([1.2, -.5], dtype=q.dtype, device=q.device)
    seeds = torch.tensor([[0., 0.], [.1, .2], [1.4, -1.]], dtype=torch.float64)
    low, high = torch.tensor([0., -.4]), torch.tensor([1., .5])
    q, cost, history = bounded_lm(residual, seeds, low, high, SizeIKSettings(iterations=12, starts=3))
    assert bool((history[1:] <= history[:-1]+1.e-12).all())
    assert bool((q >= low).all() and (q <= high).all())
    torch.testing.assert_close(q, torch.tensor([[1., -.4]]*3, dtype=q.dtype))
    assert bool((cost <= history[0]).all())
    assert bool((cost[:2] < history[0, :2]).all())


def test_side_projection_cannot_substitute_a_disk_cap_for_finger_side_contact():
    vertices, faces = primitive_geometry('disk', .04, .015)
    surface = ObjectMeshSurface(vertices, faces)
    points = torch.tensor([[[.01, 0., .009], [.039, 0., 0.]]])
    p, n, d = GraspAudit(surface, 'cpu').query(points[0])
    assert abs(float(n[0, 2])) == pytest.approx(1.)
    side = CylinderSideProjection(surface, 'cpu', [0])
    p2, n2, d2 = side.apply(points, p[None], n[None], d[None])
    assert abs(float(n2[0, 0, 2])) < 1.e-6
    assert float(d2[0, 0]) > .02
    torch.testing.assert_close(p2[0, 1], p[1])


def test_wrapped_disk_requires_one_fixed_side_slot_per_finger(tmp_path):
    from retarget.config.run import ROOT, load
    from retarget.env.grasp.template_mppi_runner import validate
    from retarget.env.grasp.template_mppi_inputs import load_template_contacts
    from retarget.env.grasp.template_mppi import TemplatePoseObjective
    from retarget.env.grasp.primitives import primitive_mesh
    args = validate(load(ROOT/'configs/grasp_template_mppi_all.yaml'))
    selected = []
    for radius in (.04, .06):
        mesh = primitive_mesh(tmp_path/f'disk_{radius}.obj', 'disk', radius, .015)
        objective = TemplatePoseObjective(mesh, device='cpu')
        template = load_template_contacts('10_Power_Disk', args.dexonomy_root,
                                         objective.cpu, device='cpu')
        before = template.local_points.copy(), template.local_normals.copy()
        q = objective.cpu.qpos0.copy()
        q[6:] = template.finger_qpos
        objective.bind(template, q)
        objective.configure_object_regions('10_Power_Disk', dict(kind='disk'))
        slots = objective.side_projection.slots.tolist()
        assert len(slots) == 5 < len(template.body_names)
        assert len({template.body_names[i].split('_link')[0] for i in slots}) == 5
        np.testing.assert_array_equal(template.local_points, before[0])
        np.testing.assert_array_equal(template.local_normals, before[1])
        selected.append(slots)
    assert selected[0] == selected[1]


def test_prismatic_posture_prevents_reverse_flexion_without_locking_opening(tmp_path):
    from retarget.env.grasp.template_mppi import TemplatePoseObjective
    from retarget.env.grasp.template_mppi_inputs import TemplateContacts
    from retarget.env.grasp.primitives import primitive_mesh
    mesh = primitive_mesh(tmp_path/'cylinder.obj', 'cylinder', .02, .18)
    objective = TemplatePoseObjective(mesh, device='cpu')
    template = TemplateContacts(('right_finger2_link4',), [[0.,0.,.01]], [[0.,1.,0.]],
        np.zeros(20), (('right_finger2_link4',),), {})
    q = objective.cpu.qpos0.copy()
    objective.bind(template,q)
    for name in ('6_Prismatic_4_Finger', '7_Prismatic_3_Finger'):
        objective.configure_object_regions(name,dict(kind='cylinder'))
        lower,upper = objective.qlo.clone(),objective.qhi.clone()
        indices = objective.grasp_posture_bounds['joint_indices']
        assert bool((lower[indices] >= 0).all())
        assert bool((upper[indices] > 1.).all())  # Free opening/closing, no fixed angle.
        reverse = torch.as_tensor(q, dtype=torch.float32)
        reverse[12],reverse[13] = -.9,1.55
        projected = objective.project(reverse)
        assert projected[12] == 0 and projected[13] == pytest.approx(1.55)
        assert objective.grasp_posture_bounds['reference_pose_used'] is False
    objective.configure_object_regions('8_Prismatic_2_Finger',dict(kind='cylinder'))
    assert objective.grasp_posture_bounds == {} and objective.qlo[12] < 0


@pytest.mark.parametrize('values', [dict(starts=0),dict(iterations=True),dict(damping=-1),dict(enabled=1)])
def test_invalid_ik_options_are_rejected(values):
    with pytest.raises(ValueError):
        SizeIKSettings(**values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_native_collision_linearization_matches_actual_cuda_distance_gradient(tmp_path):
    import trimesh
    from retarget.env.grasp.template_mppi import TemplatePoseObjective
    from retarget.env.grasp.template_mppi_inputs import TemplateContacts
    from retarget.env.grasp.template_mppi_size_ik import NativeCollisionLinearization
    from retarget.env.grasp.template import WujiContactKinematics
    from retarget.env.grasp.hand import HAND_BODIES
    mesh = tmp_path/'box.obj'
    trimesh.creation.box([.03]*3).export(mesh)
    o = TemplatePoseObjective(mesh)
    template = TemplateContacts(('right_palm_link',), [[0, .02, -.04]], [[0, 1, 0]],
        np.zeros(20), (('right_palm_link',),), {})
    o.bind(template, o.cpu.qpos0)
    o.set_batch_size(4)
    fk = WujiContactKinematics(o.cpu, [o.cpu.body(n).id for n in HAND_BODIES], o.device)
    oracle = NativeCollisionLinearization(o, fk)
    q = fk.tensor(o.cpu.qpos0)
    for offset in (0., .01, .02, .03):
        q[0] = offset
        frozen = oracle(q[None].expand(4, -1))
        if float(frozen['distance'][0, 0]) < -.003:
            break
    else:
        pytest.fail('Test geometry must provide an active hand-object collision')
    row = {k: v[0] for k, v in frozen.items()}
    jac = torch.func.jacfwd(lambda x: oracle.residual(x, row))(q)
    axis = int(jac[0, :3].abs().argmax())
    assert abs(float(jac[0, axis])) > 1000
    values = []
    for sign in (-1, 1):
        trial = q.clone()
        trial[axis] += sign*1.e-4
        trial_frozen = oracle(trial[None].expand(4, -1))
        values.append(float(oracle.residual(trial, {k:v[0] for k,v in trial_frozen.items()})[0]))
    finite_difference = (values[1]-values[0])/2.e-4
    assert float(jac[0, axis]) == pytest.approx(finite_difference, rel=.02, abs=100)
    assert float(jac[1, :3].abs().max()) < 1.e-6


def test_palmar_pinch_keeps_both_original_slots_on_cylinder_side(tmp_path):
    from retarget.config.run import ROOT
    from retarget.env.grasp.template_mppi_inputs import load_template_contacts
    from retarget.env.grasp.template_mppi import TemplatePoseObjective
    from retarget.env.grasp.primitives import primitive_mesh
    mesh=primitive_mesh(tmp_path/'cylinder.obj','cylinder',.018,.12)
    objective=TemplatePoseObjective(mesh,device='cpu')
    template=load_template_contacts('9_Palmar_Pinch',ROOT/'data/grasp/templates',objective.cpu,device='cpu')
    before=template.to_dict();q=np.r_[np.zeros(6),template.finger_qpos]
    objective.bind(template,q);objective.configure_object_regions('9_Palmar_Pinch',dict(kind='cylinder'))
    assert objective.side_projection.slots.tolist()==[0,1]
    assert objective.antipodal_weight==1. and objective.posture is not None
    assert float(objective.low[13])>=0  # Index DIP can no longer reverse 45 degrees.
    assert template.to_dict()==before


def test_opposed_pinch_seeds_preserve_anchors_and_center_in_object(tmp_path):
    from retarget.config.run import ROOT
    from retarget.env.grasp.template_mppi_inputs import load_template_contacts
    from retarget.env.grasp.template_mppi import TemplatePoseObjective
    from retarget.env.grasp.template_mppi_pinch import opposed_seeds
    from retarget.env.grasp.primitives import primitive_mesh
    from scipy.spatial.transform import Rotation
    import mujoco
    objective=TemplatePoseObjective(primitive_mesh(tmp_path/'cylinder.obj','cylinder',.018,.12),device='cpu')
    template=load_template_contacts('9_Palmar_Pinch',ROOT/'data/grasp/templates',objective.cpu,device='cpu')
    before=template.to_dict();seeds,metadata=opposed_seeds(objective.cpu,template)
    assert .004 < metadata['original_anchor_separation_m'] < .006
    assert metadata['hand_anchors_changed'] is False and len(seeds)==4
    for q in seeds:
        np.testing.assert_array_equal(q[6:],template.finger_qpos)
        data=mujoco.MjData(objective.cpu);data.qpos[:]=q;mujoco.mj_kinematics(objective.cpu,data)
        ids=[objective.cpu.body(n).id for n in template.body_names]
        p=data.xpos[ids]+np.einsum('nij,nj->ni',data.xmat[ids].reshape(-1,3,3),template.local_points)
        n=np.einsum('nij,nj->ni',data.xmat[ids].reshape(-1,3,3),template.local_normals)
        np.testing.assert_allclose(p.mean(0),0,atol=1e-8)
        assert n[0,0] < -.96 and n[1,0] > .96
        assert np.linalg.det(Rotation.from_euler('XYZ',q[3:6]).as_matrix())==pytest.approx(1)
    assert template.to_dict()==before
