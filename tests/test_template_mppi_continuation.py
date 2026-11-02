"""Physical grid transport and inverse-wrist pose continuity regressions."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from retarget.env.grasp.template_mppi_continuation import (
    densify,outward_chains,pose_change,inverse_pose,MotionLimits,ContinuationObjective)
from retarget.env.grasp.primitives import size_specifications


def test_dense_grid_retains_ranges_and_physical_thicknesses():
    rows=size_specifications(22,'plate');dense=densify(rows,4)
    assert len(dense)==37
    assert dense[0]==rows[0] and dense[-1]==rows[-1]
    assert all(dense[4*i]==p for i,p in enumerate(rows))
    assert all(p['radius_m']==pytest.approx(.075) for p in dense)
    assert dense[1]['height_m']==pytest.approx(.0003125)
    assert dense[1]['dimensions_m'][2]==dense[1]['height_m']


def test_dense_plan_all_primary_families_has_640_sizes_and_exact_standards():
    from retarget.env.grasp.primitives import CATALOG
    count=0
    for typ,families in CATALOG.items():
        kind,r,h=families[0]
        if r is None:continue
        coarse=size_specifications(typ,kind);canonical=dict(coarse[0],size_scale=1.)
        rows,chains,pivot=outward_chains(densify(coarse),canonical)
        assert rows[pivot]['size_scale']==1.
        assert sorted(chains[0]+chains[1]+[pivot])==list(range(len(rows)))
        assert all(rows[a]['size_scale']>rows[b]['size_scale'] for a,b in zip(chains[0],chains[0][1:]))
        count+=len(rows)
    assert count==640


def test_inverse_pose_matches_cpu_and_is_differentiable():
    rng=np.random.default_rng(4);q=rng.normal(size=(10,26));q[:,:3]*=.1
    t,r=inverse_pose(torch.tensor(q,dtype=torch.float64))
    expected=Rotation.from_euler('XYZ',q[:,3:6]).as_matrix().transpose(0,2,1)
    np.testing.assert_allclose(r,expected,atol=1e-14)
    np.testing.assert_allclose(t,-np.einsum('nij,nj->ni',expected,q[:,:3]),atol=1e-14)
    x=torch.tensor(q[0],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda v:inverse_pose(v)[0],(x,))


def test_motion_measures_object_in_wrist_and_wrapped_rotation():
    p=np.zeros(26);p[0]=.1
    q=p.copy();q[5]=2*np.pi
    d=pose_change(p,q)
    assert d['object_translation_m']<1e-15 and d['object_rotation_deg']<1e-5
    q[5]=.02
    assert pose_change(p,q)['object_translation_m']==pytest.approx(2*.1*np.sin(.01))


def test_continuity_rejects_inverse_translation_even_if_hand_root_translation_static(tmp_path):
    from retarget.env.grasp.primitives import primitive_mesh
    p=np.zeros(26);p[0]=.1
    o=ContinuationObjective(primitive_mesh(tmp_path/'x.obj','cylinder',.01,.05),device='cpu',
        parent_q=p,limits=MotionLimits(.001,3.,8.))
    q=torch.tensor(p,dtype=torch.float32);q[5]=.03
    _,valid=o.continuity(q)
    assert not valid
    q[5]=0.;q[0]+=.0005
    assert o.continuity(q)[1]
    q[8]+=np.deg2rad(9.)
    assert not o.continuity(q)[1]


def test_continuation_bounds_are_parent_relative_after_recenter(tmp_path):
    from retarget.env.grasp.primitives import primitive_mesh
    p=np.zeros(26)
    o=ContinuationObjective(primitive_mesh(tmp_path/'x.obj','cylinder',.01,.05),device='cpu',parent_q=p)
    q=p.copy();q[:6]=.01
    o._search_bounds(q)
    torch.testing.assert_close(o.qlo[:6],torch.zeros(6))
    torch.testing.assert_close(o.qhi[:6],torch.zeros(6))
    assert float(o.qhi[10])==pytest.approx(np.deg2rad(8.))
    assert MotionLimits().joint_deg==8.


@pytest.mark.parametrize('subdivisions',[True,0,-1,2.5])
def test_bad_density_rejected(subdivisions):
    with pytest.raises(ValueError):densify([],subdivisions)


def test_adjacent_reference_can_start_from_legacy_exact_1x_grasp(tmp_path):
    import json
    from retarget.env.grasp.template_mppi_dense_runner import reference_from_parent
    from retarget.env.grasp.template_mppi_reference import POLICY,template_signature,load_reference
    from retarget.env.grasp.template_mppi_inputs import TemplateContacts
    t=TemplateContacts(('right_palm_link',),[[0,0,0]],[[1,0,0]],np.zeros(20),(('right_palm_link',),),{})
    primitive=size_specifications(26,'sphere')[0]
    case=tmp_path/'legacy';case.mkdir();reference=case/'original_reference.json'
    reference.write_text(json.dumps(dict(policy=POLICY,template='26_Sphere_4_Finger',template_signature=template_signature(t),
        initial_qpos=np.zeros(26).tolist(),object_in_wrist=np.eye(4).tolist(),wrist_in_object=np.eye(4).tolist(),
        canonical_primitive=primitive,seconds=1.,selected=0,candidate_costs=[0.],matcher={},selection_objective='root',
        object_points=[[.02,0,0]],object_outward_normals=[[1,0,0]])))
    (case/'initialization.json').write_text(json.dumps(dict(reference=str(reference))))
    (case/'template_contacts.json').write_text(json.dumps(t.to_dict()))
    (case/'projection_force_closure.json').write_text(json.dumps(dict(projected_points=[[.02,0,0]],object_normals=[[1,0,0]])))
    q=np.zeros(26);q[6]=.2;np.savez(case/'grasp.npz',qpos=q)
    path,_,_=reference_from_parent(tmp_path/'next',dict(name='old26',trajectory=str(case/'trajectory.npz'),primitive=primitive))
    actual,_=load_reference(path,t,'26_Sphere_4_Finger')
    np.testing.assert_array_equal(actual,q)
    saved=json.loads(path.read_text());assert saved['root_match_initial_qpos'][6]==0
