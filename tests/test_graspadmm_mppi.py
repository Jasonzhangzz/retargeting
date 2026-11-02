"""MPPI pose/frame/collision and paired-denominator regression checks."""
from dataclasses import replace
import numpy as np
import pytest
import torch
import mujoco
import trimesh
import warp as wp

pytest.importorskip('mujoco_warp')
from retarget.env.grasp.graspadmm_mppi import (
    LocalPoseObjective,RefinementSettings,native_to_local,local_to_native,refinement_model)
from retarget.env.grasp.grasp_evaluation import evaluate_native_grasp,PROTOCOL
from scripts.refine_graspadmm_mppi import aggregate,size_index


def test_pose_rotation_roundtrip():
    for angle in ([.1,.2,-.3],[1.4,1.5,-2.],[-2.,-.7,1.8]):
        q=np.r_[[.03,-.07,.11],angle,np.linspace(-.1,.4,20)]
        recovered=native_to_local(local_to_native(q))
        np.testing.assert_allclose(local_to_native(recovered),local_to_native(q),atol=1e-12)
        np.testing.assert_array_equal(recovered[6:],q[6:])


def test_pair_accounting_includes_runtime_failures():
    rows=[dict(baseline_success=a,task_success=b,failure_stage=stage) for a,b,stage in
        [(True,True,None),(True,False,'grasp_filter'),(False,True,None),(False,False,'refinement_runtime')]]
    report=aggregate(rows)
    assert report['trials']==4
    assert report['baseline_success_rate']==report['refined_success_rate']==.5
    assert report['validated_fallback_success_rate']==.75
    assert report['transitions']==dict(retained=1,lost=1,recovered=1,neither=1)
    assert size_index(dict(case_id='t26_sphere_s04_seed09'))==4


def test_native_worker_receives_exact_candidate_and_groups(tmp_path):
    mesh=tmp_path/'sphere.obj';trimesh.creation.icosphere(radius=.03).export(mesh)
    grasp=local_to_native(np.zeros(26));groups=[['right_finger1_link4'],['right_palm_link']]
    class Worker:
        def evaluate(self,path,timeout):
            import json
            np.testing.assert_array_equal(np.load(path/'candidate.npz')['grasp_qpos'],grasp)
            assert json.loads((path/'candidate.json').read_text())['required_cbody']==groups
            assert (path/'object.obj').read_bytes()==mesh.read_bytes()
            return dict(protocol=PROTOCOL,success=True,evaluation_wall_seconds=.1)
    result=evaluate_native_grasp(grasp,mesh,tmp_path/'test',method='graspadmm_mppi',
        required_cbody=groups,worker=Worker())
    assert result['success']


@pytest.fixture(scope='module')
def objective(tmp_path_factory):
    if not torch.cuda.is_available():pytest.skip('Native MuJoCo-Warp requires CUDA')
    mesh=tmp_path_factory.mktemp('mppi_geometry')/'sphere.obj'
    trimesh.creation.icosphere(subdivisions=2,radius=.03).export(mesh)
    return LocalPoseObjective(mesh,RefinementSettings(worlds=8,population=32,iterations=2,stages=1,points_per_body=16))


def test_native_geometry_preserves_object_frame_and_finger_pads(objective):
    model=objective.cpu
    assert (model.nq,model.nu)==(26,26)
    np.testing.assert_array_equal(model.body('grasp_object').pos,[0,0,0])
    pads=[i for i in range(model.nbody) if 'sensor_frame' in model.body(i).name]
    assert len(pads)==5
    for body in pads:
        assert any(model.geom_contype[g]!=0 for g in np.flatnonzero(model.geom_bodyid==body))
    assert 'right_palm_link' in objective.slices


def test_gpu_collision_and_fk_agree_with_cpu_without_time_integration(objective):
    # A sphere at the wrist gives a robust nonzero collision, and translated
    # configurations test that contact buffers are reset between populations.
    o=objective;q=np.tile(o.cpu.qpos0,(3,1));q[:,0]=[0,.15,-.01];q[:,6:]=.1
    o.bind(local_to_native(q[0]),[['right_palm_link']])
    # Compare independently of the local search box.
    o.qlo=o.low;o.qhi=o.high
    with torch.no_grad():o.evaluate(torch.tensor(q,device=o.device,dtype=torch.float32)[:,None])
    depth=wp.to_torch(o.depth)[:3].cpu().numpy();positions=o.physics.xpos[:3].cpu().numpy()
    native=[]
    for k,pose in enumerate(q):
        data=mujoco.MjData(o.cpu);data.qpos[:]=pose;mujoco.mj_forward(o.cpu,data)
        values=[0.,0.]
        obj=o.cpu.body('grasp_object').id
        for c in data.contact:
            kind=0 if obj in o.cpu.geom_bodyid[c.geom] else 1
            values[kind]=max(values[kind],o.settings.collision_clearance_m-c.dist)
        native.append(values)
        np.testing.assert_allclose(positions[k],data.xpos,atol=2e-7)
    assert np.max(native)>0
    np.testing.assert_allclose(depth,native,atol=2e-5,rtol=.01)
    assert not bool(wp.to_torch(o.physics.data.time).any())


def test_refinement_is_seeded_local_and_cost_monotonic(objective):
    o=objective;q=o.cpu.qpos0.copy();q[0]=.04;q[6:]=.15
    grasp=local_to_native(q);groups=[['right_palm_link']]
    with torch.no_grad():
        a,report=o.refine(grasp,groups,73)
        b,second=o.refine(grasp,groups,73)
    np.testing.assert_array_equal(a,b)
    assert report['final_cost']<=report['initial_cost']
    assert report['sampled']==32*2
    assert report['acceptance_during_optimization'] is False
    assert np.max(abs(a[:3]-q[:3]))<=o.settings.max_pos_delta+1e-7
    assert np.max(abs(a[6:]-q[6:]))<=o.settings.max_joint_delta+1e-7
    assert not bool(wp.to_torch(o.physics.data.time).any())
