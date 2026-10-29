"""Settling gate, release dynamics and drop classification regressions."""
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from retarget.grasp_simulation import stable_velocity,stable_pose_window,retention_metrics,hand_surface_motion_bound


def test_stability_checks_every_joint_and_free_object():
    v=np.zeros(32)
    assert stable_velocity(v,np.arange(26),26)
    for col,value in ((0,.02),(5,.1),(25,.1),(26,.02),(31,.1)):
        altered=v.copy();altered[col]=value
        assert not stable_velocity(altered,np.arange(26),26)
    v[31]=np.nan
    assert not stable_velocity(v,np.arange(26),26)


def test_drop_is_relative_to_the_hand_and_records_lateral_escape():
    q=np.r_[np.zeros(26),[0.,0.,.1,1.,0.,0.,0.]]
    moved=q.copy();moved[:3]=[.2,.1,-.3];moved[26:29]+=moved[:3]
    assert not retention_metrics(moved,q,26)['dropped']
    falling=moved.copy();falling[28]-=.04
    result=retention_metrics(falling,q,26)
    assert result['dropped'] and result['escaped']
    np.testing.assert_allclose(result['relative_fall_mm'],40.)
    lateral=q.copy();lateral[26]+=.04
    result=retention_metrics(lateral,q,26)
    assert result['escaped'] and not result['dropped']
    # Rotate the hand in place while preserving the object's hand-local pose.
    rotated=q.copy();rotated[3:6]=[.2,-.1,.4]
    rotated[26:29]=Rotation.from_euler('XYZ',rotated[3:6]).apply(q[26:29])
    assert retention_metrics(rotated,q,26)['relative_displacement_mm']<1e-10


def test_cli_enables_visualization_and_physics_by_default(monkeypatch):
    from run_grasp import main
    import retarget.grasp
    import retarget.grasp_compact
    import retarget.grasp_results
    monkeypatch.setenv('DISPLAY',':99')
    monkeypatch.setattr(retarget.grasp,'run',lambda args:args)
    monkeypatch.setattr(retarget.grasp_compact,'run_compact',lambda args:args)
    monkeypatch.setattr(retarget.grasp_results,'register_results',lambda *args, **kwargs: None)
    args=main(['--primitives','--legacy-serial'])
    assert args.viewer and args.render and args.drop_test
    assert args.settle_mode=='fixed-object' and args.drop_seconds==2.
    assert args.friction==1. and args.impratio==10.
    assert args.physics_overrides=={}
    assert args.hold_controller=='mpc' and args.planning_gravity==9.81
    assert args.closing_depth_mm is None and args.closing_seconds==.3
    assert args.physics_dt==.00025 and args.load_conditioning
    assert args.physics_graph
    gpu=main(['--primitives'])
    assert gpu.gpu_nodes == 8 and not gpu.legacy_serial and not gpu.drop_test
    args=main(['--primitives','--no-viewer','--no-render','--no-drop-test'])
    assert not args.viewer and not args.render and not args.drop_test
    args=main(['--primitives','--no-load-conditioning'])
    assert args.impratio==1. and not args.load_conditioning


def test_hand_surface_bound_contains_actual_point_motion():
    radius=np.array([.03,.05])
    p=np.zeros((21,2,3));p[:,0,0]=np.linspace(0.,.00002,21)
    angle=np.sin(np.arange(21)*1.7)*.0008
    matrices=Rotation.from_rotvec(np.c_[angle,np.zeros((21,2))]).as_matrix()
    matrices=np.tile(matrices[:,None],(1,2,1,1))
    bound=hand_surface_motion_bound(p,matrices,radius)
    rng=np.random.default_rng(3);points=rng.normal(size=(2,40,3))
    points*=radius[:,None,None]/np.linalg.norm(points,axis=-1,keepdims=True)
    actual=p[:,:,None]+np.einsum('tbij,bpj->tbpi',matrices,points)
    diameter=np.linalg.norm(actual[:,None]-actual[None,:],axis=-1).max()
    assert diameter<=bound+1e-10 and bound<.0002
    matrices[10]=Rotation.from_rotvec([.02,0.,0.]).as_matrix()
    assert hand_surface_motion_bound(p,matrices,radius)>.0002


def test_position_plateau_rejects_slow_drift_and_accepts_small_contact_chatter():
    q=np.tile(np.r_[np.zeros(26),[0.,0.,0.,1.,0.,0.,0.]],(21,1))
    q[:,10]=.0002*np.sin(np.arange(21)*1.7)
    assert stable_pose_window(q,np.arange(26),26)
    drifting=q.copy();drifting[:,28]=np.linspace(0.,.004,21)
    assert not stable_pose_window(drifting,np.arange(26),26)
    translating=q.copy();translating[:,0]=np.linspace(0.,.0003,21)
    assert not stable_pose_window(translating,np.arange(26),26)
    rotating=q.copy();rotating[:,29:]=Rotation.from_rotvec(
        np.c_[np.zeros((21,2)),np.linspace(0.,.002,21)]).as_quat(scalar_first=True)
    assert not stable_pose_window(rotating,np.arange(26),26)


@pytest.fixture
def unsupported_case(tmp_path):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA physics required')
    from retarget.grasp import build_scene,StaticHand
    from retarget.grasp_primitives import primitive_mesh
    from retarget.grasp_initialization import from_dexonomy_result
    from retarget.grasp_runtime import prepare_experiment
    from retarget.dexonomy import ROOT
    model,surface,_,_=build_scene(primitive_mesh(tmp_path/'sphere.obj','sphere',.02,.04))
    body=np.array([model.body('right_finger1_link4').id])
    hand=StaticHand(model,surface,body,np.array([[0.,0.,.02]]),np.array([[1.,0.,0.]]))
    q=np.r_[[.5,0.,0.,0.,0.,0.],np.full(20,.3)]
    guess=from_dexonomy_result(dict(obj_pose=np.array([0.,0.,0.,1.,0.,0.,0.]),
        grasp_pose=np.r_[q[:3],1.,0.,0.,0.],grasp_qpos=q[6:],
        obj_cp_w=np.zeros((1,3)),obj_cn_w=np.array([[1.,0.,0.]])))
    args=SimpleNamespace(solver_config=ROOT/'data/example/forceaware.yaml',object_mass=.2,friction=.6,
                         init_device='cuda:0',settle_seconds=.25,settle_timeout=.4,
                         drop_seconds=.3,drop_distance_mm=30.,settle_mode='fixed-object')
    case=tmp_path/'case';case.mkdir()
    experiment=prepare_experiment(case/'input_00',hand,guess,args)
    import comfree_warp
    _,cpu=comfree_warp.load_model(str(experiment/'sequence/scene/wuji_right/primitive_sdf.xml'),device='cuda:0')
    state=cpu.qpos0.copy();state[:26]=q
    np.savez(case/'grasp.npz',qpos=state,qvel=np.zeros(cpu.nv),hold_action=q)
    (case/'result.json').write_text(json.dumps(dict(selected=dict(start=0))))
    mujoco.mj_saveLastXML(str(case/'scene.xml'),cpu)
    return case,args


@pytest.mark.parametrize('gravity_compensated', [False, True])
def test_cuda_fixture_releases_object_and_detects_unsupported_freefall(unsupported_case, gravity_compensated):
    from retarget.grasp_simulation import run_drop_test
    case,args=unsupported_case
    if gravity_compensated:
        source=json.loads((case/'result.json').read_text())
        source['selected']['solver_diagnostics']=dict(explicit_prediction=dict(hand_gravity_compensated_at_candidate=True))
        (case/'result.json').write_text(json.dumps(source))
        args.settle_timeout=3.
    report,arrays=run_drop_test(case,args)
    assert report['preparation_gravity_world']==[0.,0.,-9.81 if gravity_compensated else 0.]
    assert report['settled'] and report['gravity_test_ran']
    assert report['dropped'] and not report['retained']
    assert report['settling_object_fixture'] and not report['gravity_test_object_fixture']
    zero=arrays['phase']==0
    np.testing.assert_allclose(arrays['qpos'][zero,26:29],0.,atol=1e-8)
    np.testing.assert_allclose(arrays['qvel'][zero,26:32],0.,atol=1e-8)
    # Independent ballistic oracle: gravity is active and the former fixture
    # cannot keep supporting the free object in the test phase.
    duration=report['gravity_seconds']
    np.testing.assert_allclose(arrays['qpos'][-1,28],-.5*9.81*duration**2,atol=.004)
    assert arrays['qvel'][-1,28]<-2.8
    assert report['first_drop_seconds']<.12
    np.testing.assert_array_equal(arrays['hold_action'],arrays['qpos'][0,:26])
    assert 'drop_test' in json.loads((case/'result.json').read_text())
    np.testing.assert_allclose(arrays['object_contact_force_world'],0.,atol=1e-8)
    np.testing.assert_allclose(arrays['normal_force_by_finger'],0.,atol=1e-8)


def test_cuda_physics_graph_matches_uncaptured_steps_and_releases_fixture(unsupported_case):
    from retarget.grasp_simulation import run_drop_test
    case,args=unsupported_case
    args.physics_graph=False
    baseline,plain=run_drop_test(case,args)
    args.physics_graph=True
    actual,captured=run_drop_test(case,args)
    assert baseline['physics_graph_count']==0 and actual['physics_graph_count']==2
    assert actual['dropped']==baseline['dropped'] and actual['settled']==baseline['settled']
    for key in ('qpos','qvel','action','normal_force_by_finger','object_contact_force_world'):
        np.testing.assert_array_equal(captured[key],plain[key])
    assert set(actual['phase_timings'])=={'preload','settling','gravity'}
    np.testing.assert_allclose(actual['phase_timings']['gravity']['simulated_seconds'],args.drop_seconds,atol=1e-8)
    assert all(value['wall_seconds']>=0 for value in actual['phase_timings'].values())


def test_cuda_free_settling_timeout_never_claims_a_drop_test_pass(unsupported_case):
    from retarget.grasp_simulation import run_drop_test
    case,args=unsupported_case;args.settle_mode='free'
    with np.load(case/'grasp.npz') as saved:
        values={k:saved[k] for k in saved.files}
    values['qvel'][28]=.02
    np.savez(case/'grasp.npz',**values)
    report,arrays=run_drop_test(case,args)
    assert report['status']=='not_settled' and not report['gravity_test_ran']
    assert report['retained'] is None and report['dropped'] is None
    assert np.all(arrays['phase']==0)


def test_cuda_preload_is_ramped_then_fixture_releases_without_support(unsupported_case):
    from retarget.grasp_simulation import run_drop_test
    case,args=unsupported_case
    with np.load(case/'grasp.npz') as saved:
        original={k:saved[k].copy() for k in saved.files}
    q=original['qpos'][:26];command=q.copy();command[6:]+=.003
    original['initial_qpos']=original['qpos'].copy()
    np.savez(case/'grasp.npz',**original)
    np.savez(case/'input_00/load_input.npz',command=command,gravity_command=q)
    args.load_conditioning=True;args.formation_seconds=.06;args.preload_ramp_seconds=.03
    report,arrays=run_drop_test(case,args)
    assert report['load_conditioned'] and report['preparation_gravity_world']==[0.,0.,-9.81]
    np.testing.assert_allclose(arrays['action'][0],q)
    for i in range(1,4):
        np.testing.assert_allclose(arrays['action'][i],q+(command-q)*(i/3),atol=1e-12)
    assert report['gravity_test_ran'] and report['dropped'] and not report['verified_grasp']
    assert not report['gravity_test_object_fixture']
    np.testing.assert_allclose(arrays['object_contact_force_world'],0.,atol=1e-8)


def test_cuda_mpc_uses_original_state_and_partial_release_window(unsupported_case):
    from retarget.grasp_simulation import run_drop_test
    case,args=unsupported_case
    with np.load(case/'grasp.npz') as saved:
        values={k:saved[k].copy() for k in saved.files}
    initial=values['qpos'].copy()
    values['initial_qpos']=initial.copy()
    values['qpos'][0]+=.1  # A deliberately different unexecuted prediction.
    np.savez(case/'grasp.npz',**values)
    args.hold_controller='mpc';args.formation_seconds=.03;args.drop_seconds=.04
    args.settle_seconds=.5;args.settle_timeout=3.
    args.physics_dt=.002;args.iterations=1;args.multistarts=1;args.seed=0
    args.hand_solver='adamw';args.contact_radius_mm=5.;args.contact_motion_tolerance_mm=.1
    report,arrays=run_drop_test(case,args)
    assert report['settled'] and report['gravity_test_ran']
    np.testing.assert_array_equal(arrays['qpos'][0],initial)
    assert report['controller_windows']==3
    windows=json.loads((case/'hold_windows.json').read_text())
    assert [w['execution_steps'] for w in windows]==[3,3,1]
    assert [w['phase'] for w in windows]==[-1,1,1]
    assert set(arrays['phase'])=={-1,0,1}
    # MPC cannot fabricate a contact/support force for the distant object.
    np.testing.assert_allclose(arrays['object_contact_force_world'],0.,atol=1e-8)
    np.testing.assert_allclose(arrays['qpos'][-1,28],-.5*9.81*.04**2,atol=.001)
    assert report['control_policy']=='shared run.py rolling MPC'


def test_reference_physics_reads_resolved_object_properties(tmp_path):
    from retarget.grasp_physics import reference_physics,resolve_physics_args
    xml=tmp_path/'reference.xml'
    xml.write_text('''<mujoco><option impratio="7"/><worldbody><body>
      <freejoint name="obj_joint"/><geom type="box" size=".02 .03 .04"
      friction=".8 .003 .0002" solref=".03 1.2" solimp=".7 .9 .002 .4 3"/>
      </body></worldbody></mujoco>''')
    args=SimpleNamespace(physics_reference=xml,friction=None,impratio=None,contact_solref=None,contact_solimp=None)
    resolve_physics_args(args)
    assert args.impratio==7 and args.friction==.8
    np.testing.assert_allclose(args.contact_solref,[.03,1.2])
    np.testing.assert_allclose(args.contact_solimp,[.7,.9,.002,.4,3])
    assert args.physics_overrides['impratio']==7
    assert reference_physics(str(xml))['condim']==3


def test_cuda_replay_parameter_overrides_reach_compiled_physics_without_editing_input(unsupported_case):
    from retarget.grasp_simulation import run_drop_test
    case,args=unsupported_case
    scene=case/'input_00/sequence/scene/wuji_right/primitive_sdf.xml'
    original=scene.read_bytes()
    args.physics_overrides=dict(impratio=10.,friction=[1.,.005,.0001],solref=[.05,1.],
                               solimp=[.8,.9,.005,.5,2.])
    report,_=run_drop_test(case,args)
    p=report['physics']
    assert p['impratio']==10.
    np.testing.assert_allclose(p['gpu_impratio_invsqrt'],[1/np.sqrt(10)],rtol=1e-6)
    np.testing.assert_allclose(p['compiled_contact_friction'],[[1.,1.,.005,.0001,.0001]],rtol=1e-6)
    np.testing.assert_allclose(p['compiled_contact_solref'],[[.05,1.]],rtol=1e-6)
    np.testing.assert_allclose(p['compiled_contact_solimp'],[[.8,.9,.005,.5,2.]],rtol=1e-6)
    assert report['dropped'] and scene.read_bytes()==original
    assert (case/'drop_test_scene.xml').is_file()
