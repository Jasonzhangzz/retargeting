"""Numerical recovery, evidence separation and frozen-preview contracts."""
import copy,threading,time
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from moxian_web.qnbot_surface import QnbotSurface,FINGERS
from moxian_web.contact import default_mapping,PATCH_IDS
from moxian_web.qnbot_lsq_model import Coordinates,Measurements,Residuals,contact_evidence,observed
from moxian_web.qnbot_lsq_solver import Solver,options,schur_subspace,pose_sweep,total_cost
from moxian_web.qnbot_worker import StreamingModel
from moxian_web.checkpoints import preview_revision_compatible,CheckpointCatalog
from moxian_web.qnbot_least_squares import FIT_REVISION


def synthetic(n=8,device='cpu'):
    torch.set_num_threads(1);m=QnbotSurface('right',device);m.bind_mapping(dict(assignments=default_mapping('right'),confirmed=[]))
    c=Coordinates(m);x=torch.zeros(c.size,device=m.device)
    generator=torch.Generator(device=m.device).manual_seed(123)
    q=m.q0[None].expand(n,-1).clone()+torch.rand((n,25),device=m.device,generator=generator)*.35
    return m,c,x,q


def rows_for(m,q,x=None,c=None,partition='train'):
    with torch.no_grad():p,r=m.sensor_prediction(q,geometry=c.geometry(x) if x is not None else None)
    rows=[]
    for i in range(len(q)):
        rows.append(dict(hand_side='right',timestamp_ms=i*30.,partition=partition,stage='motion_spread_together',window_id='window',mapping=m.mapping,
            skeleton=dict(joint_names=m.joint_names,joint_positions_rad=q[i].tolist()),fingers={f:dict(position=p[i,k].cpu().tolist(),quaternion_xyzw=Rotation.from_matrix(r[i,k].cpu().numpy()).as_quat().tolist()) for k,f in enumerate(FINGERS)}))
    return rows


def tactile_row(assignments):
    a=PATCH_IDS.index('thumb_distal');b=PATCH_IDS.index('index_distal')
    cells=[False]*460;v=[0.]*460
    for p in (a,b):
        i=assignments.index(p);cells[i]=True;v[i]=.8
    return dict(timestamp_ms=100.,tactile_timestamp_ms=100.,stage='contact_pair_index',contact_confirmed=False,contact_constraint_source='trajectory_assumed_contact',
                contact=dict(ready=True,cell_contact=cells,normalized=v,patches=[dict(id=PATCH_IDS[p],contact=True) for p in (a,b)]))


def test_functional_geometry_matches_existing_forward_and_sparse_support():
    m,c,x,q=synthetic();x[:15]=torch.linspace(-.4,.4,15)
    geometry=c.geometry(x);support=torch.arange(0,len(m.vertices0),127)
    expected=m(q,geometry=geometry)[0][:,support]
    torch.testing.assert_close(m(q,vertex_ids=support,geometry=geometry)[0],expected,atol=2e-7,rtol=1e-5)
    c.install(x);torch.testing.assert_close(m(q)[0][:,support],expected,atol=2e-7,rtol=1e-5)
    for k in ('length','width','depth'):
        for f in range(5):
            values=m.geometry[k][c.movable&(c.owner==f)]
            torch.testing.assert_close(values,values[0].expand_as(values))
    assert torch.count_nonzero(m.geometry['residual'])==0


def test_only_fresh_two_sided_measured_contact_can_create_positive_equality():
    assignments=default_mapping('right');row=tactile_row(assignments)
    _,edges,_=contact_evidence(row,assignments);assert edges==[(PATCH_IDS.index('thumb_distal'),PATCH_IDS.index('index_distal'),1)]
    assert row['contact_confirmed'] is False
    stale=copy.deepcopy(row);stale['tactile_timestamp_ms']=0;assert contact_evidence(stale,assignments)[1]==[]
    one=copy.deepcopy(row);one['contact']['patches'][1]['contact']=False
    one['contact']['cell_contact'][assignments.index(PATCH_IDS.index('index_distal'))]=False
    assert contact_evidence(one,assignments)[1]==[]
    assumed=copy.deepcopy(row);assumed['contact']={};assumed['contact_confirmed']=True
    assert contact_evidence(assumed,assignments)[1]==[]
    palm=copy.deepcopy(row);palm['stage']='contact_palm_index'
    assert contact_evidence(palm,assignments)[1]==[]


def test_missing_orientation_is_not_identity_supervision():
    rows=[dict(fingers={'thumb':dict(position=[.01,.02,.03])})]
    _,_,valid,rot=observed(rows,'cpu')
    assert valid[0,0]==1 and rot.sum()==0 and valid.sum()==1


def test_schur_eliminates_pose_confounded_coordinate_without_prior_confidence():
    jg=torch.tensor([[[1.,0.,0.],[0.,2.,0.],[0.,0.,0.]]])
    jq=torch.tensor([[[1.],[0.],[0.]]])
    basis,report=schur_subspace(jg,jq)
    assert report['rank']==1
    assert basis.shape==(3,1);torch.testing.assert_close(basis.abs(),torch.tensor([[0.],[1.],[0.]]))
    assert not report['heldout_used'] and not report['priors_used_for_rank']


@pytest.mark.parametrize('block',[list(range(5)),list(range(15,30)),list(range(30,45))])
def test_synthetic_dimensions_and_mounts_recover_with_known_pose(block):
    m,c,x,q=synthetic(18);truth=x.clone();truth[block]=torch.linspace(-.2,.2,len(block))
    data=Measurements(m,rows_for(m,q,truth,c),noise=dict(position_scale_m=.001,rotation_scale_rad=.05,source='known_synthetic_noise'))
    o=Residuals(c,data);settings=options(dict(max_seconds=60,shared_batch_size=18,chunk_size=18))
    solver=Solver(o,threading.Event(),lambda _:None,settings);ids=torch.tensor(block)
    before=total_cost(o,x,q,18)
    for _ in range(4):x,_=solver.shared_step(x,q,ids,torch.eye(len(ids)),list(range(18)),.001)
    after=total_cost(o,x,q,18)
    assert after<before*.08
    assert float((x[ids]-truth[ids]).abs().max())<.04


def test_latent_pose_sweep_reduces_emf_error_without_contact_labels():
    m,c,x,q=synthetic(6);rows=rows_for(m,q)
    for row in rows:row['skeleton']['joint_positions_rad'][7]-=.08
    d=Measurements(m,rows);o=Residuals(c,d);before=total_cost(o,x,d.q0,6,contact=False)
    solved,report=pose_sweep(o,x,d.q0,threading.Event(),options(dict(chunk_size=6)),contact=False,iterations=4)
    assert total_cost(o,x,solved,6,contact=False)<before*.1
    assert report['iterations']>0
    assert bool((solved>=d.low).all()&(solved<=d.high).all())


def test_validation_labels_do_not_change_sensor_only_pose_or_statistics():
    m,c,x,q=synthetic(5);train=Measurements(m,rows_for(m,q))
    a=rows_for(m,q,partition='validation');b=copy.deepcopy(a)
    for row in b:row.update(stage='contact_pair_index',contact_confirmed=True,expected_contact_regions={'secret':'unused'})
    da=Measurements(m,a,training=False,noise=train.noise);db=Measurements(m,b,training=False,noise=train.noise)
    ids=torch.arange(5)
    torch.testing.assert_close(Residuals(c,da).frame(x,q,ids,contact=False),Residuals(c,db).frame(x,q,ids,contact=False))
    assert da.noise==db.noise==train.noise
    with pytest.raises(ValueError,match='训练分区'):Measurements(m,a,training=True)


def test_time_regularizer_resets_at_window_partition_and_gaps():
    m,c,x,q=synthetic(6);rows=rows_for(m,q)
    rows[2]['window_id']='other';rows[4]['timestamp_ms']=1000;rows[5]['timestamp_ms']=1030
    d=Measurements(m,rows)
    assert d.prev.tolist()==[0,0,2,3,4,4]
    assert d.previous2[5]==4


def test_material_offsets_geometry_and_preview_roundtrip_are_frozen():
    m,c,x,q=synthetic(3);baseline=m.artifact();original_binding_hash=m.binding_hash;x[:15]=.1;x[71]=.7;x[69]=.6
    c.install(x);candidate=m.artifact();candidate.update(delta_policy=None,report=dict(method='least_squares',fit_revision=FIT_REVISION,frozen_forward=True,online_optimization=False))
    assert preview_revision_compatible(candidate)
    runtime=StreamingModel(m);runtime.load(baseline);base_binding=m.artifact()['material_binding']
    runtime.load(candidate);shown=m.artifact();assert shown['material_binding']==candidate['material_binding']
    versions=[]
    for row in rows_for(m,q):
        runtime.infer(row);versions.append(m.version())
    assert len(set(versions))==1 and runtime.policy is None
    runtime.load(baseline);assert m.artifact()['material_binding']==base_binding
    assert m.binding_hash==original_binding_hash
    for k,v in m.geometry.items():torch.testing.assert_close(v,torch.tensor(baseline['geometry'][k]))
    bad=copy.deepcopy(candidate);bad['material_binding']['normal_offset_m'][0]=float('nan')
    original=m.artifact()
    with pytest.raises(ValueError):runtime.load(bad)
    assert m.artifact()==original


def test_low_rank_surface_modes_are_smooth_and_orthogonal_to_rigid_motion():
    m,c,x,q=synthetic(2);c.build_surface_basis(x)
    basis=c.surface_basis.reshape(-1,20)
    assert c.surface_basis_report['rank']<=20 and torch.isfinite(basis).all()
    assert float(c.surface_basis.sum(0).abs().max())<.005
    assert float((c.surface_basis[m.edges[:,0]]-c.surface_basis[m.edges[:,1]]).square().mean())<.02
    assert torch.count_nonzero(c.geometry(x)['residual'])==0


def test_preview_catalog_accepts_single_complete_native_branch(tmp_path):
    import hashlib,json
    m,c,x,q=synthetic(1);candidate=m.artifact();candidate.update(serial='unit',mapping=m.mapping,delta_policy=None,
        report=dict(fit_revision=FIT_REVISION,method='least_squares',frozen_forward=True,online_optimization=False,accepted=False))
    path=tmp_path/'hand_model'/hashlib.sha256(b'unit').hexdigest()[:16]/'qnbot/methods/least_squares/candidate.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(candidate))
    items=CheckpointCatalog(tmp_path).list('unit','right')
    assert len(items)==1 and items[0]['method']=='least_squares' and not items[0]['accepted']


def test_solver_options_and_cancel_are_explicit():
    assert options()['max_seconds']==600
    for invalid in ({'adam_steps':480},{'rank_rtol':float('nan')},{'stages':['unknown']},{'chunk_size':0}):
        with pytest.raises(ValueError):options(invalid)
    m,c,x,q=synthetic(2);d=Measurements(m,rows_for(m,q));cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError):Solver(Residuals(c,d),cancel,lambda _:None,options()).solve()


def test_complete_cpu_pipeline_exports_assets_without_publishing(tmp_path):
    from moxian_web.mano_calibration import fit_capture
    m,c,x,q=synthetic(6);rows=rows_for(m,q[:3])+rows_for(m,q[3:],partition='validation')
    for i,row in enumerate(rows):row['timestamp_ms']=30.*i
    profile=dict(surface_backend='qnbot',serial='unit',side='right',device='cpu',artifact_home=str(tmp_path),run_output=str(tmp_path/'run'),calibration_fingerprint='f',skeleton_runtime_revision='r')
    candidate,report=fit_capture(None,profile,m.mapping,rows,lambda _:None,threading.Event(),method='least_squares',solver_options=dict(stages=[],pose_iterations=1,max_seconds=5))
    assert candidate['delta_policy'] is None and not report['online_optimization']
    assert not report['performance_is_acceptance_gate'] and not report['validation_complete'] and not report['accepted']
    expected=['candidate.json','report.json','validation.json','hand_shape.json','kinematics.json','emf_extrinsics.json','tactile_map.json','contact_geometry.json','hand_mesh.obj','collision_mesh.obj','skin_weights.npy','surface_basis.npz','offline_poses.npz']
    assert all((tmp_path/'run'/name).exists() for name in expected)
    assert not list(tmp_path.rglob('current.json'))
    assert len(report['runtime_validation']['frames'])==3
    assert 'training_q' not in candidate and candidate['geometry_parameterization']=='qnbot_lsq_coordinates_v1'


def test_dispatch_rejects_adam_controls_and_mano_backend():
    from moxian_web.mano_calibration import fit_capture
    with pytest.raises(ValueError,match='solver_options'):
        fit_capture(None,dict(surface_backend='qnbot'),{},[],lambda _:None,threading.Event(),method='least_squares',training_options={'optimizer':'adam'})
    with pytest.raises(ValueError,match='Qnbot'):
        fit_capture(None,dict(surface_backend='mano'),{},[],lambda _:None,threading.Event(),method='least_squares')


def test_bound_active_coordinate_does_not_block_other_identifiable_coordinates():
    from moxian_web.qnbot_lsq_solver import bounded_quadratic
    h=torch.eye(2);g=torch.tensor([-3.,-2.]);basis=torch.eye(2);x=torch.zeros(2)
    step,report=bounded_quadratic(h,g,basis,x,torch.tensor([-1.,-1.]),torch.tensor([0.,2.]),0.)
    torch.testing.assert_close(step,torch.tensor([0.,2.]),atol=1e-6,rtol=0)
    assert report['success'] and not report['stationary']


def test_bounded_subspace_never_acquires_a_nullspace_component():
    from moxian_web.qnbot_lsq_solver import bounded_quadratic
    basis=torch.tensor([[1.],[1.]])/2**.5
    step,report=bounded_quadratic(torch.eye(1),torch.tensor([-3.]),basis,torch.zeros(2),-torch.ones(2),torch.tensor([.1,1.]),0.)
    direction=basis@step
    torch.testing.assert_close(direction,torch.tensor([.1,.1]),atol=1e-6,rtol=0)
    assert report['success']


def test_exported_joint_origins_and_body_frames_have_explicit_matching_names():
    from moxian_web.qnbot_least_squares import kinematic_asset
    m,c,x,q=synthetic(1);asset=kinematic_asset(m)
    assert len(asset['joint_names'])==len(asset['joint_origins_m'])==len(asset['joint_axes_local'])==25
    assert len(asset['body_names'])==len(asset['body_positions_m'])
    np.testing.assert_allclose(np.linalg.norm(asset['joint_axes_local'],axis=-1),1,atol=1e-6)
