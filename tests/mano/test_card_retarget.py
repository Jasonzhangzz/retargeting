"""Independent numerical contracts for CARD energies, manifold CEM and data flow."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
import trimesh
from scipy.spatial.transform import Rotation
from retarget.baseline.card.math import ALGORITHM,CardSettings,exp_so3,log_so3,rotation_mean,cem,axis_angle_near,elite_weights,project
from retarget.baseline.card.geometry import PseudoObject,TrackedMesh,source_wrist,segment_gap
from retarget.baseline.card.retarget import TactileRetarget
from moxian_web.wuji_surface import mirror_matrix,CHAINS


def test_production_constants_and_squared_translation_weight():
    s=CardSettings();assert (s.population,s.iterations,int(s.population*s.elite_fraction))==(1024,16,51)
    assert s.wt**2==pytest.approx(125) and s.wc**2==1600 and s.radius_mm==10
    assert (s.wfix,s.wn,s.wb,s.wself,s.wobj_mesh,s.hard_penetration_mm)==(2.5,60,20,1.5,14,3)


def test_so3_pi_crossing_and_weighted_mean():
    v=torch.tensor([[0.,0.,np.pi-1e-4],[0,0,-np.pi+1e-4]],dtype=torch.float64)
    r=exp_so3(v);torch.testing.assert_close(exp_so3(log_so3(r)),r,atol=1e-8,rtol=1e-8)
    m=rotation_mean(r,torch.tensor([.5,.5],dtype=torch.float64))
    assert torch.linalg.vector_norm(log_so3(m))==pytest.approx(np.pi,abs=1e-6)
    assert torch.linalg.vector_norm(log_so3(r[0].T@r[1]))==pytest.approx(.0002,abs=1e-6)


def test_cem_returns_global_best_budget_projection_and_determinism():
    s=CardSettings();initial=(torch.zeros(3),torch.eye(3),torch.zeros(20));previous=initial;seen=[]
    def evaluate(t,r,j):
        assert torch.all(t.norm(dim=-1)<=.020001) and torch.all(j.abs()<=.250001)
        cost=((t-t.new_tensor([.01,0,0]))*1000).square().sum(-1)+j.square().sum(-1);seen.extend(cost.tolist());return cost
    best,info=cem(initial,previous,torch.full((20,),-1.),torch.ones(20),evaluate,s,torch.Generator().manual_seed(7))
    assert info['sampled_candidates']==16384 and info['iterations']==16 and info['elite_count']==51
    assert info['best_cost']==pytest.approx(min(seen))
    assert info['final_sigma_ratio']==.2
    other,again=cem(initial,previous,torch.full((20,),-1.),torch.ones(20),evaluate,s,torch.Generator().manual_seed(7))
    for a,b in zip(best,other):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert again==info


def test_cem_no_finite_elites_never_shortens_budget_or_fakes_success():
    s=CardSettings(population=32,iterations=16);calls=[]
    def invalid(t,r,j):calls.append(len(t));return torch.full((len(t),),torch.inf)
    _,info=cem((torch.zeros(3),torch.eye(3),torch.zeros(20)),None,-torch.ones(20),torch.ones(20),invalid,s,torch.Generator().manual_seed(7))
    assert not info['feasible'] and info['best_cost'] is None and len(calls)==18


def test_cem_empty_elites_keep_mean_without_violation_recovery():
    settings=CardSettings(population=32,iterations=3)
    initial=(torch.zeros(3),torch.eye(3),torch.zeros(20));seen=[]
    incumbent=(initial[0],initial[1],torch.ones(20)*.1)
    def objective(t,r,j):
        seen.append(j.clone())
        return torch.tensor([torch.inf,0.]) if len(seen)==1 else torch.full((len(j),),torch.inf)
    best,info=cem(initial,None,-torch.ones(20),torch.ones(20),objective,settings,torch.Generator().manual_seed(7),incumbents=[incumbent])
    torch.testing.assert_close(best[2],incumbent[2])
    generator=torch.Generator().manual_seed(7)
    for i,values in enumerate(seen[1:-1]):
        noise=torch.randn((32,26),generator=generator)
        torch.testing.assert_close(values,noise[:,6:]*.05*.2**(i/3))
    assert info['empty_elite_restarts']==0 and info['empty_elite_iterations']==3
    assert info['feasibility_recovery_iterations']==0


def test_cem_warm_start_and_bounds_use_previous_solution():
    settings=CardSettings(population=32,iterations=3)
    initial=(torch.tensor([.5,-.5,.1]),exp_so3(torch.tensor([0.,0.,1.5])),torch.full((20,),.8))
    previous=(torch.tensor([-.5,.5,-.1]),torch.eye(3),torch.full((20,),-.8));calls=[]
    def objective(t,r,j):
        calls.append((t.clone(),r.clone(),j.clone()))
        assert torch.all((t-previous[0]).norm(dim=-1)<=.020001)
        assert torch.all((j-previous[2]).abs()<=.250001)
        return j.square().sum(-1)
    _,report=cem(initial,previous,-torch.ones(20),torch.ones(20),objective,settings,torch.Generator().manual_seed(7))
    expected=project((initial[0]+previous[0])*.5,exp_so3(torch.tensor([0.,0.,.75])),(initial[2]+previous[2])*.5,previous,-torch.ones(20),torch.ones(20),settings)
    for got,want in zip(calls[0],expected):torch.testing.assert_close(got[0],want)
    assert report['search_center']=='half_previous_half_initializer' and report['previous_used_as_search_center']
    assert report['previous_evaluated_as_incumbent']
    assert report['evaluations']==settings.population*settings.iterations+3


def test_first_frame_has_no_initializer_neighbourhood(monkeypatch):
    settings=CardSettings(population=4,iterations=1)
    initial=(torch.zeros(3),torch.eye(3),torch.zeros(20))
    monkeypatch.setattr(torch,'randn',lambda shape,**kwargs:torch.full(shape,10.,device=kwargs['device'],dtype=kwargs['dtype']))
    seen=[]
    def objective(t,r,j):seen.append((t.clone(),j.clone()));return -j.sum(-1)
    best,report=cem(initial,None,-torch.ones(20),torch.ones(20),objective,settings,torch.Generator())
    assert seen[1][0].norm(dim=-1).min()>.02 and seen[1][1].min()>.25
    assert best[2].min()==pytest.approx(.5) and report['feasible']


def test_native_mesh_verification_is_diagnostic_only():
    initial=(torch.zeros(3),torch.eye(3),torch.zeros(20));settings=CardSettings(population=32,iterations=4)
    def evaluate(t,r,j):return (j[:,0]-.1).square()
    def verify(t,r,j):return j[:,0]<=.03
    best,report=cem(initial,None,-torch.ones(20),torch.ones(20),evaluate,settings,torch.Generator().manual_seed(7),verify=verify)
    baseline,_=cem(initial,None,-torch.ones(20),torch.ones(20),evaluate,settings,torch.Generator().manual_seed(7))
    for a,b in zip(best,baseline):torch.testing.assert_close(a,b)
    assert best[2][0]>.03 and report['feasible'] and report['mesh_diagnostic_satisfied'] is False
    assert report['dense_verification_rejected']==0


def test_segment_distance_parallel_and_degenerate():
    a=torch.tensor([[0.,0,0],[0,0,0]]);b=torch.tensor([[1.,0,0],[0,0,0]])
    c=torch.tensor([[0.,2,0],[0,3,0]]);d=torch.tensor([[1.,2,0],[0,3,0]])
    torch.testing.assert_close(segment_gap(a,b,c,d),torch.tensor([2.,3.]))


@pytest.fixture(scope='module')
def source():
    from moxian_web.mano_runtime import ManoSurface
    from moxian_web.hand_model import DEFAULT_MANO
    from moxian_web.contact import default_mapping
    if not DEFAULT_MANO.exists():pytest.skip('Licensed MANO needed')
    m=ManoSurface(DEFAULT_MANO,'cpu');m.build_binding({'assignments':default_mapping('left'),'version':'card-test'})
    return m.topology()


def frame(source,stamp=1000.,active=()):
    j=np.array(source['rest_joints_m']);tips=np.array(source['tip_anchors_m'])
    return dict(vertices_m=source['rest_vertices_m'],joints_m=np.vstack((j[0],*[np.vstack((j[list(c)],tips[f])) for f,c in enumerate(CHAINS)])).tolist(),
                timestamp_ms=stamp,tactile_timestamp_ms=stamp,geometry_version=source['geometry_version'],binding_version=source['binding_version'],source_contact=dict(ready=True,timestamp_ms=stamp,cell_contact=np.isin(source['sid_patch_indices'],active).tolist(),normalized=[.5]*460))


@pytest.fixture(scope='module')
def solver(source,tmp_path_factory):
    if not torch.cuda.is_available():pytest.skip('CUDA needed')
    torch.set_num_threads(1)
    return TactileRetarget(source,Path('output/moxian/debug/card/test-cache'),settings=CardSettings(cem_backend='torch'))


@pytest.fixture
def feasible_start(monkeypatch):
    """An explicit solvable test problem, independent of the preserved initializer.

    The MANO rest fixture's real initializer violates the restored palm gate;
    keep that behaviour in a separate all-infeasible integration regression.
    """
    def install(solver):
        original=solver.initializer.__class__.__call__
        joints=solver.tensor([-.33863163,-1.07754838,1.09299409,.56184137,
            -.35446763,.06534481,-.87199229,-.09088475,-.31689179,-.08252352,.93919194,1.46624601,
            .00451851,-.36949059,1.42076147,.63012493,.02301729,.31297415,1.29250276,.27703118])
        def initialize(owner,*args,**kwargs):
            t,r,_=original(owner,*args,**kwargs)
            return t,r,joints.clone()
        monkeypatch.setattr(solver.initializer.__class__,'__call__',initialize)
    return install


@pytest.mark.parametrize('backend',['torch','robokit'])
def test_real_rest_initializer_reports_paper_infeasibility(source,backend):
    settings=CardSettings(cem_backend=backend)
    solver=TactileRetarget(source,Path('output/moxian/debug/card/test-cache'),settings=settings)
    result=solver.step(frame(source))
    assert not result['solution_feasible'] and result['solver']['best_cost'] is None
    assert result['collision']['palm_hard_rejected'] and result['collision']['min_palm_gap_mm']<6
    assert result['solver']['sampled_candidates']==16384 and result['solver']['evaluations']==16386
    assert solver.previous is None and not solver.previous_contacts
    assert np.isfinite(result['qpos']).all()


def test_left_reflects_right_does_not(source):
    assert np.linalg.det(mirror_matrix(source))==pytest.approx(-1)
    right={**source,'hand_side':'right'};np.testing.assert_array_equal(mirror_matrix(right),np.eye(3))


def test_robokit_fk_matches_native_joint_order_axes_and_actual_tips(solver):
    init=solver.initializer;rng=np.random.default_rng(31)
    for _ in range(3):
        q=solver.adapter.low+(solver.adapter.high-solver.adapter.low)*solver.tensor(rng.uniform(.1,.9,26));q[:6]=0
        state=init.robot.state(q=q[6:].cpu().numpy()[init.to_robokit][None])
        init.robot.forward_kinematics(state)
        got=state.T_world_link.numpy()[0,init.link_ids,:3]
        np.testing.assert_allclose(got,solver.adapter.keypoints(q).cpu().numpy(),atol=2e-7)
    assert init.robot.spec.has_link_capsules and init.artifact['self_collision_enabled']


def test_observed_ring_pinky_mesh_penetration_is_reported_without_extra_cost(source,solver):
    # Captured from the previously accepted live CEM result. Root placement
    # cannot affect self-penetration, so keep the fixture to 20 joint values.
    q=solver.tensor([.274118751,-.045868330,-.050382204,.227653921,
        .038960382,-.293445766,.823306739,.033878349,
        .091479026,-.267860025,.385163337,-.003184885,
        .048058301,.056948282,.471494287,-.120417595,
        -.224339113,-.358234167,-.038637199,.679082036])
    t=solver.tensor([0.,0.,0.]);r=torch.eye(3,device=solver.device)
    p,rr=solver.frames(t[None],r[None],q[None]);depth,valid=solver.mesh_collision.query(p,rr,dense=True)
    coarse,_=solver.mesh_collision.query(p,rr,dense=False)
    assert torch.all(depth>=coarse-1e-4) # final audit includes every search sample
    report=solver.mesh_collision.report(depth[0],valid[0])
    assert report['query_valid'] and report['pair_depths_mm']['ring_link2__pinky_link3']>2.8
    assert not solver.verify_mesh(t[None],r[None],q[None])[0]
    (v,j,_,_),_,anchors=solver.extractor.extract(frame(source),False);c=solver.context(v,j,anchors)
    c['self_pairs']=[(10,14)] # contact evidence never exempts native mesh depth
    cost,detail=solver.evaluate(t[None],r[None],q[None],c,True)
    assert detail['hard_mesh'][0] and detail['parts']['collision_mesh'][0]==0
    assert bool(torch.isfinite(cost[0]))==bool(~detail['hard_palm'][0])
    assert detail['max_penetration_mm'][0]>2.8 and detail['object_max_penetration_mm'][0]==0


def test_robokit_cold_seed_tracks_current_frame_without_retarget_or_cem_history(solver,monkeypatch):
    init=solver.initializer;q=solver.adapter.neutral.clone();p=solver.adapter.keypoints(q).cpu().numpy()
    monkeypatch.setattr(init,'use_history',False)
    a=init(p)
    # A different grasp between two identical inputs cannot contaminate the seed.
    bent=q.clone();bent[6::4]=.9;bent[8::4]=.8;bent[9::4]=.6
    other=solver.adapter.keypoints(bent).cpu().numpy();b=init(other);again=init(p)
    for left,right in zip(a,again):torch.testing.assert_close(left,right,atol=1e-7,rtol=1e-6)
    assert float((a[2]-b[2]).abs().max())>.3
    for target,(t,r,j) in ((p,a),(other,b)):
        predicted=solver.adapter.keypoints(torch.cat((j.new_zeros(6),j)))@r.T+t
        assert float((predicted-solver.tensor(target)).norm(dim=-1).max())<.003
    with pytest.raises(ValueError):init(np.full((21,3),np.nan))


def test_robokit_stream_warmstart_uses_own_joints_and_current_palm_and_resets(solver):
    init=solver.initializer;init.reset();q=solver.adapter.neutral.clone();p=solver.adapter.keypoints(q).cpu().numpy()
    a=init(p);assert not init.last_report['uses_retarget_history']
    assert init.previous_q.shape==(1,20)
    np.testing.assert_allclose(init.previous_q[0][init.to_native],a[2].cpu(),atol=1e-7)
    same=init(p);assert init.last_report['reused_identical_source']
    for left,right in zip(a,same):torch.testing.assert_close(left,right,atol=0,rtol=0)
    moved=p+np.array([.02,-.01,.03],np.float32)
    b=init(moved);assert init.last_report['uses_retarget_history'] and not init.last_report['uses_cem_history']
    predicted=solver.adapter.keypoints(torch.cat((b[2].new_zeros(6),b[2])))@b[1].T+b[0]
    assert float((predicted-solver.tensor(moved)).norm(dim=-1).max())<.003
    bent=q.clone();bent[6::4]=.9;bent[8::4]=.8;bent[9::4]=.6
    other=solver.adapter.keypoints(bent).cpu().numpy();init(other)
    assert init.last_report['history_reset_reason']=='source_jump_over_40mm' and not init.last_report['uses_retarget_history']
    solver.reset();again=init(p)
    for left,right in zip(a,again):torch.testing.assert_close(left,right,atol=1e-7,rtol=1e-6)


def test_semantics_merge_palms_split_thumb_and_gate_unknown(source,solver):
    _,weights,a=solver.extractor.extract(frame(source,active=range(20)))
    assert {x['region'] for x in a}==set(range(16)) and weights['mask'].sum()==460
    assert len([x for x in a if x['finger']==0])==3 and len([x for x in a if x['region']==0])==1
    f=frame(source,active=range(20));f['source_contact']['timestamp_ms']=900
    assert not solver.extractor.extract(f)[2]
    f['source_contact']['ready']=False;assert not solver.extractor.extract(f)[2]


def test_dynamic_argmin_and_contact_cost_match_numpy_oracle(source,solver):
    solver.reset();(v,j,_,_),_,anchors=solver.extractor.extract(frame(source,active=[0]))
    solver.force_max=anchors[0]['force'];c=solver.context(v,j,anchors);targets=solver.style_targets(v,j)
    t,r,theta=solver.initializer(j,targets['normals'].cpu().numpy());ts=torch.stack((t,t+t.new_tensor([.02,0,0])))
    costs,d=solver.evaluate(ts,r[None].expand(2,-1,-1),theta[None].expand(2,-1),c,True)
    anchor=anchors[0];region=anchor['region'];p,rr=solver.frames(ts,r[None].expand(2,-1,-1),theta[None].expand(2,-1));body=solver.region_body[region]
    pool=solver.pools[region].cpu().numpy();owner=solver.region_owners[region]
    triangles=solver.pad_triangles[region].cpu().numpy()
    psi=solver.surface.axes[owner]*solver.surface.lengths[owner]*anchor['u'];n=solver.surface.normals[owner];psi=psi+n*np.mean((pool-psi)@n)
    choices=[]
    for i in range(2):
        pp=p[i,body].cpu().numpy();rot=rr[i,body].cpu().numpy()
        target=(np.array(anchor['position_m'])-pp)@rot;query=(target+.15*psi)/1.15
        points=trimesh.triangles.closest_point(triangles,np.broadcast_to(query,(len(triangles),3)))
        index=np.sum((points-query)**2,axis=1).argmin();choices.append(points[index])
        np.testing.assert_allclose(d['local_points'][region][i].cpu(),points[index],atol=1e-7)
        face,bary=d['material_points'][region]
        np.testing.assert_allclose(bary[i].cpu().numpy()@triangles[index],points[index],atol=1e-7)
        expected=1.5**2*40**2*np.sum((points[index]@rot.T+pp-anchor['position_m'])**2)*1e6
        assert float(d['parts']['contact_position'][i])==pytest.approx(expected,rel=2e-5)
        normal=rot@solver.pad_normals[region][index].cpu().numpy()
        expected_direction=1.5**2*40**2*2*100*(1-np.clip(normal@anchor['direction'],-1,1))
        assert float(d['parts']['contact_direction'][i])==pytest.approx(expected_direction,rel=2e-5,abs=.1)
    assert not np.allclose(choices[0],choices[1])
    assert not d['parts']['contact_fix'].any()


def test_force_cone_variant_changes_only_direction_force_terms_without_pair_targets(source,solver,monkeypatch):
    from dataclasses import replace
    solver.reset();(v,j,_,_),_,anchors=solver.extractor.extract(frame(source,active=[0]))
    solver.force_max=anchors[0]['force'];c=solver.context(v,j,anchors);t,r,q=solver.initializer(j)
    _,paper=solver.evaluate(t[None],r[None],q[None],c,True)
    monkeypatch.setattr(solver,'settings',replace(solver.settings,contact_direction_model='coulomb_force',friction_coefficient=.5,normal_force_weight=1.))
    _,cone=solver.evaluate(t[None],r[None],q[None],c,True)
    for key in ('contact_position','contact_fix','contact_self_closure','style_tip','style_normal','style_bone','collision_mesh','collision_self','collision_object'):
        torch.testing.assert_close(paper['parts'][key],cone['parts'][key],rtol=0,atol=0)
    assert cone['parts']['contact_direction'][0]==0
    region=anchors[0]['region'];fit=cone['force_support'][region]
    assert fit['projected_in_cone'][0]
    assert cone['parts']['contact_force_vector'][0]>=0 and cone['parts']['contact_normal_force'][0]>=0
    assert not c['self_pairs'] and not solver.self_contact_targets
    empty=solver.context(v,j,[]);_,off=solver.evaluate(t[None],r[None],q[None],empty,True)
    assert not off['force_support'] and off['parts']['contact_normal_force'][0]==0 and off['parts']['contact_force_vector'][0]==0


def test_anti_slip_and_smoothness_millimetres_and_so3(source,solver):
    solver.reset();(v,j,_,_),_,anchors=solver.extractor.extract(frame(source,active=[0]));solver.force_max=anchors[0]['force']
    c=solver.context(v,j,anchors);t,r,q=solver.initializer(j,solver.style_targets(v,j)['normals'].cpu().numpy())
    _,d=solver.evaluate(t[None],r[None],q[None],c,True);region=anchors[0]['region'];solver.previous_contacts[region]=d['local_points'][region][0]
    _,d2=solver.evaluate(t[None],r[None],q[None],c,True);assert float(d2['parts']['contact_fix'][0])<1e-6
    c['previous']=(t+t.new_tensor([.001,0,0]),r@exp_so3(t.new_tensor([.1,0,0])),q+.1)
    _,d3=solver.evaluate(t[None],r[None],q[None],c,True)
    assert float(d3['parts']['smooth_translation'][0])==pytest.approx(125,rel=2e-5)
    old=Rotation.from_matrix(c['previous'][1].cpu().numpy()).as_rotvec()
    current=Rotation.from_matrix(r.cpu().numpy()).as_rotvec();axis=current/np.linalg.norm(current)
    current+=2*np.pi*np.round((old-current)@axis/(2*np.pi))*axis
    assert float(d3['parts']['smooth_rotation'][0])==pytest.approx(100*np.sum((current-old)**2),rel=1e-4)
    assert float(d3['parts']['smooth_joint'][0])==pytest.approx(.002,rel=2e-5)


def test_sdf_degeneracy_exact_inside_distance_and_tracked_transform(source,solver):
    joints=np.array(frame(source)['joints_m']);o,r=source_wrist(joints);field=PseudoObject(solver.device)
    flat=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
    field.add([dict(region=1,position_m=(o+x@r.T).tolist()) for x in flat],joints,1)
    assert not field.metadata()['ready']
    box=np.array([[x,y,z] for x in (0,.01) for y in (0,.01) for z in (0,.01)])
    field.add([dict(region=1,position_m=(o+x@r.T).tolist()) for x in box],joints,2)
    points=solver.tensor(o+np.array([[.005,.005,.005],[.012,.005,.005]])@r.T)
    np.testing.assert_allclose(field.penetration(points,joints).cpu(),[.005,0],atol=1e-7)
    mesh=trimesh.creation.box(extents=[.01]*3);tracked=TrackedMesh(mesh.vertices,mesh.faces,solver.device)
    np.testing.assert_allclose(tracked.penetration(solver.tensor([[0,0,0],[.01,0,0]])).cpu(),[.005,0],atol=1e-7)
    with pytest.raises(ValueError):TrackedMesh(mesh.vertices,mesh.faces,solver.device,np.zeros((4,4)))


def test_best_feasible_real_gpu_output_and_limits(source,solver,feasible_start):
    feasible_start(solver)
    solver.reset();a=solver.step(frame(source));b=solver.step(frame(source,1100.))
    assert a['solver']['sampled_candidates']==16384 and a['solver']['elite_count']==51
    assert b['solution_feasible'] and b['algorithm']==ALGORITHM
    assert np.max(abs(np.array(b['qpos'])[6:]-a['qpos'][6:]))<=.250001
    assert np.linalg.norm(np.array(b['qpos'])[:3]-a['qpos'][:3])<=.020001
    assert set(b['quality']['cost_terms'])=={'contact','style','smooth','collision'}
    assert b['postprocessing']=='raw_cem' and not b['collision']['dynamics']
    assert solver.initializer.artifact['method']=='robokit_hand_retargeter_online'
    assert not b['solver']['initializer']['uses_cem_history']
    timing=b['solver']['timing_ms']
    stages=['extract','robokit','prepare','cem','final_audit','report']
    assert all(np.isfinite(timing[k]) and timing[k]>=0 for k in stages)
    assert sum(timing[k] for k in stages)==pytest.approx(timing['total'])
    assert b['quality']['processing_ms']==timing['total']


def test_causal_field_and_no_contacts_clear_fix_history(source,solver):
    solver.reset();solver.step(frame(source,active=[0,2]));first=solver.sdf.metadata()
    assert first['mode']=='causal' and first['source_range_ms'][1]<=1000
    solver.step(frame(source,1100.));assert not solver.previous_contacts


def test_object_hard_gate_uses_three_mm_and_weight_fourteen(source,solver):
    solver.reset();f=frame(source);(v,j,_,_),_,anchors=solver.extractor.extract(f);c=solver.context(v,j,anchors)
    t,r,q=solver.initializer(j,solver.style_targets(v,j)['normals'].cpu().numpy())
    class Field:
        kind='tracked_mesh'
        def metadata(self):return {'ready':True}
        def penetration(self,p,j):return torch.full(p.shape[:-1],.0031,device=p.device)
    c['sdf']=Field();cost,d=solver.evaluate(t[None],r[None],q[None],c,True)
    assert not torch.isfinite(cost).any()
    count=len(solver.collision_local)
    assert float(d['parts']['collision_object'][0])==pytest.approx(14**2*(3.1-2)**2*count,rel=1e-5)
    c['sdf'].kind='pseudo_object'
    cost,d=solver.evaluate(t[None],r[None],q[None],c,True)
    assert d['hard_object'].all() and not torch.isfinite(cost).any()
    assert float(d['parts']['collision_object'][0])==pytest.approx(3.5**2*(3.1-2)**2*count,rel=1e-5)


def test_search_grading_preserves_every_strict_hard_gate(source,solver):
    solver.reset();(v,j,_,_),_,anchors=solver.extractor.extract(frame(source),False);c=solver.context(v,j,anchors)
    t,r,q=solver.initializer(j);generator=torch.Generator(device=solver.device).manual_seed(42)
    values=(q+torch.randn((64,20),device=solver.device,generator=generator)*.15).clamp(solver.adapter.low[6:],solver.adapter.high[6:])
    tt=t.expand(64,-1);rr=r.expand(64,-1,-1)
    strict=solver.evaluate(tt,rr,values,c);objective,violation=solver.evaluate(tt,rr,values,c,search=True)
    torch.testing.assert_close(torch.isfinite(strict),torch.isfinite(objective)&(violation==0))
    assert torch.isfinite(objective).all() and (violation>=0).all()


def test_unreachable_step_displays_current_initializer_without_claiming_validity(source,solver,monkeypatch,feasible_start):
    feasible_start(solver)
    solver.reset();a=solver.step(frame(source));previous=tuple(v.clone() for v in solver.previous);original=solver.evaluate
    def impossible(t,r,j,c,details=False,search=False):
        if details:return original(t,r,j,c,details)
        return torch.full((len(j),),torch.inf,device=j.device)
    monkeypatch.setattr(solver,'evaluate',impossible)
    b=solver.step(frame(source,1100.,active=[0]))
    assert b['status']=='infeasible' and not b['ready'] and not b['solution_feasible'] and b['display_available']
    assert b['contact_mask']==[None]*460 and not solver.previous_contacts
    initial=(solver.tensor(b['initial_qpos'][:3]),solver.tensor(Rotation.from_euler('XYZ',b['initial_qpos'][3:6]).as_matrix()),solver.tensor(b['initial_qpos'][6:]))
    expected=project(*initial,previous,solver.adapter.low[6:],solver.adapter.high[6:],solver.settings)
    np.testing.assert_allclose(np.r_[b['qpos'][:3],b['qpos'][6:]],np.r_[expected[0].cpu(),expected[2].cpu()],atol=1e-6)
    np.testing.assert_allclose(Rotation.from_euler('XYZ',b['qpos'][3:6]).as_matrix(),Rotation.from_euler('XYZ',b['initial_qpos'][3:6]).as_matrix(),atol=1e-6)
    assert not b['solver']['used_history_fallback'] and b['solver']['previous_evaluated_as_incumbent']
    assert b['solver']['output_selection']=='current_initializer_no_finite_objective'
    for x,y in zip(previous,solver.previous):torch.testing.assert_close(x,y)


def test_offline_prepass_uses_full_clip_and_centered_surface_coordinates(source,solver):
    from retarget.baseline.card.offline import prepare_clip
    solver.reset();f=[frame(source,1000.,[0]),frame(source,1100.,range(20))]
    info=prepare_clip(solver,f)
    assert solver.sdf.frozen and solver.sdf.metadata()['ready']
    assert solver.sdf.metadata()['source_range_ms']==[1000.,1100.] and info['object_mode']=='whole_clip'
    assert solver.offline_force_max>0 and len(solver.offline_anchors[1000.])==1
    assert len(solver.offline_anchors[1100.])==16
    for a in solver.offline_anchors[1100.]:
        v=np.asarray(f[1]['vertices_m'])@solver.mirror.T
        np.testing.assert_allclose(np.asarray(a['barycentric'])@v[np.array(source['faces'])[a['source_face']]],a['position_m'],atol=1e-8)
    solver.reset()


def test_supported_self_contacts_do_not_create_a_solid_pseudo_object(source,solver,monkeypatch):
    from dataclasses import replace
    from retarget.baseline.card.offline import prepare_clip
    monkeypatch.setattr(solver,'settings',replace(solver.settings,experimental_self_contact=True))
    solver.reset();frames=[frame(source,1000.,[0,5]),frame(source,1100.,[0,5])]
    for f in frames:f['self_contact_pairs']=[{'regions':[3,9]}]
    prepare_clip(solver,frames)
    assert solver.self_contact_pairs==[(3,9)]
    assert solver.sdf.metadata()['vertices']==0 and not solver.sdf.metadata()['ready']
    solver.reset()


def test_two_finger_palm_contacts_keep_distinct_palm_surface_points(source,solver,monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(solver,'settings',replace(solver.settings,experimental_self_contact=True))
    solver.reset();f=frame(source,active=range(20));(v,j,_,_),_,anchors=solver.extractor.extract(f)
    t,r,q=solver.initializer(j,solver.style_targets(v,j)['normals'].cpu().numpy());p,rr=solver.frames(t,r,q)
    local=solver.pools[0];a=int(local[:,0].argmin());b=int(local[:,0].argmax());body=solver.region_body[0]
    targets=[p[body]+rr[body]@local[k] for k in (a,b)]
    f['self_contact_pairs']=[dict(regions=[0,region],point_a_m=(point.cpu().numpy()@solver.mirror).tolist(),point_b_m=(point.cpu().numpy()@solver.mirror).tolist()) for region,point in zip((6,9),targets)]
    solver.update_self_contacts(f,anchors);c=solver.context(v,j,anchors)
    _,detail=solver.evaluate(t[None],r[None],q[None],c,True)
    selected=[detail['pair_points'][(0,region)][0][0] for region in (6,9)]
    for actual,expected in zip(selected,targets):torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    assert (selected[0]-selected[1]).norm()>.01
    solver.reset()


def test_offline_filter_validates_bounds_and_preserves_raw_result(source,solver,feasible_start):
    feasible_start(solver)
    from retarget.baseline.card.offline import prepare_clip,filter_clip
    solver.reset();first=solver.step(frame(source));assert first['solution_feasible']
    frames=[frame(source,1000.+i*50.) for i in range(17)]
    results=[]
    for i,f in enumerate(frames):
        result=copy.deepcopy(first);result['raw_qpos'][0]+=.0001*np.sin(i);result['qpos']=result['raw_qpos'].copy();results.append(result)
    prepare_clip(solver,frames);filtered=filter_clip(solver,frames,results)
    assert all(x['postprocessing']=='sg15_order3_validated' and x['filtered_qpos'] for x in filtered)
    for raw,new in zip(results,filtered):assert new['raw_qpos']==raw['raw_qpos']
    q=np.array([r['qpos'] for r in filtered]);assert np.max(abs(np.diff(q[:,6:],axis=0)))<=.250001
    assert np.max(np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1))<=.020001


def test_offline_jsonl_replay_has_no_old_lm_results(source,tmp_path):
    from scripts.retarget_moxian_wuji2 import replay
    src=tmp_path/'source.jsonl';dst=tmp_path/'output.jsonl'
    rows=[{'hand_model_topology':source,'hand_model':frame(source)}, {'hand_model':frame(source,1100.)}, {'hand_model':frame(source,1100.),'wuji2':{'algorithm':'old_lm'}}, {'hand_model':{},'wuji2':{'algorithm':'old_lm'}}]
    src.write_text('\n'.join(json.dumps(r) for r in rows))
    info=replay(src,dst,cache=Path('output/moxian/debug/card/test-cache'),solver='cem')
    out=[json.loads(x) for x in dst.read_text().splitlines()]
    assert info['solved_frames']==2 and info['duplicate_or_missing_frames']==2
    assert out[2]['wuji2']==out[1]['wuji2']
    assert out[-1]['wuji2'] is None
    assert out[0]['wuji2']['algorithm']==ALGORITHM
    assert out[0]['wuji2']['raw_qpos'] and not out[0]['wuji2']['solution_feasible']
    assert out[0]['wuji2']['offline_preprocessing']['object_mode']=='whole_clip'
    with pytest.raises(FileExistsError):replay(src,dst)


def test_offline_filter_rechecks_each_tracked_object_pose(source,solver,monkeypatch,feasible_start):
    feasible_start(solver)
    from retarget.baseline.card.offline import prepare_clip,filter_clip
    solver.reset();raw=solver.step(frame(source));assert raw['solution_feasible']
    mesh=trimesh.creation.box(extents=[.01]*3);provider=TrackedMesh(mesh.vertices,mesh.faces,solver.device)
    monkeypatch.setattr(solver,'external_sdf',provider);monkeypatch.setattr(solver,'sdf',provider)
    frames=[frame(source,1000.),frame(source,1100.)]
    for i,f in enumerate(frames):
        transform=np.eye(4);transform[0,3]=1.+i*.1;f['object_transform']=transform.tolist()
    prepare_clip(solver,frames);observed=[];original=provider.penetration
    def check(points,joints=None):
        observed.append(provider.transform[0,3]);return original(points,joints)
    monkeypatch.setattr(provider,'penetration',check)
    version=solver.topology()['topology_version']
    filtered=filter_clip(solver,frames,[copy.deepcopy(raw),copy.deepcopy(raw)])
    assert all(r['solution_feasible'] for r in filtered)
    assert set(observed)=={1.,1.1} and solver.topology()['topology_version']==version
    solver.reset()


def test_worker_latest_frame_transport_reset_and_card_payload(source,tmp_path):
    import os,sys,subprocess,threading,queue,time,shutil
    cache=Path('output/moxian/debug/card/test-cache');shutil.copytree(cache,tmp_path/'mappings')
    fd,write_fd=os.pipe();log=(tmp_path/'worker.log').open('w')
    root=Path(__file__).resolve().parents[2]
    proc=subprocess.Popen([sys.executable,'-I',str(root/'moxian_web/wuji_worker.py'),'--solver','cem','--event-fd',str(write_fd),'--home',str(tmp_path)],stdin=subprocess.PIPE,stdout=log,stderr=log,text=True,pass_fds=(write_fd,))
    os.close(write_fd);events=queue.Queue()
    def reader():
        with os.fdopen(fd) as stream:
            for line in stream:events.put(json.loads(line))
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    def send(message):proc.stdin.write(json.dumps(message)+'\n');proc.stdin.flush()
    def wait_for(predicate):
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            try:event=events.get(timeout=.1)
            except queue.Empty:continue
            if event.get('status')=='error':pytest.fail(str(event)+' '+(tmp_path/'worker.log').read_text()[-2000:])
            if predicate(event):return event
        pytest.fail('Worker event timed out: '+(tmp_path/'worker.log').read_text()[-2000:])
    try:
        send({'type':'topology','model':source});wait_for(lambda x:x.get('status')=='initializing')
        stamp=time.time()*1000
        for i in range(20):send({'type':'frame','frame':frame(source,stamp+i)});time.sleep(.005)
        result=wait_for(lambda x:x.get('type')=='frame' and x['source_timestamp_ms']==stamp+19)
        assert len(result['qpos'])==26 and result['algorithm']==ALGORITHM
        assert result['solver']['sampled_candidates']==16384
        session=result['solver_session_id']
        send({'type':'control','action':'reset','id':'reset'});wait_for(lambda x:x.get('id')=='reset')
        send({'type':'frame','frame':frame(source,stamp+1000)})
        newer=wait_for(lambda x:x.get('type')=='frame')
        assert newer['solver_session_id']!=session
        assert newer['settings']['iterations']==16
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        proc.stdin.close();thread.join(timeout=1);log.close()


def test_native_euler_chart_stays_continuous_through_gimbal_crossing():
    from retarget.baseline.card.math import continuous_euler_xyz
    previous=np.array([.3,1.55,-.4])
    for y in (1.56,np.pi/2,1.58,1.59):
        matrix=Rotation.from_euler('XYZ',[.3,y,-.4]).as_matrix()
        angles=continuous_euler_xyz(matrix,previous)
        np.testing.assert_allclose(Rotation.from_euler('XYZ',angles).as_matrix(),matrix,atol=1e-7)
        assert np.max(abs(angles-previous))<.03;previous=angles


@pytest.fixture(scope='module')
def robokit_solver(source):
    if not torch.cuda.is_available():pytest.skip('CUDA needed')
    torch.set_num_threads(1)
    return TactileRetarget(source,Path('output/moxian/debug/card/test-cache'))


def test_robokit_card_cost_matches_independent_reference_with_fitted_capsules(source,robokit_solver,monkeypatch):
    from retarget.baseline.card.robokit import RobokitCardSolver,PARTS
    o=robokit_solver;o.reset();(v,j,_,_),_,anchors=o.extractor.extract(frame(source,active=[0,4,9]))
    o.force_max=max(a['force'] for a in anchors)/2;c=o.context(v,j,anchors);initial=o.initializer(j)
    c['previous']=(initial[0]+.001,initial[1],initial[2]+.01)
    c['motion_joint']=None
    o.previous_contacts={a['region']:o.pools[a['region']][4].clone() for a in anchors}
    backend=RobokitCardSolver(o);backend.task.update(c);got=backend.parts(initial)
    d=backend.task.data
    monkeypatch.setattr(o,'cap_a',o.tensor(d.cap_a.numpy()));monkeypatch.setattr(o,'cap_b',o.tensor(d.cap_b.numpy()));monkeypatch.setattr(o,'radii',o.tensor(d.cap_radius.numpy()))
    _,detail=o.evaluate(initial[0][None],initial[1][None],initial[2][None],c,True)
    for k in PARTS:
        assert got[k]==pytest.approx(float(detail['parts'][k][0]),rel=3e-4,abs=.05),k


def test_robokit_card_sampler_matches_paper_elite_softmax_and_annealing():
    import warp as wp
    from robokit.opt import VectorVar,VarValues
    from robokit.opt.var_values import VarKey
    from retarget.baseline.card.robokit import CardSampler
    sampler=CardSampler(num_draws=64,tangent_dim=26,instances=1,init_std=.001,min_std=1e-9,
        temperature=.1,gamma=0.,noise_decay=.8,elite_ratio=.1,mean_elites=True,device='cuda:0')
    sampler.lower=wp.full((1,26),-1.,dtype=float,device='cuda:0');sampler.upper=wp.full((1,26),1.,dtype=float,device='cuda:0');sampler.radius=1.
    var=VarValues.from_var(VectorVar(wp.zeros((1,26),dtype=float,device='cuda:0')),VarKey.ROBOT)
    draws=sampler.sample(var,None);x=draws.get(VarKey.ROBOT).x.numpy().copy();cost=np.arange(64,dtype=np.float32)[::-1].copy()
    ids=np.argsort(cost)[:6];weights=np.exp(-(cost[ids]-cost[ids].min())/cost[ids].std()/.1);weights/=weights.sum()
    sampler.sample(draws,wp.array(cost,dtype=float,device='cuda:0'))
    np.testing.assert_allclose(var.get(VarKey.ROBOT).x.numpy()[0],(weights[:,None]*x[ids]).sum(0),atol=2e-8)
    np.testing.assert_allclose(sampler.std.numpy(),.0008,atol=1e-9)


def test_robokit_card_captured_solve_budget_previous_bounds_and_global_best(source,robokit_solver,feasible_start):
    feasible_start(robokit_solver)
    o=robokit_solver;o.reset();a=o.step(frame(source));b=o.step(frame(source,1100.))
    assert b['solver']['backend']=='robokit_population_solver_warp_card' and b['solver']['cuda_graph']
    assert b['solver']['sampled_candidates']==16384 and b['solver']['evaluations']==16387
    assert b['solver']['feasible'] and b['collision']['min_palm_gap_mm']>=6
    assert b['solver']['previous_used_as_search_center'] and not b['solver']['used_history_fallback']
    assert np.max(np.abs(np.array(b['qpos'])[6:]-a['qpos'][6:]))<=.250001
    assert np.linalg.norm(np.array(b['qpos'])[:3]-a['qpos'][:3])<=.020001
    assert sum(b['quality']['cost_terms'].values())==pytest.approx(b['solver']['best_cost'],rel=3e-4)
    assert b['solver']['dense_verified_candidates']==0 and b['solver']['validation_limit'] is None
    assert b['solver']['style_finger_weights']==[1.]*5 and b['solver']['smooth_joint_units']=='rad'
    assert b['solver']['joint_motion_reference'] is None and b['solver']['source_motion_samples_per_round']==0
    assert b['solver']['output_selection_scope']=='all_scored_candidates'
    widths=o.robokit_cem.task.archive_width.numpy();costs=o.robokit_cem.task.archive_cost.numpy()
    archived=[costs[i*1024:i*1024+n].min() for i,n in enumerate(widths) if n]
    assert b['solver']['best_cost']==pytest.approx(min(*archived,b['solver']['initial_cost'],b['solver']['previous_incumbent_cost']),rel=1e-5)


def test_robokit_card_previous_incumbent_has_current_cost_without_native_gate(source,robokit_solver,monkeypatch,feasible_start):
    feasible_start(robokit_solver)
    o=robokit_solver;o.reset();o.step(frame(source));r=o.step(frame(source,1100.))
    assert r['solver']['previous_evaluated_as_incumbent'] and r['solver']['previous_used_as_search_center']
    assert r['solver']['best_cost']<=r['solver']['previous_incumbent_cost']+1e-3
    monkeypatch.setattr(o,'verify_mesh',lambda *args:pytest.fail('native diagnostic must not gate CEM'))
    result=o.step(frame(source,1200.))
    assert result['solution_feasible'] and not result['solver']['used_history_fallback']


def test_robokit_card_costs_react_to_pose_and_temporal_reference(source,robokit_solver):
    from retarget.baseline.card.robokit import RobokitCardSolver
    o=robokit_solver;o.reset();(v,j,_,_),_,anchors=o.extractor.extract(frame(source));initial=o.initializer(j)
    c=o.context(v,j,anchors)
    if o.robokit_cem is None:o.robokit_cem=RobokitCardSolver(o)
    task=o.robokit_cem;task.task.update(c);base=task.parts(initial)
    changed={**c,'tips':c['tips']+.02}
    task.task.update(changed);style=task.parts(initial)
    assert style['style_tip']!=pytest.approx(base['style_tip'])
    changed['previous']=(initial[0],initial[1],initial[2]+.2);task.task.update(changed);smooth=task.parts(initial)
    assert smooth['smooth_joint']==pytest.approx(CardSettings().wtheta**2*20*.2**2,rel=1e-4)


def test_contacted_fingers_keep_style_without_default_motion_prior(source,robokit_solver):
    o=robokit_solver;o.reset();(v,j,_,_),_,anchors=o.extractor.extract(frame(source,active=[0,4,9]));initial=o.initializer(j)
    o.previous=tuple(x.clone() for x in initial);o.previous_initial_joint=initial[2].clone()
    c=o.context(v,j,anchors,initial)
    assert c['free'].tolist()==[1.]*5 and c['motion_joint'] is None
    from retarget.baseline.card.robokit import RobokitCardSolver
    backend=RobokitCardSolver(o);backend.task.update(c)
    assert backend.parts(initial)['motion_joint']==0
    assert o.settings.objective_profile=='paper'


def test_gaussian_population_has_no_forced_center_or_source_motion_cloud():
    import warp as wp
    from robokit.opt import VectorVar,VarValues
    from robokit.opt.var_values import VarKey
    from retarget.baseline.card.robokit import CardSampler
    sampler=CardSampler(num_draws=32,tangent_dim=26,instances=1,init_std=.01,seed=7,device='cuda:0')
    sampler.lower=wp.array(np.r_[[-.02]*3,[-1.]*3,[-.25]*20][None].astype(np.float32),device='cuda:0')
    sampler.upper=wp.array(-sampler.lower.numpy(),device='cuda:0');sampler.radius=.02
    center=np.r_[[0.]*6,[.249]*20][None].astype(np.float32)
    var=VarValues.from_var(VectorVar(wp.array(center,device='cuda:0')),VarKey.ROBOT)
    x=sampler.sample(var,None).get(VarKey.ROBOT).x.numpy();noise=sampler.noise.numpy()
    np.testing.assert_allclose(x-center,noise,atol=2e-8)
    assert np.linalg.norm(x[:,:3],axis=1).max()<=.020001
    assert np.any(noise[0]!=0) and np.all(x>=sampler.lower.numpy()) and np.all(x<=sampler.upper.numpy())
    assert np.any(x[:,6:]==.25) and np.any(x[:,6:]<.249)


def test_mixed_source_motion_uses_one_rotation_chart_through_pi():
    from retarget.baseline.card.robokit import pack
    r=Rotation.from_rotvec([np.pi+.001,0,0]).as_matrix()
    initial=(torch.zeros(3),torch.tensor(r),torch.zeros(20))
    packed=pack(initial,np.array([np.pi-.001,0,0]))
    assert np.linalg.norm(packed[3:6]-[np.pi-.001,0,0])<.002001
    np.testing.assert_allclose(Rotation.from_rotvec(packed[3:6]).as_matrix(),r,atol=1e-7,rtol=0)


def test_robokit_card_source_jump_still_respects_previous_output_velocity(source,robokit_solver,feasible_start):
    feasible_start(robokit_solver)
    o=robokit_solver;o.reset();a=o.step(frame(source));b=o.step(frame(source,1100.))
    moved=frame(source,1200.);delta=np.array([.05,-.02,.01])
    for key in ('vertices_m','joints_m'):moved[key]=(np.array(moved[key])+delta).tolist()
    c=o.step(moved)
    assert set(c['solver']['temporal_candidate_costs'])=={'previous'}
    assert c['solver']['previous_used_as_search_center'] and c['solution_feasible']
    assert np.linalg.norm(np.array(c['qpos'])[:3]-np.array(b['qpos'])[:3])<=.020001


def test_native_mesh_diagnostic_does_not_change_solution_or_history(source,robokit_solver,monkeypatch,feasible_start):
    feasible_start(robokit_solver)
    o=robokit_solver;o.reset();f=frame(source)
    baseline=o.step(f)
    o.reset();query=o.mesh_collision.query
    def deep(*args,**kwargs):
        depth,valid=query(*args,**kwargs)
        return depth+10.,valid
    monkeypatch.setattr(o.mesh_collision,'query',deep)
    result=o.step(f)
    assert result['solution_feasible'] and result['collision']['mesh_diagnostic_exceeded']
    assert not result['collision']['mesh_hard_rejected'] and o.previous is not None
    np.testing.assert_allclose(result['qpos'],baseline['qpos'],atol=1e-6)
    assert result['solver']['dense_verified_candidates']==0


@pytest.mark.parametrize('tracked',[False,True])
def test_robokit_card_object_cost_and_material_coordinates(source,robokit_solver,monkeypatch,tracked):
    from retarget.baseline.card.robokit import RobokitCardSolver
    from scipy.spatial import ConvexHull
    o=robokit_solver;o.reset();(v,j,_,_),_,anchors=o.extractor.extract(frame(source,active=[0,4,9]))
    o.force_max=max(a['force'] for a in anchors)
    if tracked:
        box=trimesh.creation.box([1.,1.,1.]);transform=np.eye(4)
        transform[:3,:3]=Rotation.from_rotvec([.1,.2,.3]).as_matrix();transform[:3,3]=[.01,.02,.03]
        field=TrackedMesh(box.vertices,box.faces,o.device,transform)
    else:
        field=PseudoObject(o.device);points=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        field.hull=ConvexHull(points);field.points=points;field.planes=o.tensor(field.hull.equations)
    monkeypatch.setattr(o,'sdf',field);c=o.context(v,j,anchors);initial=o.initializer(j)
    backend=RobokitCardSolver(o);backend.task.update(c);actual=backend.details(initial,c)
    if tracked:assert backend.parts(initial,total=True)>=1e29
    _,expected=o.evaluate(initial[0][None],initial[1][None],initial[2][None],c,True)
    assert float(actual['parts']['collision_object'][0])>0
    torch.testing.assert_close(actual['parts']['collision_object'],expected['parts']['collision_object'],rtol=3e-5,atol=.5)
    torch.testing.assert_close(actual['object_max_penetration_mm'],expected['object_max_penetration_mm'],rtol=1e-5,atol=1e-3)
    for a in anchors:
        region=a['region'];face,bary=actual['material_points'][region]
        local_index=np.flatnonzero(o.pad_face_ids[region].cpu().numpy()==int(face[0]))[0]
        reconstructed=bary[0]@o.pad_triangles[region][local_index]
        torch.testing.assert_close(reconstructed,actual['local_points'][region][0],atol=1e-7,rtol=1e-5)
        # Independent FP32 BVH and analytic triangle projections agree within 1 um.
        torch.testing.assert_close(actual['points'][region],expected['points'][region],atol=1e-6,rtol=1e-5)


def test_translated_triangle_distance_uses_relative_projection_once():
    from retarget.baseline.card.geometry import triangle_distance
    triangle=torch.tensor([[[2.,3.,4.],[4.,3.,4.],[2.,5.,4.]]])
    point=torch.tensor([[2.5,3.5,4.2]])
    torch.testing.assert_close(triangle_distance(point,triangle),torch.tensor([.2]),atol=1e-6,rtol=1e-6)


def test_robokit_card_selects_entire_archive_without_native_shortlist(source,robokit_solver,monkeypatch):
    import warp as wp
    from retarget.baseline.card.robokit import RobokitCardSolver,pack
    o=robokit_solver;o.reset();(v,j,_,_),_,anchors=o.extractor.extract(frame(source));c=o.context(v,j,anchors);initial=o.initializer(j)
    backend=RobokitCardSolver(o);backend.initialized=True;backend.solver.samplers=[]
    winner=pack(initial);winner[6]+=.01
    def fake_solve(var):
        values=np.tile(pack(initial),(1024,1));values[777]=winner
        costs=np.full(1024,np.inf,np.float32);costs[777]=1.
        wp.copy(backend.task.archive_x,wp.array(values,device=str(o.device)),count=1024*26)
        wp.copy(backend.task.archive_cost,wp.array(costs,device=str(o.device)),count=1024)
        backend.task.archive_width.assign(np.r_[1024,np.zeros(len(backend.task.archive_width)-1)].astype(np.int32))
    monkeypatch.setattr(backend.solver,'solve',fake_solve)
    monkeypatch.setattr(backend,'_parts_vector',lambda *args,**kwargs:2.)
    monkeypatch.setattr(o,'verify_mesh',lambda *args:pytest.fail('no shortlist verification'))
    best,report=backend.solve(initial,c)
    assert report['feasible'] and report['best_cost']==1.
    torch.testing.assert_close(best[2],o.tensor(winner[6:]))
    assert report['validation_limit'] is None and report['output_selection_scope']=='all_scored_candidates'


@pytest.mark.parametrize('cost_values',[
    [4.,3.,2.,1.,0.,5.,6.,7.],
    [float('inf'),float('nan'),2.,float('-inf'),1.,float('inf'),float('inf'),float('inf')],
    [float('inf')]*8,
    [7.]*8,
])
def test_warp_and_torch_updates_share_injected_candidates_and_finite_elites(cost_values):
    import warp as wp
    from retarget.baseline.card.robokit import CardSampler
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    cost=np.array(cost_values,np.float32);rng=np.random.default_rng(71)
    noise=rng.normal(size=(8,26)).astype(np.float32)*.01
    sampler=CardSampler(num_draws=8,tangent_dim=26,instances=1,init_std=.01,elite_ratio=.5,
        temperature=.1,noise_decay=.8,device=device)
    wp.copy(sampler.noise,wp.array(noise,device=device))
    delta=wp.zeros((1,26),dtype=float,device=device)
    sampler._move(wp.array(cost,device=device),delta)
    ids,weights=elite_weights(torch.tensor(cost),4,.1)
    expected=np.zeros(26,np.float32)
    finite=np.flatnonzero(np.isfinite(cost));finite=finite[np.argsort(cost[finite],kind='stable')][:4]
    if len(finite):
        scores=cost[finite].astype(np.float64)
        oracle=np.exp(-(scores-scores.min())/(max(scores.std(),1e-12)*.1));oracle/=oracle.sum()
        expected=(noise[finite]*oracle[:,None]).sum(0)
        np.testing.assert_array_equal(ids.numpy(),finite)
        np.testing.assert_allclose(weights.numpy(),oracle,atol=1e-7)
    actual=sampler.weights.numpy()[0]
    assert np.isfinite(actual).all() and np.isfinite(delta.numpy()).all()
    np.testing.assert_allclose(delta.numpy()[0],expected,atol=1e-8)
    np.testing.assert_allclose(sampler.std.numpy(),.008,atol=1e-9)
    if len(finite):
        _,scaled=elite_weights(torch.tensor(cost)*100,4,.1)
        torch.testing.assert_close(weights,scaled)
    else:assert not actual.any() and not delta.numpy().any()


@pytest.mark.parametrize('zero_force',[False,True])
def test_force_priority_uses_current_frame_and_zero_force_is_one(source,robokit_solver,zero_force):
    from retarget.baseline.card.robokit import RobokitCardSolver
    o=robokit_solver;o.reset();(v,j,_,_),_,anchors=o.extractor.extract(frame(source,active=[0,4,9]))
    anchors=copy.deepcopy(anchors)
    for index,anchor in enumerate(anchors):anchor['force']=0. if zero_force else float(index+1)
    initial=o.initializer(j);c=o.context(v,j,anchors)
    o.force_max=1e9;o.offline_force_max=1e12
    backend=RobokitCardSolver(o);backend.task.update(c)
    _,detail=o.evaluate(*(value[None] for value in initial),c,True)
    got=backend.parts(initial)
    expected_position=0.
    for anchor in anchors:
        rho=1. if zero_force else 1+.5*anchor['force']/max(a['force'] for a in anchors)
        point=detail['points'][anchor['region']][0].cpu().numpy()
        expected_position+=rho**2*40**2*np.sum((point-anchor['position_m'])**2)*1e6
    assert float(detail['parts']['contact_position'][0])==pytest.approx(expected_position,rel=2e-5)
    assert got['contact_position']==pytest.approx(expected_position,rel=2e-5)
    np.testing.assert_allclose(backend.task.data.rho.numpy()[[a['region'] for a in anchors]],
        [1. if zero_force else 1+.5*a['force']/len(anchors) for a in anchors],rtol=1e-6)


@pytest.mark.parametrize('gap_mm',[5.9,6.1])
def test_palm_gate_and_soft_hinge_match_independent_scalar_formula(source,robokit_solver,monkeypatch,gap_mm):
    import warp as wp
    from retarget.baseline.card.robokit import RobokitCardSolver
    o=robokit_solver;o.reset();(v,j,_,_),_,anchors=o.extractor.extract(frame(source));initial=o.initializer(j)
    # A single point-capsule pair makes gap and soft cost independently known.
    monkeypatch.setattr(o,'cross_pairs',[]);monkeypatch.setattr(o,'palm_pairs',[(0,3)])
    monkeypatch.setattr(o,'pair_a',o.tensor([0],torch.long));monkeypatch.setattr(o,'pair_b',o.tensor([3],torch.long))
    cap=o.tensor(np.zeros((16,3)));p,_=o.frames(*(value for value in initial))
    distance=float((p[o.region_body[0]]-p[o.region_body[3]]).norm())
    radii=o.tensor(np.zeros(16));radii[0]=distance-gap_mm/1000
    monkeypatch.setattr(o,'cap_a',cap);monkeypatch.setattr(o,'cap_b',cap);monkeypatch.setattr(o,'radii',radii)
    c=o.context(v,j,anchors);backend=RobokitCardSolver(o)
    backend.task.data.cap_a=wp.from_torch(cap,dtype=wp.vec3);backend.task.data.cap_b=wp.from_torch(cap,dtype=wp.vec3)
    backend.task.data.cap_radius=wp.from_torch(radii);backend.task.update(c)
    cost,detail=o.evaluate(*(value[None] for value in initial),c,True)
    warp_cost=backend.parts(initial,total=True);parts=backend.parts(initial)
    assert bool(detail['hard_palm'][0])==(gap_mm<6)
    assert np.isfinite(warp_cost)==bool(torch.isfinite(cost[0]))==(gap_mm>=6)
    expected=1.5**2*max(0.,6-gap_mm)**2
    assert float(detail['parts']['collision_self'][0])==pytest.approx(expected,abs=1e-5)
    assert parts['collision_self']==pytest.approx(expected,abs=1e-5)


def test_style_and_smooth_energy_groups_match_numpy_formulas(source,robokit_solver):
    from retarget.baseline.card.robokit import RobokitCardSolver
    o=robokit_solver;o.reset();(v,j,_,_),_,anchors=o.extractor.extract(frame(source,active=[0,4,9]))
    t,r,q=o.initializer(j);c=o.context(v,j,anchors)
    old_vector=Rotation.from_matrix(r.cpu().numpy()).as_rotvec()+np.array([.03,-.04,.02])
    c['previous']=(t+o.tensor([.001,-.002,.003]),o.tensor(Rotation.from_rotvec(old_vector).as_matrix()),q+.02)
    _,detail=o.evaluate(t[None],r[None],q[None],c,True)
    p,rr=o.frames(t,r,q);p=p[o.region_body].cpu().numpy();rr=rr[o.region_body].cpu().numpy()
    tip_regions=[3,6,9,12,15];apices=o.apices.cpu().numpy()
    tips=p[tip_regions]+np.einsum('nij,nj->ni',rr[tip_regions],apices[tip_regions])
    normals=np.einsum('nij,nj->ni',rr[tip_regions],o.surface.normals[o.region_owners[tip_regions]])
    chains=np.concatenate((p[1:].reshape(5,3,3),tips[:,None]),axis=1)
    bones=np.diff(chains,axis=1);bones/=np.linalg.norm(bones,axis=-1,keepdims=True)
    expected=dict(style_tip=3**2*np.sum(((tips-c['tips'].cpu().numpy())*1000)**2),
        style_normal=60**2*np.sum(1-np.clip(np.sum(normals*c['normals'].cpu().numpy(),axis=-1),-1,1)),
        style_bone=20**2*np.sum(1-np.clip(np.sum(bones*c['directions'].cpu().numpy(),axis=-1),-1,1)),
        smooth_joint=.1**2*20*.02**2,smooth_translation=125*(1+4+9),
        smooth_rotation=100*np.sum((Rotation.from_matrix(r.cpu().numpy()).as_rotvec()-old_vector)**2))
    backend=RobokitCardSolver(o);backend.task.update(c);actual=backend.parts((t,r,q))
    for key,value in expected.items():
        assert float(detail['parts'][key][0])==pytest.approx(value,rel=2e-4,abs=.05),key
        assert actual[key]==pytest.approx(value,rel=2e-4,abs=.05),key
    assert detail['parts']['motion_joint'][0]==0 and detail['parts']['collision_mesh'][0]==0


@pytest.mark.parametrize('kind',['tracked_mesh','pseudo_object'])
@pytest.mark.parametrize('depth_mm',[2.9,3.1])
def test_both_object_fields_enforce_three_mm_gate_and_paper_weight(source,robokit_solver,monkeypatch,feasible_start,kind,depth_mm):
    from scipy.spatial import ConvexHull
    from retarget.baseline.card.robokit import RobokitCardSolver
    o=robokit_solver;feasible_start(o);o.reset()
    (v,j,_,_),_,anchors=o.extractor.extract(frame(source));initial=o.initializer(j)
    positions,rotations=o.frames(*initial)
    local=(rotations[o.region_body].transpose(-1,-2)@(initial[0]-positions[o.region_body])[...,None]).squeeze(-1)
    monkeypatch.setattr(o,'collision_local',local)
    monkeypatch.setattr(o,'collision_regions',o.tensor(np.arange(16),torch.long))
    box=trimesh.creation.box(extents=[2*depth_mm/1000]*3)
    if kind=='tracked_mesh':
        transform=np.eye(4);transform[:3,3]=initial[0].cpu().numpy()
        field=TrackedMesh(box.vertices,box.faces,o.device,transform)
        weight=14.
    else:
        origin,rotation=source_wrist(j);center=(initial[0].cpu().numpy()-origin)@rotation
        points=np.asarray(box.vertices)+center
        field=PseudoObject(o.device);field.hull=ConvexHull(points);field.points=points
        field.planes=o.tensor(field.hull.equations);weight=3.5
    c={**o.context(v,j,anchors),'sdf':field}
    cost,detail=o.evaluate(*(value[None] for value in initial),c,True)
    backend=RobokitCardSolver(o);backend.task.update(c)
    parts=backend.parts(initial);warp_cost=backend.parts(initial,total=True)
    assert bool(torch.isfinite(cost[0]))==np.isfinite(warp_cost)==(depth_mm<3)
    assert bool(detail['hard_object'][0])==(depth_mm>3)
    expected=16*weight**2*(depth_mm-2)**2
    # FK, inverse transforms and BVH all use FP32; allow the resulting
    # sub-micrometre position error in the sum of 16 squared depths.
    assert float(detail['parts']['collision_object'][0])==pytest.approx(expected,rel=1e-4,abs=.01)
    assert parts['collision_object']==pytest.approx(expected,rel=1e-4,abs=.01)


def test_cem_samples_and_averages_in_one_axis_angle_chart_across_pi(monkeypatch):
    settings=CardSettings(population=4,iterations=1,elite_fraction=1.)
    previous=(torch.zeros(3,dtype=torch.float64),exp_so3(torch.tensor([np.pi-.01,0.,0.],dtype=torch.float64)),torch.zeros(20,dtype=torch.float64))
    initial=(previous[0],exp_so3(torch.tensor([np.pi+.01,0.,0.],dtype=torch.float64)),previous[2])
    innovations=torch.zeros((4,26),dtype=torch.float64)
    innovations[:,3:6]=torch.tensor([[-1.,2.,0.],[1.,-2.,0.],[.5,0.,2.],[-.5,0.,-2.]])
    monkeypatch.setattr(torch,'randn',lambda shape,**kwargs:innovations.to(device=kwargs['device'],dtype=kwargs['dtype']))
    seen=[]
    def objective(t,r,j):
        seen.append(axis_angle_near(r,axis_angle_near(previous[1])))
        if len(seen)==2:return torch.tensor([1.,2.,3.,4.],dtype=j.dtype)
        return torch.full((len(j),),100. if len(seen)==1 else -1.,dtype=j.dtype)
    best,_=cem(initial,previous,-torch.ones(20),torch.ones(20),objective,settings,torch.Generator())
    candidates=torch.tensor([np.pi,0.,0.],dtype=torch.float64)+innovations[:,3:6]*.016
    torch.testing.assert_close(seen[1],candidates,atol=1e-10,rtol=1e-10)
    costs=np.arange(1.,5.);weights=np.exp(-(costs-costs.min())/costs.std()/.1);weights/=weights.sum()
    expected=(weights[:,None]*candidates.numpy()).sum(0)
    np.testing.assert_allclose(best[1].numpy(),Rotation.from_rotvec(expected).as_matrix(),atol=1e-10)
