"""Physical contact, implicit integration and shared patch criteria checks."""
from dataclasses import replace
import torch,pytest
from retarget.env.grasp.spring_history import SpringHistorySettings,SpringHistoryRollout,spring_force
from retarget.env.grasp.comfree_rollout import RolloutSettings
from retarget.env.grasp.mano_patch_anchor import PatchAnchorSettings
from retarget.env.grasp.mano_patch_consistent import patch_fit_terms
from retarget.solver.base import Problem
from retarget.solver.cem import SamplingSettings as Settings
from retarget.solver.cem import _solve_patch_cem as solve_patch_cem


def test_tangential_memory_produces_static_shear_without_sliding_velocity():
    n=torch.tensor([1.,0,0]);history=torch.tensor([0.,0,.0005]);v=torch.zeros(3)
    f,fn,ft=spring_force(v,n,torch.tensor(.005),history,.0025,SpringHistorySettings(),.6)
    torch.testing.assert_close(ft,torch.tensor([0.,0.,-.25]));assert fn==2.5
    _,fn,ft=spring_force(v,n,torch.tensor(-.001),history,.0025,SpringHistorySettings(),.6)
    assert fn==0 and not ft.any()


@pytest.mark.parametrize('sliding',[False,True])
def test_contact_tangent_matches_autograd_in_smooth_active_regions(sliding):
    s=SpringHistorySettings();n=torch.tensor([0.,0,1.],dtype=torch.float64)
    xi=torch.tensor([.05 if sliding else .0001,0.,0.],dtype=torch.float64)
    v=torch.tensor([.02,-.01,.03],dtype=torch.float64)
    fn=lambda x:spring_force(x,n,torch.tensor(.003,dtype=x.dtype),xi,.0025,s,.6)[0]
    expected=torch.func.jacfwd(fn)(v)
    _,normal,tangent,j=spring_force(v,n,torch.tensor(.003,dtype=v.dtype),xi,.0025,s,.6,jacobian=True)
    torch.testing.assert_close(j,expected,atol=1e-9,rtol=1e-9)
    assert tangent.norm()<=.6*normal+1e-10


def test_patch_loss_zero_exactly_matches_coverage_and_required_fingers():
    torch.manual_seed(11);s=PatchAnchorSettings();g=torch.rand(200,5)*.004;o=torch.rand(200,5)*.009
    g[0]=.001;o[0]=.004;g[1]=.001;o[1]=.004;g[1,0]=.003
    groups=[torch.tensor([0]),torch.tensor([1,2,3]),torch.tensor([4])];fingers=torch.tensor([0,1])
    terms=patch_fit_terms(g,o,groups,fingers,s)
    fit=(g<=s.surface_tolerance_m+1e-8)&(o<=s.anchor_tolerance_m+1e-8)
    good=(fit.float().mean(-1)+1e-7>=.8)&fit[:,0]&fit[:,1:4].any(-1)
    assert torch.equal(terms['cost']==0,good)
    assert terms['cost'][0]==0 and terms['required'][1]>0
    # The palm can be the single omitted slot; it is not an extra finger gate.
    g[2]=.001;o[2]=.004;g[2,4]=.004
    assert patch_fit_terms(g[2:3],o[2:3],groups,fingers,s)['cost']==0


def test_feasible_candidate_retained_even_at_higher_scalar_cost():
    class Adapter:
        retain_feasible=True
        def evaluate_candidates(self,x,rows):return x.square().sum(-1),x[:,0]>.1
        def validate(self,x,rows):return x.square().sum(-1),dict(patch_joint_accepted=x[:,0]>.1)
    q=torch.zeros(1,1);p=Problem(q,torch.ones_like(q)*.5,None)
    r=solve_patch_cem(p,Adapter(),Settings(population=32,iterations=3),7,early_stop=False)
    assert r['passed'].item() and r['cost'].item()>0


def test_saved_mano_v5_input_keeps_original_shadow_anchor_source(tmp_path):
    import json,numpy as np
    from types import SimpleNamespace
    from scripts.solve_mano_comfree import packets
    source=tmp_path/'source';source.mkdir();shard=tmp_path/'shard';shard.mkdir()
    expected=np.arange(51,dtype=np.float32)[None];np.savez(source/'poses.npz',grasp_qpos=expected)
    np.savez(shard/'poses.npz',qpos=np.zeros((1,6),np.float32))
    np.savez(shard/'geometry.npz',test=np.zeros(1))
    (shard/'manifest.json').write_text(json.dumps(dict(cases=[dict(grasp_type='type',source_manifest=str(source/'manifest.json'),source_cached_case=0)])))
    args=SimpleNamespace(diagnostic_dir=None,input=str(shard),rows=None,initialization='saved-mano',contact_objective='patch_consistent_v5')
    result=list(packets(args));np.testing.assert_array_equal(result[0][1]['shadow_qpos'],expected)


def pinch(worlds=3,substeps=4):
    d='cuda:0';n=torch.tensor([[[1.,0,0],[-1.,0,0]]],device=d).expand(worlds,-1,-1).clone()
    p=.02*n;gap=torch.full((worlds,2),-.005,device=d);hand=p+gap[...,None]*n
    center=torch.zeros(worlds,3,device=d);r=torch.eye(3,device=d)[None].expand(worlds,-1,-1).clone()
    I=(torch.eye(3,device=d)*(.1*.04**2/6))[None].expand(worlds,-1,-1).clone()
    rollout=SpringHistoryRollout(worlds,2,settings=RolloutSettings(cost_mode='velocity',initial_angular_velocity=(0,0,0)),spring=SpringHistorySettings(substeps=substeps))
    return rollout,(hand,p,n,gap,center,r,I)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_pinch_support_history_reset_friction_limits_and_no_contact_reference():
    r,args=pinch();a=r.solve(*args);b=r.solve(*args)
    assert a['valid'].all() and a['stable'].all()
    assert a['tangential_displacement_m'].norm(dim=-1).max()>.0001
    assert (a['tangent_force_n'].norm(dim=-1)<=.6*a['normal_force_n']+1e-6).all()
    for k in ['linear_velocity','angular_velocity','tangential_displacement_m']:torch.testing.assert_close(a[k],b[k],atol=1e-7,rtol=1e-5)
    free=r.solve(*args,eligible=torch.zeros((3,2),device='cuda',dtype=torch.bool))
    assert free['valid'].all() and not free['stable'].any() and not free['tangential_displacement_m'].any()
    torch.testing.assert_close(free['velocity_cost'],free['free_velocity_cost'],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(free['linear_speed_m_s'],torch.full((3,),.4905,device='cuda'))
    torch.testing.assert_close(free['pose_cost'],free['free_pose_cost'],atol=1e-5,rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_physical_stiffness_survives_many_contacts_and_timestep_refinement():
    r,args=pinch(1,4);a=r.solve(*args)
    fine,_=pinch(1,8);b=fine.solve(*args)
    assert a['valid'].all() and b['valid'].all() and a['stable'].all() and b['stable'].all()
    assert (a['linear_speed_m_s']-b['linear_speed_m_s']).abs().max()<.02
    n=torch.cat((torch.eye(3,device='cuda'),-torch.eye(3,device='cuda')))[None]
    p=n*.02;gap=torch.full((1,6),-.005,device='cuda');hand=p+gap[...,None]*n
    many=SpringHistoryRollout(1,6,settings=RolloutSettings(cost_mode='velocity'),spring=SpringHistorySettings(normal_stiffness_n_m=5000,tangent_stiffness_n_m=5000))
    out=many.solve(hand,p,n,gap,*args[4:]);assert out['valid'].all() and out['linear_speed_m_s'].max()<1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_capture_replay_and_independent_worlds():
    r,args=pinch(2,2);r.solve(*args);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):r.solve(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=r.solve(*args)
    graph.replay();expected=r.solve(*args)
    torch.testing.assert_close(out['linear_velocity'],expected['linear_velocity'],atol=1e-7,rtol=1e-5)
    single,_=pinch(1,2);one=single.solve(*(v[:1] for v in args))
    torch.testing.assert_close(out['linear_velocity'][:1],one['linear_velocity'],atol=1e-7,rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_detachment_clears_existing_history_and_rotated_setup_is_equivariant():
    from scripts.calibrate_wuji_mano import rotmat
    r,args=pinch(1);hand,p,n,gap,center,rotation,inertia=args
    old=torch.zeros_like(hand);old[...,2]=.001
    state=r._step(torch.zeros_like(center),torch.zeros_like(center),center,rotation,p,n,p+.01*n,old,
        inertia,torch.linalg.inv(inertia),torch.ones_like(gap,dtype=torch.bool))
    assert state[5].all() and not state[4].any() and not state[6].any() and not state[7].any()
    a=r.solve(*args);Q=rotmat(torch.tensor([[.3,-.2,.7]],device='cuda'))
    rotate=lambda v:torch.einsum('bij,b...j->b...i',Q,v)
    gravity=rotate(center.new_tensor([[0.,0.,-9.81]]))[0].tolist()
    other=SpringHistoryRollout(1,2,settings=RolloutSettings(cost_mode='velocity',initial_angular_velocity=(0,0,0),gravity=tuple(gravity)))
    b=other.solve(rotate(hand),rotate(p),rotate(n),gap,rotate(center),Q,inertia)
    assert b['valid'].all()
    torch.testing.assert_close(b['linear_velocity'],rotate(a['linear_velocity']),atol=1e-6,rtol=1e-4)
    torch.testing.assert_close(b['tangential_displacement_m'],rotate(a['tangential_displacement_m']),atol=1e-7,rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_fused_matches_reference_with_sliding_detachment_and_variable_contact_count():
    torch.manual_seed(20261105)
    b,k=48,16
    n=torch.nn.functional.normalize(torch.randn(b,k,3,device='cuda'),dim=-1)
    # Off-centre, asymmetric contacts exercise all blocks of the Newton system.
    p=.02*n+torch.randn_like(n)*.004
    gap=torch.rand(b,k,device='cuda')*.008-.006
    hand=p+gap[...,None]*n
    center=torch.zeros(b,3,device='cuda')
    r=torch.eye(3,device='cuda')[None].repeat(b,1,1)
    inertia=torch.diag(torch.tensor([2e-5,3e-5,4e-5],device='cuda'))[None].repeat(b,1,1)
    eligible=torch.rand(b,k,device='cuda')>.4
    eligible[0]=False
    args=hand,p,n,gap,center,r,inertia
    reference=SpringHistoryRollout(b,k,backend='torch').solve(*args,eligible=eligible)
    fused=SpringHistoryRollout(b,k,backend='warp').solve(*args,eligible=eligible)
    assert torch.equal(reference['valid'],fused['valid'])
    assert torch.equal(reference['stable'],fused['stable'])
    for key in ['center_world','rotation','linear_velocity','angular_velocity','tangential_displacement_m',
                'normal_force_n','tangent_force_n','max_row_force_n','velocity_cost']:
        torch.testing.assert_close(fused[key],reference[key],atol=1e-5,rtol=2e-4,msg=lambda m:key+': '+m)
    assert (fused['newton_linear_solves']<160).all()
    assert fused['newton_linear_solves'][0]==0
