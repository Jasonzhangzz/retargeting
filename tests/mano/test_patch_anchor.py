"""Patch witnesses, independent finger penalties and per-pose CEM stopping."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import warp as wp

from retarget.env.grasp.mano_patch_anchor import (PatchAnchorSettings,
    patch_witness_kernel,patch_penetration_terms)
from retarget.solver.base import Problem
from retarget.solver.cem import SamplingSettings as Settings
from retarget.solver.cem import _solve_patch_cem as solve_patch_cem


def kernel_witness(hand,object_vertices,object_faces,anchor,device='cpu',scale=1.,backend='reference'):
    wp.init()
    mesh=wp.Mesh(points=wp.array(object_vertices,dtype=wp.vec3,device=device),
                 indices=wp.array(np.asarray(object_faces).ravel(),dtype=int,device=device))
    h=wp.zeros((1,1,1),dtype=wp.vec3,device=device);o=wp.zeros_like(h)
    scores=wp.zeros((1,1,1),dtype=float,device=device)
    if backend=='optimized':
        from retarget.env.grasp.patch_query import patch_witness_compact
        kernel=patch_witness_compact;dims=(1,1)
        triangle_inputs=[wp.array([[0,1,2]],dtype=int,device=device),wp.array(np.eye(3)[None],dtype=wp.mat33,device=device),
                        wp.array([[0,0]],dtype=wp.vec2i,device=device)]
    else:
        kernel=patch_witness_kernel;dims=(1,1,1)
        triangle_inputs=[wp.array([[[0,1,2]]],dtype=int,device=device),wp.array(np.eye(3)[None,None],dtype=wp.mat33,device=device),wp.array([1],dtype=int,device=device)]
    wp.launch(kernel,dim=dims,inputs=[wp.array([mesh.id],dtype=wp.uint64,device=device),
        wp.array([0],dtype=int,device=device),wp.array(np.asarray(hand)[None],dtype=wp.vec3,device=device),
        *triangle_inputs,
        wp.array([[anchor]],dtype=wp.vec3,device=device),wp.array([scale],dtype=float,device=device),.002,.005],
        outputs=[h,o,scores],device=device)
    return h.numpy()[0,0,0],o.numpy()[0,0,0],scores.numpy()[0,0,0]


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_optimized_witness_preserves_crossings_edges_degenerate_and_thresholds(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    obj=[[-.01,-.01,0],[.01,-.01,0],[0,.01,0]]
    cases=[([[-.01,0,-.01],[.01,0,-.01],[0,0,.01]],[0,0,0]),
           ([[.03,0,0],[.04,0,0],[.03,.01,0]],[0,0,0]),
           ([[0,0,0],[0,0,0],[.01,0,0]],[.004,0,0]),
           (np.asarray(obj)+[0,0,.001],[.04,0,0])]
    for gap in [.002-1e-7,.002,.002+1e-7]:
        for offset in [.005-1e-7,.005,.005+1e-7]:
            cases.append((np.asarray(obj)+[0,0,gap],[offset,0,0]))
    for hand,anchor in cases:
        a=kernel_witness(hand,obj,[[0,1,2]],anchor,device)
        b=kernel_witness(hand,obj,[[0,1,2]],anchor,device,backend='optimized')
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-7,rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_depth_only_query_matches_full_signed_query_on_mixed_objects_and_invalid_points():
    import trimesh
    from retarget.env.grasp.mano_solver import SurfaceBatch
    from retarget.env.grasp.triangle_penetration import TrianglePenetration
    meshes=[trimesh.creation.box([.04,.03,.02]),trimesh.creation.icosphere(subdivisions=2,radius=.025)]
    geo={i:dict(vertices=m.vertices,faces=m.faces,watertight=True) for i,m in enumerate(meshes)}
    poses=np.array([[.1,0,0,1,0,0,0],[-.1,.2,0,.9238795,0,0,.3826834]],np.float32)
    s=SurfaceBatch(geo,[0,1],poses,[.3,1.4],'cuda:0')
    torch.manual_seed(14);local=(torch.rand(2,1000,3,device='cuda')-.5)*.15
    local[:,0]=0.;local[0,1]=torch.tensor([.02,.015,.01],device='cuda')
    world=torch.einsum('bij,bkj->bki',s.rotation,local*s.scale[:,None,None])+s.origin[:,None]
    rows=torch.arange(2,device='cuda');q=TrianglePenetration(s)
    projected,_,distance,_,_=s.query(world,rows);depth,valid=q.query(world,rows)
    assert valid.all() and torch.isfinite(projected).all()
    torch.testing.assert_close(depth,(-distance).clamp_min(0).amax(-1),atol=1e-7,rtol=1e-5)
    bad=world.clone();bad[0,0]=torch.nan;bad[1,0]=1e8
    projected,_,_,_,_=s.query(bad,rows);_,valid=q.query(bad,rows)
    assert torch.equal(valid,torch.isfinite(projected).all((-1,-2))) and not valid.any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_parallel_pair_overflow_runs_exact_gpu_search():
    import trimesh
    from retarget.env.grasp.patch_query import parallel_query
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=.01);device='cuda:0'
    native=wp.Mesh(points=wp.array(mesh.vertices,dtype=wp.vec3,device=device),indices=wp.array(mesh.faces.ravel(),dtype=int,device=device))
    hand=np.array([[-.02,-.02,0],[.02,-.02,0],[0,.02,0]],np.float32);anchor=[0,0,.01]
    expected=kernel_witness(hand,mesh.vertices,mesh.faces,anchor,device)
    patches=SimpleNamespace(surfaces=SimpleNamespace(mesh_ids=wp.array([native.id],dtype=wp.uint64,device=device)),
        packed_slots=torch.tensor([[0,0]],device=device,dtype=torch.int32),packed_triangles=torch.tensor([[0,1,2]],device=device,dtype=torch.int32),
        packed_clip=torch.eye(3,device=device)[None],settings=PatchAnchorSettings())
    h=torch.empty(1,1,1,3,device=device);o=torch.empty_like(h);score=torch.empty(1,1,1,device=device)
    overflow=parallel_query(patches,torch.tensor(hand,device=device)[None],torch.tensor([[anchor]],device=device),
        torch.zeros(1,device=device,dtype=torch.long),torch.ones(1,device=device),h,o,score,capacity=1)
    assert overflow.item()>0
    np.testing.assert_allclose(float(score[0,0,0]),expected[2],atol=1e-5,rtol=1e-6)
    np.testing.assert_allclose(h[0,0,0].cpu(),expected[0],atol=1e-7,rtol=1e-5)
    np.testing.assert_allclose(o[0,0,0].cpu(),expected[1],atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_triangle_interior_crossing_and_anchor_neighborhood(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    # Crossing triangles have no coincident vertices. A vertex-only test fails.
    hand=[[-.01,0,-.01],[.01,0,-.01],[0,0,.01]]
    obj=[[-.02,-.02,0],[.02,-.02,0],[0,.02,0]]
    h,o,s=kernel_witness(hand,obj,[[0,1,2]],[0,0,0],device)
    np.testing.assert_allclose(h,o,atol=2.e-7);assert np.linalg.norm(o)<1.e-6;assert s<1000
    _,_,far=kernel_witness(hand,obj,[[0,1,2]],[.1,0,0],device)
    assert far>=1000


def test_parallel_faces_surface_tolerance_and_scaling():
    obj=[[-.01,-.01,0],[.01,-.01,0],[0,.01,0]]
    hand=np.asarray(obj)+[0,0,.001]
    h,o,s=kernel_witness(hand,obj,[[0,1,2]],[0,0,0])
    assert np.isclose(np.linalg.norm(h-o),.001,atol=1.e-7) and s<1000
    _,_,s=kernel_witness(hand+np.array([0,0,.002]),obj,[[0,1,2]],[0,0,0])
    assert s>=1000
    h,o,s=kernel_witness(hand/2,np.asarray(obj)/2,[[0,1,2]],[0,0,0],scale=2)
    assert np.isclose(np.linalg.norm(h-o)*2,.001,atol=1.e-7) and s<1000


def test_both_penetration_allowances_affect_cost_not_just_acceptance():
    from retarget.env.grasp.contact_geometry import PoseObjectiveSettings
    settings=replace(PoseObjectiveSettings(),hand_object_allowance_m=.005,penetration_tolerance_m=.005)
    depth=torch.tensor([[.005,.005],[.006,.005],[.005,.006],[.003,.004]])
    strict,soft,_=patch_penetration_terms(depth,settings,PatchAnchorSettings())
    torch.testing.assert_close(strict,torch.tensor([0.,1.,1.,0.]),atol=2.e-6,rtol=2.e-6)
    torch.testing.assert_close(soft,torch.tensor([1.,1.44,1.,.36]),atol=2.e-6,rtol=2.e-6)


def test_cem_stop_requires_joint_acceptance_and_actually_skips_finished_rows():
    q=torch.zeros(2,2);seen=[]
    class Adapter:
        def evaluate(self,x,rows):
            seen.extend(rows.tolist());return x.square().sum(-1)
        def validate(self,x,rows):
            cost=self.evaluate(x,rows)
            # First pose has good rollout but misses patch; it must keep going.
            good=rows==1
            return cost,dict(patch_joint_accepted=good,rollout_stable=torch.ones(len(rows),dtype=torch.bool))
    adapter=Adapter();problem=Problem(q,torch.ones_like(q)*.1,None)
    result=solve_patch_cem(problem,adapter,Settings(population=8,iterations=3),7)
    assert result['iterations'].tolist()==[3,0]
    assert result['early_stopped'].tolist()==[False,True]
    assert result['evaluations'].tolist()==[26,2]
    assert seen.count(1)==2  # Initial/final checks only; no population evaluation.
    assert result['passed'].tolist()==[False,True]


def test_generic_cem_entry_uses_the_same_current_update_without_grasp_gate():
    from retarget.solver.cem import _solve_generic as solve
    q=torch.tensor([[1.,-.5,.3],[-.7,.4,.5]])
    class Adapter:
        def evaluate(self,x,rows):return x.square().sum(-1)
        def validate(self,x,rows):return self.evaluate(x,rows),dict(patch_joint_accepted=torch.zeros(len(rows),dtype=torch.bool))
    adapter=Adapter()
    problem=Problem(q,torch.ones_like(q)*.3,lambda v:v.square().sum(-1),lambda v:v.clamp(-2,2))
    settings=Settings(population=16,iterations=5)
    original=solve(problem,settings,19)
    result=solve_patch_cem(problem,adapter,settings,19,early_stop=False)
    torch.testing.assert_close(result['parameters'],original.parameters,atol=2.e-7,rtol=2.e-6)
    torch.testing.assert_close(result['cost'],original.cost,atol=2.e-7,rtol=2.e-6)


def test_pose_stops_on_first_new_accepted_best():
    seen=[]
    class Adapter:
        def evaluate(self,x,rows):seen.append(len(x));return x.square().sum(-1)
        def evaluate_candidates(self,x,rows):return self.evaluate(x,rows),torch.ones(len(rows),dtype=torch.bool)
        def validate(self,x,rows):return self.evaluate(x,rows),dict(patch_joint_accepted=(x[:,0]<.9))
    problem=Problem(torch.ones(1,1),torch.ones(1,1)*.3,None)
    result=solve_patch_cem(problem,Adapter(),Settings(population=16,iterations=8),19)
    assert result['iterations'].item()==1 and result['early_stopped'].item()
    assert seen==[1,16,1,1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_real_graph_patch_objective_and_surface_witnesses(monkeypatch):
    import json
    from pathlib import Path
    from retarget.env.grasp.mano_solver import ManoDecoder,ShadowSkeleton,FixedContacts,SurfaceBatch
    from retarget.env.grasp.mano_patch_consistent import PatchConsistentAdapter
    from retarget.env.geometry import closest_triangles
    root=Path(__file__).resolve().parents[2]
    model=Path('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_RIGHT.pkl')
    frozen=root/'output/mano_patch_velocity_v4_20261105/inputs/01_1_Large_Diameter_random.npz'
    if not model.exists() or not frozen.exists():pytest.skip('Real assets unavailable')
    a=np.load(frozen);mapping=json.loads((root/'output/mano_patch_velocity_v4_20261105/assets/fixed_contact_mapping.json').read_text())['mappings']
    decoder=ManoDecoder(model,'cuda:0');decoder.configure_contact_vertices(mapping)
    records=mapping['1_Large_Diameter'];fixed=FixedContacts(decoder,records)
    skeleton=ShadowSkeleton(root/'output/mano_patch_velocity_v4_20261105/assets/shadow_assets/right.xml','cuda:0')
    indices=list(map(int,a['object_geometry_ids']));geo={i:dict(vertices=a[f'object_{i}_vertices'],faces=a[f'object_{i}_faces'],watertight=True) for i in set(indices)}
    surface=SurfaceBatch(geo,indices,a['object_pose_world'],a['scene_scale'],'cuda:0')
    rows=torch.tensor([0],device='cuda:0');q=torch.tensor(a['variables'],device='cuda:0',dtype=torch.float32)
    anchors=surface.query(skeleton.contacts(a['shadow_qpos'],records)[0],rows)[0]
    consistent=True
    adapter=PatchConsistentAdapter(decoder,fixed,surface,8,anchors=anchors)
    weights=decoder.base.weights.cpu().numpy();rest=decoder.rest.cpu().numpy()
    for rec,ids in zip(records,adapter.patches.face_ids):
        region=weights[decoder.base.faces[ids]].mean(1).argmax(-1)
        assert (region==region[0]).all()
    for k,ids in enumerate(adapter.patches.clipped_face_ids):
        bary=adapter.patches.clip[k,:len(ids)].cpu().numpy()
        points=np.einsum('tij,tjk->tik',bary,rest[decoder.base.faces[ids]])
        anchor=np.asarray(records[k]['mano_barycentric'])@rest[decoder.base.faces[records[k]['mano_face_id']]]
        assert np.linalg.norm(points-anchor,axis=-1).max()<=.01000001
    cost,parts=adapter.validate(q,rows)
    if consistent:
        assert parts['valid'].all()
        assert torch.equal(parts['patch_fit_cost']==0,
            (parts['patch_coverage']+1e-7>=.8)&parts['patch_all_required_fingers'])
        # Capture and eager execution must include identical history resets.
        adapter.cuda_graph=False
        original=ManoDecoder.__call__
        decodes=[]
        def count_decode(self,*args,**kwargs):
            decodes.append(1)
            return original(self,*args,**kwargs)
        with monkeypatch.context() as guard:
            guard.setattr(ManoDecoder,'__call__',count_decode)
            torch.testing.assert_close(adapter.evaluate(q,rows),cost,atol=1e-4,rtol=1e-4)
        assert len(decodes)==1
        adapter.cuda_graph=True
    expected=parts['anchor_cost']+1.e4*parts['penetration']+.1*parts['soft_penetration']+parts['rollout_cost']
    torch.testing.assert_close(cost,expected)
    # Actual barycentric membership of the emitted witness in the chosen skin
    # triangle and object surface. Not a convex-hull/vertex surrogate.
    v=decoder(q,full=True)[0][0].detach().cpu().numpy()
    for i in range(len(records)):
        face=int(parts['patch_face_ids'][0,i]);p=parts['patch_hand_points'][0,i].cpu().numpy()
        projected=closest_triangles(p[None],v[decoder.base.faces[[face]]])[0,0]
        np.testing.assert_allclose(p,projected,atol=5.e-7)
    _,_,gap,_,_=surface.query(parts['patch_object_points'],rows)
    assert float(gap.abs().max())<5.e-7
    # No Tensor.cpu() or NumPy pose operations are needed by score/acceptance.
    def forbidden(*a,**kw):pytest.fail('Pose chain left CUDA')
    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor,'cpu',forbidden)
        adapter.evaluate_candidates(q,rows);adapter.validate(q,rows)
    # Forward-mode MANO derivatives used by the kinematic LM match differences.
    def points(x):return fixed.evaluate(decoder(x[None])[0])[0][0]
    j=torch.func.jacfwd(points)(q[0])
    for dim,h in [(0,1.e-4),(3,1.e-3),(6,1.e-2),(20,1.e-2)]:
        step=torch.zeros_like(q[0]);step[dim]=h
        fd=(points(q[0]+step)-points(q[0]-step))/(2*h)
        torch.testing.assert_close(j[:,:,dim],fd,atol=5.e-5,rtol=3.e-3)
