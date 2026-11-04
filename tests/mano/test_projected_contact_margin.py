import numpy as np
import pytest
import torch

from retarget.env.grasp.contact_margin import geom_contact_margins
from retarget.env.grasp.closure_types import ClosureSettings
from retarget.env.grasp.projected_qp import ProjectedQPClosure, project_pressures


def solver(graph=False):
    return ProjectedQPClosure(ClosureSettings(cuda_graph=graph),center=[0.,0.,0.],radius=1.,object_mass=.1)


def tetra(device='cpu',dtype=torch.float64):
    return torch.nn.functional.normalize(torch.tensor(
        [[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],device=device,dtype=dtype),dim=-1)


def test_margin_is_max_gap_is_max_and_boundary_is_strict():
    d=torch.tensor([[-.003,0.,.0007,.0008,.001,.0015]],dtype=torch.float64)
    s=geom_contact_margins(d,.001,.0006,hand_gap=.0002,object_gap=.0001)
    torch.testing.assert_close(s.margin,torch.full_like(d,.001))
    torch.testing.assert_close(s.gap,torch.full_like(d,.0002))
    assert s.detected.tolist()==[[True,True,True,True,False,False]]
    assert s.active.tolist()==[[True,True,True,False,False,False]]
    # Penetration acceptance is checked separately, not by force closure.
    explicit=geom_contact_margins(d,.001,.0006,pair_margin=.002,pair_gap=0.)
    assert explicit.active.all()


def test_matches_native_333_contact_includemargin_and_active_set():
    import mujoco
    if mujoco.__version__!='3.3.3':pytest.skip('Pinned CPU 3.3.3 reference required')
    for distance in (-.0005,.0005,.0009,.0015):
        model=mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0"/>
          <worldbody><geom type="sphere" size=".01" margin=".001" gap=".0002"/>
          <body pos="{.02+distance} 0 0"><freejoint/>
          <geom type="sphere" size=".01" margin=".0006" gap=".0001"/></body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        selected=geom_contact_margins(torch.tensor([distance],dtype=torch.float64),.001,.0006,
                                      hand_gap=.0002,object_gap=.0001)
        assert bool(data.ncon)==selected.detected.item()
        if data.ncon:
            np.testing.assert_allclose(data.contact[0].includemargin,.0008,atol=1e-12)
            assert (data.contact[0].efc_address>=0)==selected.active.item()


def test_masked_qp_equals_removing_inactive_slots():
    torch.manual_seed(29)
    p=torch.randn((3,7,3),dtype=torch.float64)
    n=torch.nn.functional.normalize(torch.randn_like(p),dim=-1)
    mask=torch.tensor([[1,0,1,0,1,1,0],[0,1,1,1,0,0,0],[0,0,0,0,0,0,0]],dtype=torch.bool)
    s=solver();actual=s.solve(p,n,contact_mask=mask)
    for i in range(2):
        expected=s.solve(p[i:i+1,mask[i]],n[i:i+1,mask[i]])
        torch.testing.assert_close(actual.cost[i],expected.cost[0],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(actual.row_forces[i][:,mask[i]],expected.row_forces[0],atol=1e-12,rtol=1e-12)
        assert actual.certified[i]==expected.certified[0]
    assert not actual.valid[2] and not actual.certified[2]
    assert (actual.row_forces.masked_select(~mask[:,None,:,None])==0).all()


def test_geometric_margin_does_not_move_wrench_points_or_relax_certificate():
    p=tetra()[None];s=solver();original=p.clone()
    baseline=s.solve(p,p)
    # 1.5 mm is inside the sum of margins, but outside the native maximum.
    result,selection=s.solve_with_geom_margins(p,p,signed_distance=torch.full((1,4),.0015,dtype=p.dtype),
                                             hand_margin=.001,object_margin=.001)
    assert baseline.certified.item() and not result.certified.item()
    assert not selection.active.any()
    included,_=s.solve_with_geom_margins(p,p,signed_distance=torch.zeros((1,4),dtype=p.dtype),
                                       hand_margin=.001,object_margin=.001)
    torch.testing.assert_close(included.cost,baseline.cost,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(included.row_forces,baseline.row_forces,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(p,original)
    assert s.settings.certificate_tolerance==.05
    invalid,_=s.solve_with_geom_margins(p,p,signed_distance=torch.tensor([[0.,0.,0.,float('nan')]]),
                                      hand_margin=.001,object_margin=.001)
    assert not invalid.valid.item() and not invalid.certified.item()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_gpu_masked_projection_and_graph_replay():
    torch.manual_seed(47)
    mask=torch.tensor([[1,0,1,0,1],[0,0,0,0,0],[0,1,0,1,1]],dtype=torch.bool)
    value=torch.randn(3,13,5,4)*4-3
    caps=mask[:,None].expand(3,13,5).float()*10.;minimum=torch.ones((3,13))
    expected=project_pressures(value,caps,minimum,mask)
    actual=project_pressures(value.cuda(),caps.cuda(),minimum.cuda(),mask.cuda()).cpu()
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    s=solver(graph=True);p=tetra('cuda',torch.float32)[None].expand(3,-1,-1).contiguous()
    distance=torch.zeros((3,4),device='cuda')
    for selected in (torch.ones((3,4),device='cuda',dtype=torch.bool),mask[:,:4].cuda()):
        distance.copy_(torch.where(selected,0.,.0015))
        result,selection=s.solve_with_geom_margins(p,p,signed_distance=distance,hand_margin=.001,object_margin=.001)
        eager=solver().solve(p,p,contact_mask=selected)
        torch.testing.assert_close(result.cost,eager.cost,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(result.row_forces,eager.row_forces,atol=2e-5,rtol=2e-5)
        assert torch.equal(selection.active,selected)
        assert torch.equal(result.certified,eager.certified)
        assert result.row_forces.is_cuda and selection.margin.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_native_adapter_uses_current_geom_values_and_explicit_pairs():
    from types import SimpleNamespace
    import mujoco
    if tuple(map(int,mujoco.__version__.split('.')[:2]))<(3,17):
        pytest.skip('Native adapter requires the isolated MuJoCo/MJWarp 3.17 runtime')
    import warp as wp
    from retarget.env.grasp.native_mjwarp import NativePhysics
    physics=NativePhysics.__new__(NativePhysics)
    physics.device=torch.device('cuda:0');physics.worlds=3
    physics.cpu=SimpleNamespace(ngeom=3)
    physics.model=SimpleNamespace(
        geom_margin=wp.array([[.001,.0006,.0004]],dtype=wp.float32,device='cuda:0'),
        geom_gap=wp.array([[0.,0.,0.]],dtype=wp.float32,device='cuda:0'))
    physics.object_geom_mask=torch.tensor([True,False,False],device='cuda')
    physics.pair_margin=torch.full((3,3),float('nan'),device='cuda')
    physics.pair_gap=physics.pair_margin.clone()
    physics.pair_margin[1,0]=.002;physics.pair_gap[1,0]=0.
    points=tetra('cuda',torch.float32)[None].expand(3,-1,-1)
    hand=torch.tensor([[1]*4,[2]*4,[-1]*4],device='cuda')
    distance=torch.full((3,4),.0015,device='cuda')
    s=solver();result,selected=physics.projected_closure(s,points,points,distance,hand,0)
    assert selected.active.tolist()==[[True]*4,[False]*4,[False]*4]
    assert result.certified.tolist()==[True,False,False]
    # A later model margin change must be observed, without cached selection.
    wp.to_torch(physics.model.geom_margin)[0,0]=.003
    updated,_=physics.projected_closure(s,points,points,distance,hand,0)
    assert updated.certified.tolist()==[True,True,False]
