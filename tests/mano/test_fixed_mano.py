import copy,json,threading
from pathlib import Path
import numpy as np
import pytest
import torch
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.mano_runtime import ManoSurface
from moxian_web.mano_shape_prior import ShapePrior,GEOMETRY_UPPER,project,BETA_RADIUS
from moxian_web.calibration_targets import DEFINITION,pinch_patches

@pytest.mark.parametrize('side',['left','right'])
def test_canonical_prior_rejects_flat_fingers_and_has_finite_gradients(side):
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not path.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(path,device='cpu',side=side);prior=ShapePrior(m)
    beta=torch.zeros(10,requires_grad=True);geometry=torch.zeros(43,requires_grad=True)
    assert prior.report(beta,geometry)['accepted']
    thin=geometry.detach().clone();thin[20:25]=np.log(.625)
    assert not prior.report(beta,thin)['accepted']
    loss=sum(r.square().mean() for r in prior.residuals(beta+.01,geometry+.001).values())
    gradients=torch.autograd.grad(loss,(beta,geometry));assert all(torch.isfinite(g).all() for g in gradients)
    vector=torch.full((89,),2.);bound=torch.tensor(GEOMETRY_UPPER,dtype=torch.float32)
    low=torch.cat((torch.full((46,),-3.),-bound));high=-low
    actual=project(vector,low,high,89)
    assert actual[:10].norm()<=BETA_RADIUS+1e-6 and torch.all(actual[46:]<=bound)

def test_contact_contract_covers_collection_and_only_pinches_request_equalities():
    from moxian_web.motion_capture import STAGES
    for stage in STAGES:assert stage['id'] in DEFINITION['stages']
    for finger in ('index','middle','ring','pinky'):
        from moxian_web.contact import PATCH_IDS
        assert pinch_patches('pinch_'+finger)==(PATCH_IDS.index('thumb_distal'),PATCH_IDS.index(finger+'_distal'))
    assert not pinch_patches('thumb_opposition') and not pinch_patches('range_index')

@pytest.mark.parametrize('side',['left','right'])
def test_fixed_training_forward_matches_deployment_without_frame_variables(side):
    from moxian_web.mano_objective import ManoObjective
    path=DEFAULT_MANO.with_name('MANO_'+side.upper()+'.pkl')
    if not path.exists():pytest.skip('Licensed MANO required')
    m=ManoSurface(path,device='cpu',side=side)
    q=np.random.default_rng(7).normal(0,.1,(2,15,3));roots=np.tile(np.eye(3),(2,1,1))
    rows=[dict(stage='range_index',timestamp_ms=i,capture_protocol='geort_motion_v2') for i in range(2)]
    o=ManoObjective(m,rows,q,roots,shared=True,training=True,tactile=False,fixed_forward=True)
    assert len(o.initial)==o.shared_size==89 and not o.free.any()
    x=o.initial.detach().clone();x[:10]=.05;x[46:71]=.01
    _,v,j,_,_=o.surface(x);o.install(x)
    with torch.no_grad():vv,jj,_,_=m.decode(torch.tensor(q,dtype=torch.float32),torch.tensor(roots,dtype=torch.float32))
    torch.testing.assert_close(v,vv,atol=2e-7,rtol=1e-5);torch.testing.assert_close(j,jj,atol=2e-7,rtol=1e-5)

def test_display_forward_never_calls_optimizer_and_touch_cannot_move_the_hand(monkeypatch):
    source=Path(__file__).resolve().parents[2]/'output/moxian/hand_model/348640923d9e2f3f/fit-wrist-targets.json'
    if not source.exists():pytest.skip('Retained same-frame IK required')
    row=next(r for r in json.loads(source.read_text())['rows'] if r['stage']=='open')
    m=ManoSurface(DEFAULT_MANO.with_name('MANO_RIGHT.pkl'),device='cpu',side='right');m.enable_official_fk()
    m.solver_enabled=False;m.contact_pose_enabled=False;m.set_glove_geometry(np.zeros(43));m.set_shape(np.full(10,.03))
    m.set_rest_vertex_residual(np.full((778,3),.00001))
    def forbidden(*a,**kw):raise AssertionError('The display must not optimize')
    monkeypatch.setattr(torch.optim,'Adam',forbidden);monkeypatch.setattr(torch.optim,'SGD',forbidden)
    before=(m.shape.copy(),m.glove_parameters.copy(),m.mount_pos.copy(),m.mount_rot.copy(),m.model.rest_vertex_residual.clone())
    a=m.track(copy.deepcopy(row));changed=copy.deepcopy(row)
    changed.update(contact={'ready':True,'cell_contact':[True]*460},stage='pinch_pinky',guided_contact=True)
    b=m.track(changed)
    for aa,bb in zip(a[:4],b[:4]):np.testing.assert_array_equal(aa,bb)
    for aa,bb in zip(before,(m.shape,m.glove_parameters,m.mount_pos,m.mount_rot,m.model.rest_vertex_residual)):np.testing.assert_array_equal(aa,bb)
    assert m.solver_report['optimizer_steps']==0 and not m.contact_correction['applied']

def test_repeated_ema_updates_cannot_flatten_the_exported_geometry():
    m=ManoSurface(DEFAULT_MANO,device='cpu');prior=ShapePrior(m)
    average=torch.zeros(89);proposal=average.clone();proposal[66:71]=-2.;proposal[43]=1.
    for _ in range(100):average=prior.ema_update(average,proposal)
    report=prior.report(average[:10],average[46:89])
    assert report['accepted'] and report['minimum_section_ratio']>=.8
    assert average[43]>.99  # A rejected skin update does not block alignment learning.
