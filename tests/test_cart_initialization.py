"""Reference initialization is time-aligned and never replaces rollout feedback."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from moxian_web import cart_initialization as ik
from moxian_web.cart_training import windows,rollout


@pytest.fixture(autouse=True)
def one_thread():torch.set_num_threads(1)


def fake_recording(tmp_path,monkeypatch):
    folder=tmp_path/'recordings/clip';folder.mkdir(parents=True)
    for name in ('cart_preprocess.json','topology.json','profile.json'):(folder/name).write_text('{}')
    upstream=tmp_path/'upstream';upstream.mkdir();(upstream/'session.py').write_text('# fixed test solver')
    urdf=upstream/'hand.urdf';urdf.write_text('<robot/>')
    rows=[]
    for i,stamp in enumerate((0.,20.,40.,60.,80.,500.)):
        points=np.zeros((21,3));points[1,0]=.1*(i+1)
        rows.append(dict(timestamp_ms=stamp,stream_epoch=0,joints_m=points.tolist()))
    (folder/'cart_targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    names=[str(i) for i in range(20)];calls=[]
    class Session:
        def __init__(self):
            self.kinematics=SimpleNamespace(joint_names=names,urdf_path=urdf);self.reset()
        def reset(self):self._warm=None;self._filtered=None
        def step(self,points):
            raw=np.full(20,points[1,0],np.float32);self._warm=raw.astype(float)
            result=raw if self._filtered is None else self._filtered+.2*(raw-self._filtered)
            self._filtered=result.copy();self.last_trace=dict(stages=[dict(fallback=False)])
            calls.append(float(points[1,0]));return result
    monkeypatch.setattr(ik,'manifest',lambda root,key:(folder,dict(frames=len(rows),source=dict(hand_side='left'))))
    monkeypatch.setattr(ik,'REFERENCE_ROOT',upstream)
    monkeypatch.setattr(ik,'RobotAdapter',lambda:SimpleNamespace(joint_names=names,lower=-np.ones(20),upper=np.ones(20),spec={}))
    monkeypatch.setattr(ik,'RetargetSession',SimpleNamespace(for_hand=lambda model,side:Session()))
    return folder,rows,calls


def test_original_solver_cache_resume_and_gap_reset(tmp_path,monkeypatch):
    folder,rows,calls=fake_recording(tmp_path,monkeypatch)
    count=0
    def cancel():
        nonlocal count
        count+=1
        if count==4:raise InterruptedError('test interrupt')
    with pytest.raises(InterruptedError):ik.prepare_ik(tmp_path,'clip',check=cancel)
    assert len(calls)==3 and (folder/'cart_ik.pending.npz').exists()
    info=ik.prepare_ik(tmp_path,'clip')
    assert len(calls)==6 and info['initialization_only'] and not info['tube_required']
    assert info['source_side']=='left' and info['solver_side']=='right'
    with np.load(folder/'cart_ik.npz') as cache:
        np.testing.assert_allclose(cache['q'][:,0],[.1,.12,.156,.2048,.26384,.6],atol=1.e-7)
    assert ik.prepare_ik(tmp_path,'clip')==info and len(calls)==6
    sessions=[(dict(id='clip'),rows,{})];ik.attach_ik(tmp_path,sessions)
    assert all(r['ik_valid'] for r in rows) and rows[2]['ik_q_rad'][0]==pytest.approx(.156)
    # Input changes invalidate the sidecar, without rewriting any raw capture.
    with (folder/'cart_targets.jsonl').open('a') as stream:stream.write(' ')
    with pytest.raises(json.JSONDecodeError):ik.prepare_ik(tmp_path,'clip')


def test_window_uses_current_frame_ik_and_no_teacher_forcing():
    rows=[]
    for i in range(35):
        rows.append(dict(timestamp_ms=i*20.,stream_epoch=0,ik_valid=True,ik_q_rad=[i/100.]*20,
            emf_pose=np.zeros((5,9)).tolist(),emf_valid=[True]*5,tactile_positions_m=np.zeros((460,3)).tolist(),
            tactile_state=[0]*460,tactile_valid=[True]*460))
    batch=[windows([(dict(id='clip'),rows,{})],length=2)[-1]]
    offset=len(batch[0]['rows'])-12;index=batch[0]['indices'][offset]
    assert batch[0]['initial_q']==rows[index]['ik_q_rad'] and batch[0]['initial_q']!=rows[index+1]['ik_q_rad']
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.2));self.inputs=[]
        def forward(self,q,**inputs):self.inputs.append(q);return q*self.weight+.1
        def unnormalize(self,q):return q
    class Objective:
        collision=None
        hand=SimpleNamespace(device='cpu',tensor=lambda x:torch.tensor(x,dtype=torch.float32))
        def batch(self,q,rows,*args,**kwargs):return q.square().mean(),{},[]
    model=Model();loss,_,_,predictions=rollout(model,Objective(),batch)
    assert len(model.inputs)==2  # no neutral-pose prewarming calls
    torch.testing.assert_close(model.inputs[0],torch.tensor([rows[index]['ik_q_rad']]))
    torch.testing.assert_close(model.inputs[1],predictions[0])
    loss.backward();assert model.weight.grad.abs()>0


def test_uniform_reference_reset_noise_is_bounded_and_does_not_change_cached_q():
    hand=SimpleNamespace(low=torch.zeros(20),high=torch.ones(20),tensor=lambda q:torch.tensor(q,dtype=torch.float32))
    batch=[dict(initial_q=[.5]*20) for _ in range(50)];torch.manual_seed(8)
    changed=ik.perturb_initial_states(batch,hand,.1)
    values=torch.stack([w['initial_q'] for w in changed])
    assert values.min()>=.4 and values.max()<=.6 and values.std()>.04
    assert batch[0]['initial_q']==[.5]*20
    assert ik.perturb_initial_states(batch,hand,0) is batch
    with pytest.raises(ValueError):ik.perturb_initial_states(batch,hand,float('nan'))


def test_one_reset_noise_sample_is_reused_for_collection_and_policy(monkeypatch):
    from moxian_web import cart_training as training
    rows=[dict(timestamp_ms=i*20.,stream_epoch=0,contacts=dict(self_pairs=[],object_contacts=[])) for i in range(12)]
    batch=[dict(rows=rows,length=2,initial_q=[.5]*20) for _ in range(5)]
    hand=SimpleNamespace(device='cpu',low=torch.zeros(20),high=torch.ones(20),tensor=lambda x:torch.tensor(x,dtype=torch.float32))
    model=torch.nn.Linear(1,1);collected=[];policy=[]
    objective=SimpleNamespace(hand=hand,collision=SimpleNamespace(model=torch.nn.Linear(1,1)))
    def probe(model,objective,chunk,collect_only=False,**kwargs):
        (collected if collect_only else policy).extend(w['initial_q'].clone() for w in chunk)
        return model.weight.square().mean(),{},[],[]
    monkeypatch.setattr(training,'rollout',probe)
    learner=SimpleNamespace(collect=lambda:None,update=lambda:{})
    training.policy_update(model,objective,learner,torch.optim.SGD(model.parameters(),lr=0.),batch,2,initial_noise_rad=.1)
    torch.testing.assert_close(torch.stack(collected),torch.stack(policy))
    assert not torch.equal(collected[0],collected[1])
    assert batch[0]['initial_q']==[.5]*20
