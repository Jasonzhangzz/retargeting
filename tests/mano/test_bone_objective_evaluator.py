"""Captured objective execution preserves float64 loss and derivatives."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from moxian_web.bone_objective_evaluator import ObjectiveEvaluator


class Quadratic:
    def __init__(self,device):self.model=SimpleNamespace(device=torch.device(device))
    def loss(self,x):
        terms=dict(square=(x*x).sum(),nonlinear=torch.sin(x).sum())
        return sum(terms.values()),terms


@pytest.mark.parametrize('backend',['eager','auto'])
def test_cpu_loss_gradient_and_repeated_inputs(backend):
    x=np.array([.2,-.3,.7]);engine=ObjectiveEvaluator(Quadratic('cpu'),x,backend)
    for values in (x,x+1,x):
        loss,gradient,terms=engine(values)
        np.testing.assert_allclose(gradient,2*values+np.cos(values),rtol=1e-14)
        assert loss==pytest.approx(np.sum(values**2+np.sin(values)),rel=1e-14)
        assert sum(terms.values())==pytest.approx(loss)
    assert engine.provenance()['dtype']=='float64'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph execution')
def test_cuda_graph_is_replayable_and_does_not_accumulate_gradients():
    x=np.array([.2,-.3,.7]);eager=ObjectiveEvaluator(Quadratic('cuda:0'),x,'eager')
    graph=ObjectiveEvaluator(Quadratic('cuda:0'),x,'cuda_graph')
    for values in (x,x+1,x):
        a,da,ta=eager(values);b,db,tb=graph(values)
        assert a==pytest.approx(b,rel=1e-14)
        np.testing.assert_allclose(da,db,rtol=1e-14)
        assert ta==tb
    assert graph.backend=='cuda_graph'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph execution')
def test_capture_in_fitting_thread_with_concurrent_display_work():
    from concurrent.futures import ThreadPoolExecutor
    x=np.arange(256,dtype=float)/100
    display=torch.ones(1024,device='cuda:0')
    def build():return ObjectiveEvaluator(Quadratic('cuda:0'),x,'cuda_graph')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(build)
        while not future.done():
            display.add_(.001)
            # The mesh thread also returns GPU results to the CPU.
            display[:1].cpu()
        engine=future.result()
    _,gradient,_=engine(x)
    np.testing.assert_allclose(gradient,2*x+np.cos(x),rtol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph execution')
@pytest.mark.parametrize('side',['left','right'])
def test_real_contact_and_flat_objective_graph_matches_eager(side):
    from moxian_web.qnbot_surface import QnbotSurface
    from moxian_web.bone_calibration import Objective
    from moxian_web.bone_extrinsics import FINGERS
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(1)
    m=QnbotSurface(side,'cuda:0').double()
    q=m.q0[None].expand(2,-1).clone()+.1
    with torch.no_grad():p,r=m.sensor_prediction(q)
    rows=[]
    for i in range(2):
        rows.append(dict(stage='contact_pair_index' if i==0 else 'open',static_endpoint='target' if i==0 else 'flat',
            contact_confirmed=i==0,observation_weights=[1]*5,
            skeleton=dict(joint_names=m.joint_names,joint_positions_rad=q[i].cpu().tolist()),
            fingers={f:dict(position=p[i,j].cpu().tolist(),quaternion_xyzw=Rotation.from_matrix(r[i,j].cpu().numpy()).as_quat().tolist()) for j,f in enumerate(FINGERS)}))
    contacts={'contact_pair_index':[dict(anchors=[m.patch_anchors['thumb_distal'],m.patch_anchors['index_distal']],required=True)]}
    obj=Objective(m,rows,contacts,dict(position_sigma_m=.003,rotation_sigma_rad=.15))
    x=np.r_[np.zeros(51),q.cpu().numpy().ravel()]
    eager=ObjectiveEvaluator(obj,x,'eager');graph=ObjectiveEvaluator(obj,x,'cuda_graph')
    rng=np.random.default_rng(7)
    for values in (x,x+rng.normal(0,.002,len(x)),x):
        a,da,ta=eager(values);b,db,tb=graph(values)
        assert b==pytest.approx(a,rel=1e-10,abs=1e-8)
        np.testing.assert_allclose(db,da,rtol=1e-8,atol=1e-7)
        np.testing.assert_allclose(list(tb.values()),list(ta.values()),rtol=1e-10,atol=1e-8)
    # Independent path: separate sensor FK and full-mesh skinning, rather
    # than shared FK plus the optimized sparse affine blend.
    from unittest.mock import patch
    sensor=m.sensor_prediction;forward=m.forward
    def separate_sensor(q,geometry=None,**kwargs):return sensor(q,geometry)
    def full_mesh(q,vertex_ids=None,geometry=None,**kwargs):
        v,j,r=forward(q,geometry=geometry)
        return (v if vertex_ids is None else v[:,vertex_ids]),j,r
    values=x+rng.normal(0,.005,len(x));fast,df,tf=graph(values)
    with patch.object(m,'sensor_prediction',side_effect=separate_sensor),patch.object(m,'forward',side_effect=full_mesh):
        slow,ds,ts=eager(values)
    assert fast==pytest.approx(slow,rel=1e-10,abs=1e-8)
    np.testing.assert_allclose(df,ds,rtol=1e-8,atol=1e-7)
    np.testing.assert_allclose(list(tf.values()),list(ts.values()),rtol=1e-10,atol=1e-8)


def test_streamed_capture_hashes_preserve_original_contract_and_validation_isolation():
    import hashlib,json
    from moxian_web.bone_calibration import capture_hashes
    rows=[dict(partition='initialization',text='左手'),dict(partition='train',value=[1,2.5]),dict(partition='validation',value=[3])]
    source,training=capture_hashes(rows)
    assert source==hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    assert training==hashlib.sha256(json.dumps(rows[:2],sort_keys=True).encode()).hexdigest()
    rows[-1]['value']=[4]
    changed,unchanged=capture_hashes(rows)
    assert changed!=source and unchanged==training
    empty=hashlib.sha256(b'[]').hexdigest()
    assert capture_hashes([])==(empty,empty)
