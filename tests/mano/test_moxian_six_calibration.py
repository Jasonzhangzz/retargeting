"""Moxian six poses retain the Moxian FK, RX axes and fixed mesh variables."""
import copy
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from moxian_web.moxian_six_calibration import training_rows,SixObjective
from moxian_web.wuji_native_abi import SDK_POSES,FINGERS
from moxian_web.wuji_native_capture import PROTOCOL
from moxian_web.qnbot_surface import QnbotSurface
from moxian_web.bone_calibration import frozen_contacts
from moxian_web.bone_extrinsics import definition_from_surface,model_hash,validate_candidate
from moxian_web.contact import default_mapping


@pytest.mark.parametrize('side',['left','right'])
def test_moxian_six_fk_rotation_residual_gradient_and_frozen_geometry(side):
    torch.set_num_threads(1);model=QnbotSurface(side).double();model.bind_mapping(dict(assignments=default_mapping(side),confirmed=[]))
    for p in model.parameters():p.requires_grad_(False)
    q=model.q0[None].repeat(6,1)
    for i in range(4):q[i,model.joint_names.index(side+'_'+('index','middle','ring','little')[i]+'_mcp_flexion')]=.4
    for finger in ('index','middle','ring','little'):q[4,model.joint_names.index(side+'_'+finger+'_mcp_flexion')]=1.3
    p,r=model.sensor_prediction(q);frames={}
    for i,stage in enumerate(SDK_POSES):
        nodes=[dict(position=p[i,f].tolist(),quaternion_xyzw=Rotation.from_matrix(r[i,f].numpy()).as_quat().tolist()) for f in range(5)]
        frames[stage]=[nodes,copy.deepcopy(nodes)]
    document=dict(protocol=PROTOCOL,completed_groups=list(SDK_POSES),state=dict(state='complete'),pose_frames=frames)
    before=copy.deepcopy(document);raw,train=training_rows(document);assert before==document and len(train)==6
    for i,row in enumerate(train):row['skeleton']=dict(joint_names=model.joint_names,joint_positions_rad=q[i].tolist())
    contacts=frozen_contacts(model,train);obj=SixObjective(model,train,contacts,dict(position_sigma_m=.003,rotation_sigma_rad=.15))
    x=torch.tensor(np.r_[np.zeros(51),q.numpy().ravel()],dtype=torch.float64,requires_grad=True)
    loss,terms=obj.loss(x);gradient=torch.autograd.grad(loss,x)[0]
    assert x.numel()==201 and terms['sensor_orientation']<1e-8
    for i in (0,14,15,30,45,48,51,76,200):
        delta=torch.zeros_like(x);delta[i]=1e-6
        numeric=(obj.loss(x.detach()+delta)[0]-obj.loss(x.detach()-delta)[0])/2e-6
        assert float(gradient[i])==pytest.approx(float(numeric),rel=3e-4,abs=.005)
    geometry=obj.coordinates.geometry(x[:51]);assert all(torch.count_nonzero(geometry[k])==0 for k in ('width','depth','root','axis','residual'))
    # Rotating every RX by 90 degrees is visible in the objective; it cannot
    # silently pass merely because joint/keypoint positions agree.
    wrong=copy.deepcopy(train)
    for row in wrong:
        for n in row['fingers'].values():n['quaternion_xyzw']=(Rotation.from_euler('z',90,degrees=True)*Rotation.from_quat(n['quaternion_xyzw'])).as_quat().tolist()
    bad=SixObjective(model,wrong,contacts,obj.noise)
    assert bad.loss(x)[1]['sensor_orientation']>1000
    candidate=model.artifact();candidate.update(serial='TEST',side=side,calibration_fingerprint='fixed',bone_extrinsics=definition_from_surface(model),delta_policy=None)
    candidate['geometry']['alignment'][0]=np.pi/2;candidate['model_hash']=model_hash(candidate)
    with pytest.raises(ValueError,match='越界'):validate_candidate(candidate)


def test_live_qp_result_and_nested_numpy_flags_are_json_serializable():
    import json
    from moxian_web.ik_worker import json_value
    from moxian_web.emf_qp import QPSkeleton
    model=QnbotSurface('right').double();model.bind_mapping(dict(assignments=default_mapping('right'),confirmed=[]))
    c=model.artifact();c.update(serial='TEST',side='right',calibration_fingerprint='fixed',bone_extrinsics=definition_from_surface(model),delta_policy=None);c['model_hash']=model_hash(c)
    with torch.no_grad():p,r=model.sensor_prediction(model.q0[None])
    row=dict(timestamp_ms=1000,fingers={f:dict(position=p[0,i].tolist(),quaternion_xyzw=Rotation.from_matrix(r[0,i].numpy()).as_quat().tolist()) for i,f in enumerate(FINGERS)})
    result=QPSkeleton(c).step(row)
    assert isinstance(result['ik_diagnostics']['fingers'][0]['tracking_satisfied'],bool)
    json.dumps(json_value(result),allow_nan=False)
    assert json.loads(json.dumps(json_value(dict(flag=np.bool_(True),counts=np.array([1,2]),n=np.int64(3)))))==dict(flag=True,counts=[1,2],n=3)
