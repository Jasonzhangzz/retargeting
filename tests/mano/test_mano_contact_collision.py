import copy
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import torch
import pytest
import trimesh
import mujoco
from scipy.spatial.transform import Rotation
from retarget.env.grasp.mano_contact_collision import ManoContactCollision


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Native MJWarp collision requires CUDA')
def test_heterogeneous_mesh_selection_matches_independent_cpu_geometry():
    root=ET.Element('mujoco');option=ET.SubElement(root,'option');ET.SubElement(option,'flag',multiccd='enable')
    asset=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    cube=trimesh.creation.box(extents=[.02,.02,.02])
    for j in range(16):
        ET.SubElement(asset,'mesh',name=f'hand_{j}',vertex=' '.join(map(str,cube.vertices.ravel())),face=' '.join(map(str,cube.faces.ravel())))
        body=ET.SubElement(world,'body');ET.SubElement(body,'freejoint')
        ET.SubElement(body,'geom',type='mesh',mesh=f'hand_{j}',condim='1')
    decoder=SimpleNamespace(device=torch.device('cuda:0'),ordered_joint_ids=torch.arange(21,device='cuda:0'),
        collision_geometry=SimpleNamespace(xml=ET.tostring(root,encoding='unicode')))
    obj=trimesh.creation.box(extents=[.04,.05,.06]);obj.apply_translation([.006,-.003,.004])
    geo={0:dict(vertices=np.asarray(obj.vertices,np.float32),faces=np.asarray(obj.faces,np.int32))}
    origin=np.array([[.1,.2,.3],[-.1,.15,.2]],np.float32)
    rotation=Rotation.from_rotvec([[0,0,0],[.2,-.3,.5]]).as_matrix().astype(np.float32)
    surfaces=SimpleNamespace(object_keys=[(0,1.),(0,1.7)],geometries=geo,origin=torch.tensor(origin,device='cuda:0'),rotation=torch.tensor(rotation,device='cuda:0'))
    native=ManoContactCollision(decoder,surfaces,2)
    joints=torch.zeros(2,21,3,device='cuda:0')
    joints[:,:,0]=torch.arange(21,device='cuda:0')*10
    for i,scale in enumerate([1.,1.7]):
        local=np.array([(.006-.02)*scale-.005,-.003*scale,.004*scale])
        joints[i,0]=torch.tensor(origin[i]+rotation[i]@local,device='cuda:0')
    frames=torch.eye(3,device='cuda:0').expand(2,16,3,3).clone();frames[:,0]=surfaces.rotation
    result=native.query(joints,frames,torch.arange(2,device='cuda:0'))
    assert not result['overflow'].any()
    assert (result['count']>0).all()
    for i in range(2):
        cpu=copy.copy(native.cpu)
        mid=i
        for key,values in native.tables.items():
            v=values[mid].cpu().numpy()
            getattr(cpu,key)[native.object_geom]=v.reshape(-1) if key=='geom_aabb' else v
        data=mujoco.MjData(cpu)
        data.qpos[:]=native.data.qpos.numpy()[i]
        mujoco.mj_kinematics(cpu,data);mujoco.mj_collision(cpu,data)
        expected=max([0.]+[-c.dist for c in data.contact if native.object_geom in c.geom])
        assert result['depth'][i,0].item()==pytest.approx(expected,abs=2e-5)
        active=result['eligible'][i]
        # Endpoint convention and normal orientation must reconstruct signed gap.
        delta=result['hand'][i,active]-result['object'][i,active]
        reconstructed=(delta*result['normal'][i,active]).sum(-1)
        torch.testing.assert_close(reconstructed,result['gap'][i,active],atol=1e-7,rtol=1e-5)
