"""Native Wuji Hand 2.2, finite PD actuation and a genuinely free object."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from retarget.env.grasp.hand import hand_tree


def build_scene(mesh_path, *, mass=.1, friction=.8, timestep=1/300, floor_z=None):
    if mass<=0 or friction<=0 or timestep<=0:raise ValueError('Positive physical parameters required')
    tree=hand_tree('wuji2')
    tree.set('model','ho3d_wuji2_dynamic')
    option=tree.find('option')
    option.set('timestep',str(timestep));option.set('cone','elliptic');option.set('iterations','25');option.set('ls_iterations','10')
    option.set('gravity','0 0 -9.81')
    # Select discrete dynamics consistent with the MJWarp integrator.
    flag=option.find('flag')
    if flag is None:flag=ET.SubElement(option,'flag')
    flag.set('invdiscrete','enable')
    for geom in tree.findall('.//geom'):
        if geom.get('contype','1')!='0':
            geom.set('contype','1');geom.set('conaffinity','3');geom.set('condim','3')
            geom.set('friction',f'{friction} .005 .0001');geom.set('solref','.012 1')
    mesh_path=Path(mesh_path).resolve()
    if not mesh_path.is_file():raise FileNotFoundError(mesh_path)
    ET.SubElement(tree.find('asset'),'mesh',name='tracked_object_mesh',file=str(mesh_path))
    world=tree.find('worldbody')
    obj=ET.SubElement(world,'body',name='tracked_object',pos='0 0 .3')
    ET.SubElement(obj,'freejoint',name='tracked_object_joint')
    ET.SubElement(obj,'geom',name='tracked_object_geom',type='mesh',mesh='tracked_object_mesh',mass=str(mass),
                  contype='2',conaffinity='1',condim='3',friction=f'{friction} .005 .0001',solref='.012 1',rgba='.9 .55 .12 1')
    if floor_z is not None:
        ET.SubElement(world,'geom',name='floor',type='plane',size='2 2 .01',pos=f'0 0 {floor_z}',contype='2',conaffinity='3',friction=f'{friction} .005 .0001')
    xml=ET.tostring(tree,encoding='unicode')
    model=mujoco.MjModel.from_xml_string(xml)
    if model.nq!=33 or model.nu!=26:raise ValueError(f'Expected native 26 hand + free object: nq={model.nq}, nu={model.nu}')
    return model,xml


def fit_kinematic_reference(model,joints,device='cuda:0'):
    if not str(device).startswith('cuda'):raise ValueError('Reference IK requires CUDA; CPU fallback is disabled')
    import torch
    from retarget.env.grasp.pose_gpu import WujiPoseAdapter
    hand=mujoco.MjModel.from_xml_string(ET.tostring(hand_tree('wuji2'),encoding='unicode'))
    adapter=WujiPoseAdapter(hand,device)
    q=adapter.initialize(torch.as_tensor(joints,device=device,dtype=torch.float64)).detach().cpu().numpy()
    q[:,3:6]=np.unwrap(q[:,3:6],axis=0)
    q=np.clip(q,*model.actuator_ctrlrange.T)
    return q,dict(method='GPU_bounded_LM_absolute_bone_directions',physics_backend='none_static_GPU_FK',frames=len(q))
