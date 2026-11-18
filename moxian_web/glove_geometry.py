"""Bounded, static glove geometry shared by calibration, skin and skeleton.

The middle MCP fixes the palm gauge. Lengths/widths are log scales, root
translations are in centimetres and the three CMC axis tilts are in radians.
"""
from __future__ import annotations

import hashlib
import json
import math
import numpy as np

REVISION = 'personalized_glove_geometry_v1'
COUNT = 43
ROOT_FINGERS = (0, 1, 3, 4)
CHAINS = ((13,14,15),(1,2,3),(4,5,6),(10,11,12),(7,8,9))
LOWER = [-math.log(1.5)]*15 + [-math.log(1.6)]*10 + [-.8]*12 + [-math.radians(10)]*6
UPPER = [-x for x in LOWER]


def validate(value):
    p = np.asarray(value, dtype=float)
    if p.shape != (COUNT,) or not np.isfinite(p).all(): raise ValueError('无效的个体手套几何参数')
    # Geometric plausibility is an offline soft loss and final validation.
    # Serialization checks numerical validity, not the old fitting box.
    with np.errstate(over='ignore',under='ignore'):
        scales=np.exp(p[:25]).astype(np.float32)
    if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('个体手套尺寸不能用有限正数表示')
    return p


def fingerprint(parameters):
    return hashlib.sha256(REVISION.encode()+validate(parameters).tobytes()).hexdigest()[:16]


def validate_payload(value,side=None,parameters=None):
    if value.get('revision')!=REVISION or value.get('side') not in ('left','right') or side and value['side']!=side:
        raise ValueError('个体骨架版本或手侧不匹配')
    p=validate(value['parameters'])
    if parameters is not None and not np.array_equal(p,validate(parameters)):raise ValueError('骨架与蒙皮几何参数不一致')
    fields=dict(joints_m=(16,3),tips_m=(5,3),sensor_position_m=(5,3),sensor_rotation=(5,3,3),
                distal_frame_corrections=(5,3,3),mano_to_wrist_rotation=(3,3),binding_corrections=(16,3,3),
                raw_to_wrist_rotation=(3,3),raw_to_wrist_translation_m=(3,),pad_local_m=(5,3),pad_normal_local=(5,3))
    for key,shape in fields.items():
        a=np.asarray(value[key])
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('个体骨架字段无效：'+key)
        if shape[-2:]==(3,3) and (not np.allclose(a.swapaxes(-1,-2)@a,np.eye(3),atol=1e-5) or np.any(np.linalg.det(a)<.999)):
            raise ValueError('个体骨架旋转不是 SO(3)：'+key)
    if value['chains']!=[list(c) for c in CHAINS]:raise ValueError('个体骨架链顺序不匹配')
    raw={k:v for k,v in value.items() if k not in ('geometry_hash','urdf_path')}
    expected=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()[:16]
    if expected!=value['geometry_hash']:raise ValueError('个体骨架摘要不匹配')
    return value


def deform_rest(vertices, joints, weights, chains, tips, normal, parameters):
    """Differentiable fixed rest geometry; joint centres and skin move together."""
    import torch
    p = torch.as_tensor(parameters, device=vertices.device, dtype=vertices.dtype)
    lengths = p[:15].reshape(5,3).exp(); widths = p[15:20].exp(); depths = p[20:25].exp()
    roots = {f:p[25+3*i:28+3*i]*.01 for i,f in enumerate(ROOT_FINGERS)}
    out = weights[:,0,None]*vertices
    new_j = list(joints.unbind())
    normal = torch.nn.functional.normalize(normal, dim=0)
    for f, chain in enumerate(chains):
        root = joints[chain[0]] + roots.get(f, torch.zeros_like(joints[0]))
        ends = torch.cat((joints[list(chain[1:])],vertices[[tips[f]]]),0)
        vectors = ends-joints[list(chain)]
        for k,joint in enumerate(chain):
            centre = root if k == 0 else new_j[chain[k-1]]+vectors[k-1]*lengths[f,k-1]
            new_j[joint] = centre
            axis = torch.nn.functional.normalize(vectors[k], dim=0)
            lateral = torch.nn.functional.normalize(torch.linalg.cross(axis,normal), dim=0)
            depth = torch.linalg.cross(lateral,axis)
            offset = vertices-joints[joint]
            transformed = (offset@axis)[:,None]*axis*lengths[f,k] + (offset@lateral)[:,None]*lateral*widths[f] + (offset@depth)[:,None]*depth*depths[f] + centre
            out = out+weights[:,joint,None]*transformed
    return out, torch.stack(new_j)


def export_skeleton(model):
    """One immutable geometry/registration payload, expressed in wrist metres.

    Runtime consumes these actual joint centres, rather than independently
    fitting a second set of skeleton lengths to the same measurements.
    """
    import torch
    from scripts.calibrate_wuji_mano import CHAINS, TIPS
    model.official_fk.bind(model, CHAINS)
    rotation = model.official_fk.mano_to_sdk
    reference={**model.official_fk.reference,'hand_side':model.side,'coordinate_frame':'wrist_local'}
    if model.official_fk.fixed_binding:
        reference['personalized_model']={**model.official_fk.fixed_binding,'urdf_path':str(model.official_fk.path)}
    pose,root,*_=model.official_fk.parameters(model,dict(timestamp_ms=0.,skeleton=reference),CHAINS)
    with torch.no_grad():
        v,j,_,_=model.decode(torch.as_tensor(pose[None],device=model.device),torch.as_tensor(root[None],device=model.device))
    neutral_vertices=v[0].cpu().numpy();neutral_joints=j[0].cpu().numpy()
    joints = neutral_joints@rotation.T
    tips = neutral_vertices[list(TIPS)]@rotation.T
    payload=dict(revision=REVISION, parameters=validate(model.glove_parameters).tolist(), side=model.side,
                chains=[list(c) for c in CHAINS], joints_m=joints.tolist(), tips_m=tips.tolist(),
                raw_to_wrist_rotation=(rotation@model.raw_rotation).tolist(),
                raw_to_wrist_translation_m=(rotation@model.raw_translation).tolist(),
                sensor_position_m=model.sensor_offsets().tolist(), sensor_rotation=model.mount_rot.tolist(),
                distal_frame_corrections=model.official_fk.corrections[[c[-1] for c in CHAINS]].tolist(),
                mano_to_wrist_rotation=rotation.tolist(),
                binding_corrections=model.official_fk.corrections.tolist())
    from .contact import PATCH_IDS
    from .mano_runtime import vertex_normals
    normals=vertex_normals(neutral_vertices,model.faces);topology=model.topology()
    faces=np.asarray(topology['sid_face_ids']);bc=np.asarray(topology['sid_barycentric'])
    skin_points=(neutral_vertices[model.faces[faces]]*bc[...,None]).sum(1)
    skin_normals=(normals[model.faces[faces]]*bc[...,None]).sum(1)
    payload['pad_local_m']=[];payload['pad_normal_local']=[]
    for f,(name,chain) in enumerate(zip(('thumb','index','middle','ring','pinky'),CHAINS)):
        ids=np.flatnonzero(np.asarray(model.mapping['assignments'])==PATCH_IDS.index(name+'_distal'))
        if not len(ids):raise ValueError('指腹缺少材料锚点：'+name)
        from scipy.spatial.transform import Rotation
        distal_name=('thumb','index','middle','ring','little')[f]+'_distal'
        source_rotation=Rotation.from_quat(reference['orientations_local_xyzw'][reference['names'].index(distal_name)]).as_matrix()
        to_local=source_rotation.T@rotation
        payload['pad_local_m'].append((to_local@(skin_points[ids].mean(0)-neutral_joints[chain[-1]])).tolist())
        n=to_local@skin_normals[ids].mean(0);payload['pad_normal_local'].append((n/np.linalg.norm(n)).tolist())
    payload['geometry_hash']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
    return payload


def skeleton_urdf(payload, source, destination):
    """Export a private URDF using the same rest centres and CMC axes as skin."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    from .official_fk import urdf_reference
    validate_payload(payload);side=payload['side']
    reference=urdf_reference(source,side);names=reference['names']
    positions=dict(zip(names,np.asarray(reference['positions_local'])))
    rotations=dict(zip(names,Rotation.from_quat(reference['orientations_local_xyzw']).as_matrix()))
    tree=ET.parse(source);elements={j.get('name'):j for j in tree.getroot().findall('joint')}
    points=np.asarray(payload['joints_m']);tips=np.asarray(payload['tips_m'])
    for f,(finger,chain) in enumerate(zip(('thumb','index','middle','ring','little'),payload['chains'])):
        prefix=side+'_'+finger+'_';first='metacarpal' if f==0 else 'proximal'
        root=elements[prefix+('cmc_abduction' if f==0 else 'mcp_abduction')]
        # Move the first articulating pivot in its parent's original frame.
        parent=root.find('parent').get('link').removeprefix(side+'_')
        root.find('origin').set('xyz',' '.join(map(str,rotations[parent].T@(points[chain[0]]-positions[parent]))))
        joint_names=('mcp_abduction','ip_flexion','tip_fixed') if f==0 else ('pip_flexion','dip_flexion','tip_fixed')
        for k,name in enumerate(joint_names):
            element=elements[prefix+name]
            parent_name=element.find('parent').get('link').removeprefix(side+'_')
            delta=(points[chain[k+1]] if k<2 else tips[f])-points[chain[k]]
            element.find('origin').set('xyz',' '.join(map(str,rotations[parent_name].T@delta)))
    p=np.asarray(payload['parameters'])
    for i,name in enumerate(('cmc_abduction','cmc_flexion','cmc_opposition')):
        joint=elements[side+'_thumb_'+name];axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
        tangent=np.cross(axis,np.eye(3)[np.abs(axis).argmin()]);tangent/=np.linalg.norm(tangent)
        tilt=p[37+2*i]*tangent+p[38+2*i]*np.cross(axis,tangent)
        axis=Rotation.from_rotvec(tilt).apply(axis)
        joint.find('axis').set('xyz',' '.join(map(str,axis)))
    for name,joint in elements.items():
        if '_cmc_' in name and '_thumb_' not in name:
            joint.find('limit').set('lower','0');joint.find('limit').set('upper','0')
    from pathlib import Path
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(tree);temporary=path.with_suffix('.tmp');tree.write(temporary,encoding='unicode');temporary.replace(path)
    return path
