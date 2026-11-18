"""Dataset priors and bounded, per-frame cylinder/sphere fitting for CART.

The dataset supplies geometry priors, never robot supervision or success labels.
Template poses initialize the current hand fit; dimensions are re-estimated
before bounded contact geometry is added.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from retarget.baseline.geort.adapter import ROOT, anatomical_basis, digest
from .cart_protocol import PRIMITIVE_TYPES, DIRECTIONS, stage

REVISION = 'cart_hand_sized_primitive_v3'
DATASET = ROOT / 'data/datasets/dex_grasping_primitives_20261115'
ANNOTATIONS = ROOT / 'data/grasp/templates/assets/hand/shadow/raw_anno'


def annotation_patches(path):
    """Upstream contact-bearing finger segments, including thumb opposition."""
    names={'ff':'index','mf':'middle','rf':'ring','lf':'pinky','th':'thumb'}
    patches=[]
    for body in (yaml.safe_load(Path(path).read_text()).get('contact') or {}):
        for prefix,finger in names.items():
            for segment,part in (('proximal','proximal'),('middle','intermediate'),('distal','distal')):
                if body=='rh_'+prefix+segment:
                    patches.append(finger+'_'+('proximal' if finger=='thumb' and part=='intermediate' else part))
    return sorted(set(patches))


@lru_cache(maxsize=4)
def _templates(root, signatures):
    rows=[]
    for filename, _, _, annotation_filename, annotation_sha in signatures:
        path=Path(root)/filename
        fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
        type_id=int(path.stem.split('_')[0])
        required=annotation_patches(ANNOTATIONS/annotation_filename)
        with np.load(path,allow_pickle=False) as z:
            for i,(points,pose) in enumerate(zip(z['target_keypoints_world'],z['object_pose_world'])):
                if not np.isfinite(np.r_[points.ravel(),pose,z['radius_m'][i],z['height_m'][i]]).all():
                    raise ValueError('primitive 模板包含无效数值：'+path.name)
                basis=anatomical_basis(points)
                # Dataset quaternions are WXYZ; scipy uses XYZW.
                rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
                rows.append(dict(type_id=type_id,row=i,kind='sphere' if type_id in (11,13,26,28) else 'cylinder',
                    radius_m=float(z['radius_m'][i]),height_m=float(z['height_m'][i]),
                    center=(pose[:3]-points[0])@basis,rotation=basis.T@rotation,
                    joints=(points-points[0])@basis,template_sha256=fingerprint,
                    required_patches=required,annotation_sha256=annotation_sha,
                    template_file=path.relative_to(Path(root)).as_posix()))
    return rows


def templates(root=DATASET):
    root=Path(root)
    files=[]
    for type_id in sorted({n for ids in PRIMITIVE_TYPES.values() for n in ids}):
        found=sorted((root/'glove/grids').glob(f'{type_id}_*.npz'))
        if len(found)!=1:raise ValueError(f'缺少或重复的 CART primitive 模板：type {type_id}')
        files.append(found[0])
    signatures=[]
    for p in files:
        annotation=ANNOTATIONS/(p.stem+'.yaml')
        signatures.append((p.relative_to(root).as_posix(),p.stat().st_mtime_ns,p.stat().st_size,
                           annotation.name,hashlib.sha256(annotation.read_bytes()).hexdigest()))
    signatures=tuple(signatures)
    return _templates(str(root.resolve()),signatures)


def geometry_identity():
    from .cart_convex import geometry_identity as current_identity
    return current_identity()


def surface(points, center, rotation, radius, height, kind):
    """Analytic signed distance, closest point and outward surface normal."""
    if kind not in ('sphere','cylinder'):raise ValueError('不支持的 CART primitive：'+str(kind))
    local=(np.asarray(points)-center)@rotation
    if kind=='sphere':
        length=np.linalg.norm(local,axis=-1)
        normal=local/np.maximum(length[...,None],1e-12)
        normal=np.where((length<1e-12)[...,None],np.array([1.,0.,0.]),normal)
        hit=normal*radius
        sdf=length-radius
    else:
        radial=np.linalg.norm(local[...,:2],axis=-1)
        radial_normal=local[...,:2]/np.maximum(radial[...,None],1e-12)
        radial_normal=np.where((radial<1e-12)[...,None],np.array([1.,0.]),radial_normal)
        d=np.stack((radial-radius,np.abs(local[...,2])-height/2),axis=-1)
        sdf=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)
        side=local.copy();side[...,:2]=radial_normal*radius
        side[...,2]=np.clip(side[...,2],-height/2,height/2)
        cap=local.copy();cap[...,:2]=radial_normal*np.minimum(radial,radius)[...,None]
        cap[...,2]=np.where(local[...,2]>=0,height/2,-height/2)
        use_side=np.linalg.norm(local-side,axis=-1)<=np.linalg.norm(local-cap,axis=-1)
        hit=np.where(use_side[...,None],side,cap)
        inside=d.max(-1)<=0
        side_normal=np.concatenate((radial_normal,np.zeros_like(radial[...,None])),axis=-1)
        cap_normal=np.zeros_like(local);cap_normal[...,2]=np.where(local[...,2]>=0,1.,-1.)
        boundary_normal=np.where(use_side[...,None],side_normal,cap_normal)
        delta=local-hit
        normal=np.where(inside[...,None],boundary_normal,delta/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12))
        normal=np.where((np.abs(sdf)<1e-10)[...,None],boundary_normal,normal)
    return sdf,hit@rotation.T+center,normal@rotation.T


def primitive_target(kind,radius,height,center,rotation,reference):
    """Mesh transport for the exact same parameterized object on both hands."""
    if kind not in ('sphere','cylinder'):raise ValueError('不支持的 CART primitive：'+str(kind))
    from .cart_geometry import hull_data
    mesh=(trimesh.creation.icosphere(subdivisions=3,radius=radius) if kind=='sphere' else
          trimesh.creation.cylinder(radius=radius,height=height,sections=64))
    result=hull_data(mesh.vertices@np.asarray(rotation).T+center)
    result.update(algorithm=REVISION,status='primitive',contact_mode='primitive',
        primitive=dict(kind=kind,radius_m=float(radius),height_m=float(2*radius if kind=='sphere' else height),
            center_m=np.asarray(center).tolist(),rotation=np.asarray(rotation).tolist(),
            reference_type=int(reference['type_id']),reference_row=int(reference['row']),
            template_file=reference['template_file'],template_sha256=reference['template_sha256'],
            estimator_revision=REVISION),mesh_approximation=True)
    # Symmetric generated meshes have a known center, independent of triangulation.
    result['center_m']=np.asarray(center).tolist()
    result['support_m']=((np.asarray(result['vertices_m'])-center)@DIRECTIONS.T).max(0).tolist()
    return result


def _proposals(spec,joints,cloud):
    proposals=[]
    for ref in templates():
        if ref['type_id'] not in spec['primitive_types']:continue
        center=ref['center'].copy();rotation=ref['rotation'].copy();error=0.
        if joints is not None:
            fingers={p.split('_')[0] for p in ref['required_patches']}
            ids=[i for f,name in enumerate(('thumb','index','middle','ring','pinky')) if name in fingers
                 for i in range(1+4*f,5+4*f)]
            source_ids=target_ids=ids
            a=ref['joints'][source_ids];b=np.asarray(joints)[target_ids]
            u,_,vt=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)))
            proper=np.eye(3);proper[-1,-1]=np.linalg.det(u@vt)
            row_rotation=u@proper@vt
            scale=float(np.sum((a-a.mean(0))@row_rotation*(b-b.mean(0)))/max(np.sum((a-a.mean(0))**2),1e-12))
            shift=b.mean(0)-scale*a.mean(0)@row_rotation
            center=scale*center@row_rotation+shift;rotation=row_rotation.T@rotation
            error=float(np.mean((scale*a@row_rotation+shift-b)**2))
        else:
            center=cloud.mean(0)
            error=abs(ref['radius_m']-np.linalg.norm(np.ptp(cloud,axis=0))/4)
            # Geometry-only callers still need independent axis hypotheses.
            for vector in ([0,0,0],[np.pi/2,0,0],[0,np.pi/2,0]):
                proposals.append((error,ref,center,Rotation.from_rotvec(vector).as_matrix()))
            continue
        proposals.append((error,ref,center,rotation))
    return sorted(proposals,key=lambda p:(p[0],p[1]['type_id'],p[1]['row']))
