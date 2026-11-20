"""Shared contact queries and version identity for primitive plus delta fitting."""
import hashlib
from pathlib import Path
import numpy as np
import trimesh

from .cart_protocol import LEGACY_PROTOCOL as PROTOCOL, STAGES
from .cart_primitives import templates
from retarget.baseline.geort.adapter import digest

REVISION='cart_primitive_delta_v5'


def _primitive_policy():
    from .cart_primitive_fit import POLICY
    from .cart_primitives import REVISION as estimator_revision
    return dict(POLICY,revision=estimator_revision,code_sha256={name:hashlib.sha256(
        Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
            'cart_primitive_fit.py','cart_primitives.py','cart_convex.py','cart_geometry.py','cart_data.py')})


def geometry_identity():
    return dict(revision=REVISION,protocol=PROTOCOL,stages_sha256=digest(STAGES),
        templates_sha256=digest(sorted({(r['template_file'],r['template_sha256'],r['annotation_sha256']) for r in templates()})),
        primitive_policy=_primitive_policy())


def _patches(vertices,faces,regions,names,full):
    result={};vertices=np.asarray(vertices);faces=np.asarray(faces)
    for name in names:
        ids=np.asarray(regions.get(name,[]),int)
        if not len(ids):raise ValueError('缺少语义曲面 '+name)
        ids=ids[np.linspace(0,len(ids)-1,min(48 if full else 16,len(ids))).astype(int)]
        tri=vertices[faces[ids]]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        lengths=np.linalg.norm(normals,axis=1)
        valid=lengths>1e-12
        if not valid.any():raise ValueError('退化的语义曲面 '+name)
        tri=tri[valid];normals=normals[valid]/lengths[valid,None]
        centers=tri.mean(1)
        points=np.concatenate((centers,((tri+np.roll(tri,-1,axis=1))/2).reshape(-1,3)))
        n=np.concatenate((normals,np.repeat(normals,3,axis=0)))
        # Material tangent facets can create a proper contact face on the hull,
        # rather than a lone sharp support vertex with an unrelated normal.
        tangent=tri-centers[:,None]
        extent=np.linalg.norm(tangent,axis=-1).max(1)
        tangent*=np.minimum(1.,.001/np.maximum(extent,1e-12))[:,None,None]
        tangent=np.concatenate((tangent,np.repeat(tangent,3,axis=0)))
        result[name]=(points,n,tangent)
    return result


def hull_contacts(hull,patches,spec):
    mesh=trimesh.Trimesh(hull['vertices_m'],hull['faces'],process=False)
    contacts=[]
    for name in spec['patches']:
        p,n,_=patches[name]
        q,d,ids=trimesh.proximity.closest_point(mesh,p)
        dots=(n*mesh.face_normals[ids]).sum(-1).clip(-1,1)
        score=20*(d/.002)**2+5*(1+dots)/(1-np.cos(np.deg2rad(15)))
        k=int(score.argmin())
        contacts.append(dict(patch=name,weight=spec['patch_weights'][name],gap_m=float(d[k]),
            point_m=p[k].tolist(),normal=n[k].tolist(),object_point_m=q[k].tolist(),
            object_normal=mesh.face_normals[ids[k]].tolist(),normal_error_deg=float(np.degrees(np.arccos(-dots[k])))))
    return contacts


def estimate_convex(vertices,faces,regions,stage_id,full=False,budget_ms=120,witnesses=None,joints=None,previous=None):
    from .cart_primitive_fit import estimate_primitive
    return estimate_primitive(vertices,faces,regions,stage_id,full,budget_ms,witnesses,joints,previous)
