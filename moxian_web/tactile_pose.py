"""Tactile material constraints on official MANO pose parameters only.

Taxels measure activity, not the identity of an opposing surface. Bilateral,
nearby and opposing patches are associated conservatively; unmatched activity
remains an external-object target. No mesh vertices or shape are displaced.
"""
import numpy as np
import torch
from .contact import PATCH_IDS,FINGERS
from retarget.baseline.card.geometry import contact_weights,chart_point
from retarget.baseline.card.math import exp_so3,log_so3
from .wuji_surface import CHAINS,unit
from .calibration_targets import pinch_patches

POSE_CONTACT_POLICY='official_mano_material_contact_v2'


def material_anchors(topology,vertices,joints,contact,stamp,tactile_stamp,guided=()):
    """One projected CoP per *sensor patch*, retaining its face/barycentric ID."""
    faces=np.asarray(topology['faces']);patches=np.asarray(topology['sid_patch_indices'])
    sidfaces=np.asarray(topology['sid_face_ids']);bary=np.asarray(topology['sid_barycentric'])
    tri=np.asarray(vertices)[faces];rawnormal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal=unit(rawnormal)
    points=(tri[sidfaces]*bary[...,None]).sum(1)
    weights=contact_weights(contact or {},patches,stamp,tactile_stamp if isinstance(tactile_stamp,(int,float)) else -1e20)
    skin=np.asarray(topology['skin_weights']);sw=(skin[faces[sidfaces]]*bary[...,None]).sum(1)
    anchors=[]
    for patch in range(20):
        ids=np.flatnonzero(weights['mask']&(patches==patch));observed=bool(len(ids))
        if not observed:
            if patch not in guided:continue
            ids=np.flatnonzero(patches==patch)
        response=weights['response'][ids] if observed else np.ones(len(ids));force=float(response.sum()) if observed else 0.
        response=response if response.sum()>1e-12 else np.ones(len(ids))
        centre=np.average(points[ids],axis=0,weights=response);n=unit(np.average(normal[sidfaces[ids]],axis=0,weights=response))
        if patch>=14:f=-1;k=-1;region=0;axis=unit(joints[9]-joints[0])
        else:
            f=0 if patch<2 else 1+(patch-2)//3;k=(2 if patch==0 else int(np.average(sw[ids][:,list(CHAINS[0][:2])],axis=0,weights=response).argmax())) if f==0 else 2-(patch-2)%3
            region=1+3*f+k;axis=unit(joints[2+4*f+k]-joints[1+4*f+k])
        pool=np.flatnonzero(np.asarray(topology['face_patches'])==patch)
        pool=pool[normal[pool]@n>.1]
        if not len(pool):pool=np.unique(sidfaces[ids])
        pool=np.union1d(pool,sidfaces[ids])
        position,index,bc=chart_point(tri[pool],centre,axis,n,np.zeros(2),centre)
        source_face=int(pool[index])
        if observed and len(ids)==1:
            position=points[ids[0]];source_face=int(sidfaces[ids[0]]);bc=bary[ids[0]]
        anchors.append(dict(patch=patch,patch_id=PATCH_IDS[patch],finger=f,segment=k,region=region,
            position_m=position.tolist(),direction=n.tolist(),source_face=source_face,barycentric=bc.tolist(),
            sids=ids.tolist() if observed else [],force=force,evidence='bilateral_tactile' if observed else 'confirmed_calibration_gesture'))
    return anchors


def associate_self_contacts(anchors,guided_finger=None):
    """No single-patch activity creates an invented hand-hand contact."""
    candidates=[]
    for i,a in enumerate(anchors):
        for b in anchors[i+1:]:
            fa,fb=a['finger'],b['finger'];pa,pb=a['patch'],b['patch']
            if fa==fb:continue
            guided=guided_finger is not None and {pa,pb}==set(pinch_patches('pinch_'+guided_finger))
            d=np.asarray(b['position_m'])-a['position_m'];gap=float(np.linalg.norm(d));facing=float(np.dot(a['direction'],b['direction']))
            tip_pair=min(fa,fb)==0 and a['segment']==b['segment']==2
            finger_pair=min(fa,fb)>=0
            palm=min(fa,fb)==-1 and max(fa,fb)>=0 and max(a['segment'],b['segment'])>=1
            if not guided:
                if not (finger_pair or palm):continue
                # Bilateral pressure can come from an external object. It
                # cannot identify a distant partner or reverse bad normals.
                # Tactile refines an already nearby opposing surface only.
                limit=.012 if tip_pair or palm else .015
                toward=(float(np.dot(unit(d),a['direction']))+float(np.dot(-unit(d),b['direction'])))/2 if gap>1e-9 else 1.
                if gap>limit or facing>-.1 or (gap>.004 and toward<.15):continue
            candidates.append((gap+(1+facing)*.005,a,b,guided))
    used=set();pairs=[]
    for _,a,b,guided in sorted(candidates,key=lambda x:(not x[3],x[0])):
        if a['patch'] in used or b['patch'] in used:continue
        used.update((a['patch'],b['patch']))
        pairs.append(dict(a=a,b=b,regions=[a['region'],b['region']],patches=[a['patch'],b['patch']],
                          evidence='confirmed_calibration_gesture' if guided else 'bilateral_tactile_geometry_association'))
    return pairs


def frame_constraints(model,frame,vertices,joints,guided=False):
    from .motion_capture import confirmed_touch,PROTOCOL
    stage=frame.get('stage','');finger=stage[6:] if guided and confirmed_touch(frame) else None
    if frame.get('capture_protocol')==PROTOCOL and guided and stage.startswith('pinch_') and not confirmed_touch(frame):return [],[]
    patches=pinch_patches(stage) if finger else ()
    contact=frame.get('contact') or {};stamp=frame['timestamp_ms'];tt=frame.get('tactile_timestamp_ms')
    if not patches and (not contact.get('ready') or not isinstance(tt,(int,float)) or abs(stamp-tt)>25 or abs(stamp-contact.get('timestamp_ms',tt))>25):return [],[]
    key=(model.shape_version,model.binding_version);cached=getattr(model,'_material_contact_topology',None)
    if cached is None or cached[0]!=key:
        cached=(key,model.topology());model._material_contact_topology=cached
    anchors=material_anchors(cached[1],vertices,joints,contact,stamp,tt,patches)
    return anchors,associate_self_contacts(anchors,finger)


def correct_pose(model,frame,pose,root,vertices,joints,iterations=8):
    """Compatibility contact entry point using the shared bounded MANO solver."""
    anchors,pairs=frame_constraints(model,frame,vertices,joints)
    if not pairs:
        model.contact_pose_delta=None
        return pose,dict(policy=POSE_CONTACT_POLICY,applied=False,pairs=[],unmatched_patches=[a['patch_id'] for a in anchors])
    from .mano_objective import solve_pose
    active={a['finger'] for p in pairs for a in (p['a'],p['b']) if a['finger']>=0}
    corrected,solver,report=solve_pose(model,frame,pose,root,tactile=True,iterations=iterations,budget_ms=5000,active_fingers=active)
    model.solver_report=solver
    return corrected,report
