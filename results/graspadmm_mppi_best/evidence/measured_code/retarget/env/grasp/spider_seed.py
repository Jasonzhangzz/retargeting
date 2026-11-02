"""Unverified CA proposals for SPIDER; never forge accepted CA artifacts."""
from pathlib import Path
import copy
import json
import hashlib
import mujoco
import numpy as np
from retarget.common.results import write_json
from .hand import query_path, palmar_axis

POWER_TYPES = frozenset((1,2,3,4,5,10,11,15,17,18,19,26,28,30,31))
GROUPS = ('thumb','index','middle','ring','pinky','palm')


def finger_group(model, body):
    name=model.body(int(body)).name
    return int(name[len('right_finger')])-1 if name.startswith('right_finger') else 5


def palm_region(model):
    """Exposed volar skin from wrist to MCP bases, including base connections."""
    from .hand import rigid_mesh_arrays
    from .semantic_patches import _visible_queries
    import warp as wp
    data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
    palm=model.body('right_palm_link').id
    rotation=data.xmat[palm].reshape(3,3);origin=data.xpos[palm]
    roots=np.asarray([data.xpos[model.body(f'right_finger{f}_link1').id] for f in range(1,6)])
    roots=(roots-origin)@rotation
    normal=palmar_axis(model,palm,None)
    up=roots[1:].mean(0);up-=normal*(up@normal);up/=np.linalg.norm(up)
    side=np.cross(normal,up)
    longitudinal=max(float((roots@up).max()),.02)+.008
    lateral=(float((roots@side).min())-.012,float((roots@side).max())+.012)
    bodies,local,normals=[],[],[]
    for body in range(model.nbody):
        name=model.body(body).name
        if body!=palm and not (name.startswith('right_finger') and name.endswith(('link1','link2'))):continue
        with np.load(query_path(model,body),allow_pickle=False) as raw:
            p,n=raw['local_pos'].astype(float),raw['local_normal'].astype(float)
            source=raw['source_body'].astype(str) if 'source_body' in raw else np.repeat(name,len(p))
        R=data.xmat[body].reshape(3,3)
        chart=(p@R.T+data.xpos[body]-origin)@rotation;nn=n@R.T@rotation
        mask=((chart@up>=-.003)&(chart@up<=longitudinal)&(chart@side>=lateral[0])&
              (chart@side<=lateral[1])&(nn@normal>.5)&(source==name))
        if name.startswith('right_finger'):
            from .regions import belly_axis
            # Robot MCP/CMC housings extend beyond the anatomical palm chart.
            # The entire volar root housing and basal third of the following
            # phalanx belong to the requested broad palm; not the whole finger.
            mask=(n@belly_axis(model,body)>.5)&(source==name)
            if name.endswith('link2'):
                endpoint=model.body(name[:-1]+'3').pos
                fraction=p@endpoint/max(float(endpoint@endpoint),1e-12)
                mask &= (fraction>=-.15)&(fraction<=1/3)
        vertices,faces=rigid_mesh_arrays(model,body)
        mesh=wp.Mesh(points=wp.array(vertices,dtype=wp.vec3,device='cpu'),indices=wp.array(faces.ravel(),dtype=int,device='cpu'))
        visible=wp.empty(len(p),dtype=int,device='cpu')
        wp.launch(_visible_queries,len(p),inputs=[mesh.id,wp.array(p,dtype=wp.vec3,device='cpu'),wp.array(n,dtype=wp.vec3,device='cpu'),visible],device='cpu')
        mask &= visible.numpy().astype(bool)
        if mask.any():
            ids=np.flatnonzero(mask);bodies.extend([body]*len(ids));local.extend(p[ids]);normals.extend(n[ids])
    if not bodies:raise ValueError('No exposed broad-palm skin in WUJI query assets')
    return dict(bodies=np.asarray(bodies,int),local=np.asarray(local),normals=np.asarray(normals),
        palm=palm,normal=normal,up=up,side=side,longitudinal=longitudinal,lateral=np.asarray(lateral))


def export_seed(directory,hand,guess,args,*,template,primitive):
    from .runtime import prepare_experiment
    directory=Path(directory);proposal=copy.deepcopy(guess)
    for name in ('contactaware_reference_qpos','hand_reference_qpos'):
        if hasattr(proposal,name):delattr(proposal,name)
    original_mask=np.asarray(getattr(guess,'contact_mask',np.zeros(len(hand.bodies))),bool)
    proposal.contact_mask=np.ones(len(hand.bodies),bool)
    if np.asarray(proposal.hand_qpos).shape!=(26,) or not np.isfinite(proposal.hand_qpos).all():
        raise ValueError('input_invalid: no finite CA/geometric proposal')
    prepare_experiment(directory,hand,proposal,args)
    ids=[np.flatnonzero(hand.model.jnt_qposadr==i)[0] for i in range(26)]
    low,high=hand.model.jnt_range[ids].T;q=np.clip(proposal.hand_qpos,low,high)
    masks=hand.semantic_contact_masks;required=np.zeros(6,bool)
    for body in hand.bodies:required[finger_group(hand.model,body)]=True
    power=int(template.partition('_')[0]) in POWER_TYPES;required[5]=bool(required[5] or power)
    palm=palm_region(hand.model)
    values=dict(qpos=np.r_[q,0.,0.,0.,1.,0.,0.,0.],gesture_qpos=hand.ik_reference_qpos,
        bodies=hand.bodies,local=hand.local,normals=hand.normals,
        mask_offsets=np.r_[0,np.cumsum([len(m) for m in masks])],masks=np.concatenate(masks),
        labels=np.asarray(hand.semantic_patch_labels),required_groups=required,ca_active_mask=original_mask,
        palm_bodies=palm['bodies'],palm_local=palm['local'],palm_normals=palm['normals'])
    for name in ('pose_intent_qpos','pose_intent_center','intent_palm_position','intent_palm_rotation','cylinder_middle_slots','cylinder_middle_half_band'):
        if hasattr(hand,name):values[name]=np.asarray(getattr(hand,name))
    np.savez_compressed(directory/'spider_seed.npz',**values)
    write_json(directory/'spider_seed.json',dict(version=1,kind='ca_proposal',template=template,primitive=primitive,
        power=power,group_names=GROUPS,ca_success=guess.metadata.get('contactaware',{}).get('success',False),
        ca_diagnostics=guess.metadata.get('contactaware',{}),reference_source='unverified CA/geometric initial proposal',
        template_angles_are_tracking_target=False,joint_projection_max=float(np.max(abs(q-proposal.hand_qpos))),
        proposal_provenance=guess.metadata.get('source_experiment'),
        query_sha256={str(query_path(hand.model,body)):hashlib.sha256(query_path(hand.model,body).read_bytes()).hexdigest()
            for body in np.unique(hand.all_bodies)},scene='sequence/scene/wuji_right/primitive_sdf.xml'))
    return directory


def load_seed(source):
    source=Path(source).resolve();metadata=json.loads((source/'spider_seed.json').read_text())
    with np.load(source/'spider_seed.npz',allow_pickle=False) as raw:values={key:raw[key].copy() for key in raw.files}
    if values['qpos'].shape!=(33,) or not np.isfinite(values['qpos']).all():raise ValueError('input_invalid: SPIDER seed needs finite full qpos')
    if abs(np.linalg.norm(values['qpos'][29:])-1.)>1e-6:raise ValueError('input_invalid: invalid object quaternion')
    if values['gesture_qpos'].shape!=(26,) or not np.isfinite(values['gesture_qpos']).all():raise ValueError('input_invalid: malformed bone-direction seed')
    if values['required_groups'].shape!=(6,) or not values['required_groups'].any():raise ValueError('input_invalid: missing template contact requirements')
    count=len(values['bodies']);offsets=values['mask_offsets']
    if (values['local'].shape!=(count,3) or values['normals'].shape!=(count,3) or offsets.shape!=(count+1,) or offsets[0]!=0 or offsets[-1]!=len(values['masks']) or not np.isfinite(values['local']).all() or not np.isfinite(values['normals']).all()):
        raise ValueError('input_invalid: malformed semantic patch data')
    model=mujoco.MjModel.from_xml_path(str(source/metadata['scene']))
    if (model.nq,model.nv,model.nu)!=(33,32,26):raise ValueError('input_invalid: expected WUJI2 and one free object')
    for path,digest in metadata.get('query_sha256',{}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('input_invalid: semantic query assets changed')
    if (np.any(values['bodies']<1) or np.any(values['bodies']>=model.nbody) or
            values['palm_local'].shape!=(len(values['palm_bodies']),3) or
            not np.isfinite(values['palm_local']).all()):raise ValueError('input_invalid: malformed contact bodies/palm pool')
    for i,body in enumerate(values['bodies']):
        with np.load(query_path(model,int(body))) as raw:
            mask=values['masks'][offsets[i]:offsets[i+1]]
            if mask.shape!=(len(raw['local_pos']),) or not mask.any():raise ValueError('input_invalid: damaged or empty semantic patch')
    return source,metadata,values,model


def cached_proposal(work,mesh,template,primitive,args):
    """Reconstruct a recorded proposal without treating its CA flag as a gate."""
    from .inputs import build_scene,StaticHand
    from .initialization import GraspInitialization
    from .intent import configure_intent
    work=Path(work);case=next((work/'ca_seed').glob('**/semantic_matching.json')).parent
    matching=json.loads((case/'semantic_matching.json').read_text())
    initialization=json.loads((case/'initialization.json').read_text())[0]
    with np.load(case/'semantic_patches.npz',allow_pickle=False) as raw:
        masks=[raw[f'mask_{i:03d}'].copy() for i in range(len(matching['patch_labels']))]
    with np.load(case/'contactaware_reference.npz',allow_pickle=False) as raw:
        q=raw['proposed_hand_qpos'][0].copy();anchors=raw['proposed_object_anchors'][0].copy()
    model,surface,_,_=build_scene(mesh,hand_model='wuji2')
    bodies=np.asarray([model.body(r['wuji_body']).id for r in matching['records']])
    local=np.asarray([r['local'] for r in matching['fitted_contacts']]);normals=np.asarray([r['normal'] for r in matching['fitted_contacts']])
    hand=StaticHand(model,surface,bodies,local,normals)
    hand.semantic_contact_masks=masks;hand.semantic_patch_labels=matching['patch_labels']
    hand.semantic_contact_local=np.asarray(matching['reference_local']);hand.semantic_contact_normals=np.asarray(matching['reference_normals'])
    fit=initialization['template_fit']['numerical_seed_fit']
    hand.ik_reference_qpos=np.asarray(fit['reference_qpos']);hand.contact_reference_normals=np.asarray(fit['reference_local_normals'])
    configure_intent(hand,template,primitive,args)
    report=json.loads((case/'contactaware.json').read_text())[0]
    if q.shape!=(26,) or not np.isfinite(q).all() or not np.isfinite(anchors).all():
        q=np.asarray(report.get('initial_qpos',[]),float)
        if q.shape!=(26,) or not np.isfinite(q).all():
            raise ValueError('input_invalid: cached CA has no finite pre-CA geometric initialization')
        from retarget.env.geometry import closest_triangles
        hand.set(q);points=hand.points(hand.bodies,hand.local)
        candidates=closest_triangles(points,surface.triangles)
        indices=np.argmin(np.sum((candidates-points[:,None])**2,axis=-1),axis=1)
        anchors=candidates[np.arange(len(points)),indices]
        report={**report,'success':False,'fallback':'cached finite pre-CA geometric alignment after nonfinite CA proposal'}
    guess=GraspInitialization(q,anchors,np.zeros_like(anchors),{},dict(contactaware=report,
        source='cached finite CA proposal',source_experiment=str(case),initialization_valid=report.get('success',False)))
    guess.contact_mask=np.asarray(report.get('active_contact_mask',np.zeros(len(bodies))),bool)
    return hand,guess
