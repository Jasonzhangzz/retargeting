"""Geometry-aware candidates and explicit validity for whole_hand_v1 inputs."""

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .graspadmm_gesture import landmarks, attach_reference, reference_diagnostics
from .graspadmm_scene import from_upstream_qpos, to_upstream_qpos, normalize_grasp_data


class InitializationFailure(ValueError):
    def __init__(self, info):
        self.info = info
        super().__init__("No geometrically valid whole-hand initialization; see initialization.json")


def reanchor(hand,q,reference):
    """Prefer exposed annotated skin normals while fitting actual object faces."""
    from retarget.initialization.ik import nearest_sdf
    from .contact_regions import surface_pool
    hand.set(q)
    positions=[];normals=[]
    for i,body in enumerate(hand.bodies):
        name=hand.model.body(int(body)).name
        p,n,_=surface_pool(name)
        select=np.unique(np.r_[np.linspace(0,len(p)-1,min(512,len(p))).astype(int),
                              np.argmin(np.sum((p-hand.local[i])**2,axis=1))])
        p,n=p[select],n[select]
        rotation=hand.data.xmat[body].reshape(3,3)
        world=p@rotation.T+hand.data.xpos[body]
        _,outward,phi=nearest_sdf(hand.surface,world)
        facing=np.sum((n@rotation.T)*outward,axis=1)
        skin=n@np.asarray(reference['contact_normals_local'][i])
        score=((phi-.0007)/.005)**2+.35*(1+facing)**2+.2*(1-skin)**2
        best=int(np.argmin(score))
        positions.append(p[best]);normals.append(n[best])
    hand.local=np.asarray(positions);hand.normals=np.asarray(normals)


def collision_residual(hand, *, jacobian=False):
    """Real MuJoCo collisions, including nonparticipating fingers and pads."""
    model, data = hand.model, hand.data
    mujoco.mj_collision(model,data)
    object_id = model.body("grasp_object").id
    values = np.zeros((2,model.nbody))
    jac = np.zeros((2,model.nbody,26))
    for contact in data.contact:
        if contact.dist >= 0.:
            continue
        first,second = model.geom_bodyid[contact.geom]
        kind = int(object_id not in (first,second))
        owner = second if first == object_id else first
        depth = -contact.dist/.001
        if depth <= values[kind,owner]:
            continue
        values[kind,owner] = depth
        if jacobian:
            jp1,jp2 = np.zeros((3,model.nv)),np.zeros((3,model.nv))
            jr = np.zeros_like(jp1)
            mujoco.mj_jac(model,data,jp1,jr,contact.pos,int(first))
            mujoco.mj_jac(model,data,jp2,jr,contact.pos,int(second))
            jac[kind,owner] = -np.asarray(contact.frame[:3])@(jp2-jp1)[:,:26]/.001
    return values.flatten(),jac.reshape(-1,26)


def fit_candidate(hand, initial, reference, *, iterations=80):
    """Bounded whole-hand least squares against the actual triangle mesh.

    Discrete closest triangles and collision pairs are refreshed on every
    evaluation; the Jacobian uses their current tangent planes. No skin-region
    mask or hard posture/orientation constraint is introduced.
    """
    from retarget.initialization.ik import nearest_sdf
    from .inputs import skew

    base = np.asarray(initial,float).copy()
    ids,local = landmarks(hand.model)
    target_p = np.asarray(reference["landmark_positions_palm"])
    target_q = np.asarray(reference["hand_qpos"])
    reference_rotation = Rotation.from_euler("XYZ",base[3:6])
    joint_ids = [np.flatnonzero(hand.model.jnt_qposadr==i)[0] for i in range(26)]
    low,high = hand.model.jnt_range[joint_ids].T.copy()
    # Root rotations have periodic equivalents; use a continuous local chart.
    low[:6],high[:6] = base[:6]-np.r_[[.15]*3,[np.pi]*3],base[:6]+np.r_[[.15]*3,[np.pi]*3]
    cache = {}
    joint_weight=np.sqrt(np.asarray(reference.get('joint_weights',np.ones(20))))
    groups=[[i for i,name in enumerate(reference['hand_cbody']) if name in group]
            for group in hand.required_cbody]

    def evaluate(q):
        if "q" in cache and np.array_equal(q,cache["q"]):
            return cache["r"],cache["j"]
        hand.set(q)
        p,jp,jr = hand.points(hand.bodies,hand.local,True)
        n = np.einsum("nij,nj->ni",hand.data.xmat[hand.bodies].reshape(-1,3,3),hand.normals)
        projected,outward,_ = nearest_sdf(hand.surface,p)
        # Required bodies are unions: a multi-phalanx group does not require
        # every annotated slot to touch a small disk at the same time.
        contact_delta=p-projected-.0007*outward
        slot_score=np.sum((contact_delta/.005)**2,axis=-1)+.35*np.sum((n+outward)**2,axis=-1)
        slot_weight=np.full(len(p),.1)
        for group in groups:slot_weight[group[int(np.argmin(slot_score[group]))]]=1.
        sw=slot_weight[:,None]
        rotation = Rotation.from_euler("XYZ",q[3:6])
        lp,lj,_ = hand.points(ids,local,True)
        palm_points = rotation.inv().apply(lp-q[:3])
        lj = np.einsum("ij,njk->nik",rotation.as_matrix().T,lj)
        lj[:,:,:6] = 0.
        angular = (rotation*reference_rotation.inv()).as_rotvec()
        angular_jac = np.zeros((3,26))
        for k in range(3):
            shifted=q[3:6].copy();shifted[k]+=1.e-6
            angular_jac[:,3+k]=((Rotation.from_euler("XYZ",shifted)*reference_rotation.inv()).as_rotvec()-angular)/1.e-6
        collision,cj = collision_residual(hand,jacobian=True)
        r = np.r_[(sw*contact_delta/.005).flatten(),(sw*.35*(n+outward)).flatten(),
                  np.sqrt(.05)*joint_weight*(q[6:]-target_q[6:])/.5,
                  (np.sqrt(.1)*(palm_points-target_p)/.02).flatten(),
                  np.sqrt(.05)*angular/.5,10*collision]
        prior_jac = np.zeros((20,26));prior_jac[:,6:]=np.diag(joint_weight)*np.sqrt(.05)/.5
        j = np.concatenate(((jp*sw[:,:,None]).reshape(-1,26)/.005,
             (-.35*sw[:,:,None]*(skew(n)@jr)).reshape(-1,26),prior_jac,
             (np.sqrt(.1)*lj/.02).reshape(-1,26),np.sqrt(.05)*angular_jac/.5,10*cj))
        cache.update(q=q.copy(),r=r,j=j)
        return r,j

    start = np.clip(base,low+1.e-7,high-1.e-7)
    before = float(np.sum(evaluate(start)[0]**2))
    result = least_squares(lambda q:evaluate(q)[0],start,jac=lambda q:evaluate(q)[1],
        bounds=(low+1.e-8,high-1.e-8),max_nfev=iterations,x_scale=np.r_[[.02]*3,[.3]*23],
        ftol=1.e-5,xtol=1.e-5,gtol=1.e-5)
    hand.set(result.x)
    depths,_ = collision_residual(hand)
    return result.x,dict(initial_cost=before,final_cost=float(2*result.cost),evaluations=result.nfev,
        max_penetration_m=float(depths.max()*.001),
        **reference_diagnostics(result.x,reference))


def candidate_data(hand,q,template,scene):
    from retarget.initialization.ik import nearest_sdf
    hand.set(q)
    p=hand.points(hand.bodies,hand.local)
    n=np.einsum("nij,nj->ni",hand.data.xmat[hand.bodies].reshape(-1,3,3),hand.normals)
    projected,outward,_=nearest_sdf(hand.surface,p)
    return normalize_grasp_data(q,scene,hand_cbody=template["hand_cbody"],hand_cpn_w=np.c_[p,n],
        obj_cpn_w=np.c_[projected,outward],required_cbody=template["required_cbody"],template=template["tmpl_name"])


def validate_candidates(hand,candidates,args,scene):
    """Check native initialization and native static-grasp validity separately.

    A proven static grasp can fail the template-slot normal predicate: native
    static filtering uses actual mesh contacts and required-body groups. Either
    complete native gate may accept a seed; padding itself never accepts it.
    """
    import torch
    from .graspadmm_source import require_upstream
    from .graspadmm_inputs import skeleton_arrays
    from .initialization import matching_mesh

    source=require_upstream();device=torch.device(args.init_device)
    cfg=copy.deepcopy(source.init_config.filter)
    collision=source.init.WarpCollisionEnv(device)
    collision.load(None,[(str(scene.mesh_path),matching_mesh(hand.surface))])
    native=source.init.InitFilter(cfg,device,collision)
    from .graspadmm_scene import WujiOptEnv
    env=WujiOptEnv(source.sim.HandCfg(str(scene.hand_xml),freejoint=True),scene.scene_cfg,
        source.sim.MuJoCo_OptCfg(obj_margin=.002),debug_view=False,debug_render=False)
    static_filter=source.core.GraspFilter(source.grasp_config.grasp.filter,env,"gesture initialization static check")
    rows=[]
    for data in candidates:
        q=from_upstream_qpos(data["grasp_qpos"][0]);hand.set(q)
        # skeleton_arrays resets to zero only to find immutable endpoints.
        segments,bodies=skeleton_arrays(hand);hand.set(q)
        endpoint=segments.reshape(-1,2,3)
        r=Rotation.from_euler("XYZ",q[3:6])
        skt=np.einsum("nij,nkj->nki",hand.data.xmat[bodies].reshape(-1,3,3),endpoint)+hand.data.xpos[bodies,None,:]
        skt=r.inv().apply(skt.reshape(-1,3)-q[:3])
        hp,op=data["hand_cpn_w"],data["obj_cpn_w"]
        raw=dict(obj_id=np.zeros(1,np.int64),grasp_pose=data["grasp_qpos"][:,:7],
            obj_pose=np.array([[0.,0.,0.,1.,0.,0.,0.]]),obj_scale=np.ones((1,3)),
            hand_skt_h=skt[None],obj_cp_w=op[None,:,:3],obj_cn_w=-op[None,:,3:],
            ext_wrench=data["ext_wrench"][None],ext_center=data["ext_center"][None],
            loss_dist=np.sum((hp[:,:3]-op[:,:3])**2,axis=-1)[None],
            loss_normal=np.sum((hp[:,3:]+op[:,3:])**2,axis=-1)[None])
        tensors={key:torch.as_tensor(value,device=device,dtype=torch.long if key=="obj_id" else torch.float32) for key,value in raw.items()}
        checks={name:bool(getattr(native,f"_{name}_filter")(tensors)[0]) for name in native.valid_filter if name!="fps"}
        depth,_=collision_residual(hand)
        checks["mesh_clearance"] = bool(depth.max()*.001 <= .001)
        env.reset_qpos(data['grasp_qpos'][0])
        static_valid,actual=static_filter.forward(data)
        # Static-contact geometry is a second native gate, independent of
        # annotation-slot normals. Initialization permits <=1mm mesh overlap;
        # the subsequent native grasp filter still requires zero penetration.
        contact_valid=static_filter._contact_filter(actual,data['required_cbody'])
        actual_qp=bool(contact_valid and static_filter._qp_filter(actual,data['ext_wrench'],data['ext_center']))
        actual_geometry=dict(required_contacts=bool(contact_valid),qp=actual_qp,
            joint_limits=bool(static_filter._limit_filter()),mesh_clearance=checks['mesh_clearance'],
            skeleton_collision=checks.get('collision',True))
        rows.append(dict(checks=checks,native_initialization_valid=all(checks.values()),
            native_static_grasp_valid=bool(static_valid),actual_contact_bodies=actual.get('bn1',[]),
            actual_geometry_checks=actual_geometry,valid=bool(all(checks.values()) or all(actual_geometry.values())),
            max_contact_gap_m=float(np.sqrt(raw["loss_dist"]).max()),
            max_normal_squared_error=float(raw["loss_normal"].max()),
            max_penetration_m=float(depth.max()*.001)))
    return rows


def initialize_gesture(hand,template_q,template,native,args,scene,directory):
    from .orientation import configure_orientation,primitive_pose_seeds,aligned_disk_seeds

    reference=copy.deepcopy(template["gesture_reference"])
    hand.required_cbody=template['required_cbody']
    primitive=args.primitive_spec
    number=int(template["tmpl_name"].partition("_")[0])
    hand.ik_reference_qpos=template_q.copy()
    hand.contact_reference_normals=hand.normals.copy()
    configure_orientation(hand,template["tmpl_name"],primitive)
    proposals=[("native_matcher",from_upstream_qpos(native["grasp_qpos"][0]))]
    if number==12:
        proposals=[("approved_1x_reference",np.asarray(reference["hand_qpos"]).copy())]
    else:
        seeds=aligned_disk_seeds(hand,template_q) if hasattr(hand,"object_orientation") else primitive_pose_seeds(hand,template_q)
        if number in (23,31):
            hand.set(np.r_[np.zeros(6),template_q[6:]])
            if number==23:
                axis=hand.data.xpos[hand.model.body('right_finger2_link4').id]-hand.data.xpos[hand.model.body('right_finger2_link3').id]
            else:
                axis=hand.model.body('right_finger2_link1').pos-hand.model.body('right_finger5_link1').pos
            axis/=np.linalg.norm(axis)
            center=hand.points(hand.bodies,hand.local).mean(0)
            rotation,_=Rotation.align_vectors([[0.,0.,1.]],[axis])
            seeds=np.array([np.r_[-rotation.apply(center),rotation.as_euler('XYZ'),template_q[6:]]])
        # Axial primitives are invariant to world yaw and axis sign. Keep the
        # first unique palm-frame placements, rather than fitting duplicates.
        unique=[]
        for q in seeds:
            rotation=Rotation.from_euler("XYZ",q[3:6])
            signature=np.r_[rotation.inv().apply([0.,0.,1.]),-rotation.inv().apply(q[:3])]
            if not any(np.linalg.norm(signature-s)<1.e-5 for s in unique):
                unique.append(signature);proposals.append(("annotated_axis_candidate",q))
            if len(proposals)>=4:break
    directory=Path(directory)
    records=[];data=[]
    for label,proposal in proposals:
        hand.local=np.asarray(reference['contact_local']).copy()
        hand.normals=np.asarray(reference['contact_normals_local']).copy()
        reanchor(hand,proposal,reference)
        # Keep the exact approved 1x pose as a candidate as well as any adapted
        # pose. Other sizes change fingers/position, never the physical mesh.
        if number==12 and np.isclose(primitive["radius_m"],.045) and np.isclose(primitive["height_m"],.016):
            fitted=proposal;fit=dict(exact_approved_reference=True,final_cost=0.)
        else:
            fitted,fit=fit_candidate(hand,proposal,reference)
            for _ in range(2):
                reanchor(hand,fitted,reference)
                fitted,fit=fit_candidate(hand,fitted,reference,iterations=40)
        data.append(candidate_data(hand,fitted,template,scene))
        records.append(dict(source=label,fit=fit,grasp_qpos=to_upstream_qpos(fitted).tolist()))
    validity=validate_candidates(hand,data,args,scene)
    for record,valid in zip(records,validity):record.update(valid)
    valid=[i for i,row in enumerate(records) if row["valid"]]
    chosen=min(valid,key=lambda i:records[i]["fit"]["final_cost"]) if valid else None
    info=dict(policy="whole_hand_v1",candidates=records,selected=chosen,valid_candidates=len(valid),
        padding_implies_validity=False,failure_resampling=False,semantic_status="pending_user_review")
    np.save(directory/"gesture_candidates.npy",np.asarray(data,dtype=object))
    (directory/"gesture_initialization.json").write_text(json.dumps(info,indent=2))
    if chosen is None:
        raise InitializationFailure(info)
    result=attach_reference(data[chosen],reference,data[chosen]["grasp_qpos"][0])
    if template.get("contact_regions") is not None:result["contact_regions"]=copy.deepcopy(template["contact_regions"])
    return result,info
