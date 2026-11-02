"""Versioned whole-hand references for the local GraspADMM input adapter.

References are independent of this run's ADMM/MPPI output. All posture and
orientation guidance is soft; the native physical acceptance test is separate.
"""

import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

TYPES = frozenset((9, 10, 12, 22, 23, 30, 31))
POLICY = "whole_hand_v1"


def enabled(args, template):
    return (getattr(args, "gesture_policy", "legacy") == POLICY
            and int(str(template).partition("_")[0]) in TYPES
            and getattr(args, "primitive_seed_policy", "taxonomy") != "upstream")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def landmarks(model):
    bodies, points = [], []
    for f, name in enumerate(("thumb", "index", "middle", "ring", "pinky"), 1):
        ids = [model.body(f"right_finger{f}_link{k}").id for k in (2, 3, 4)]
        bodies.extend([*ids, ids[-1]])
        points.extend([np.zeros(3), np.zeros(3), np.zeros(3), model.site(f"right_{name}_tip").pos])
    return np.asarray(bodies), np.asarray(points)


def whole_hand_seed(model, joints, baseline, shadow, posed, contacts, bodies, local_normals, device):
    """Jointly fit all twenty joints, including cross-finger positions/normals."""
    import torch
    from .template import WujiContactKinematics

    human = np.asarray(joints)[0, 1:]
    base_rotation = Rotation.from_euler("XYZ", baseline[3:6]).as_matrix()
    ids, points = landmarks(model)
    fk = WujiContactKinematics(model, np.r_[ids, bodies], device)
    positions = fk.tensor(np.r_[points, np.zeros((len(bodies), 3))])
    normals = fk.tensor(np.r_[np.zeros_like(points), local_normals])
    root = fk.tensor(np.zeros(6))
    direction = np.diff(human.reshape(5, 4, 3), axis=1) @ base_rotation
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    target_d = fk.tensor(direction)
    # Normalize the cross-finger geometry by the index/little MCP span. Bone
    # directions alone cannot encode opposition, lateral pinches or loops.
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    robot_width = np.linalg.norm(data.xpos[ids[4]] - data.xpos[ids[16]])
    source_width = np.linalg.norm(human[4] - human[16])
    if min(robot_width, source_width) < 1.e-6:
        raise ValueError("Degenerate palm width in whole-hand reference")
    selected = np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19])
    a, b = np.triu_indices(len(selected), 1)
    a, b = selected[a], selected[b]
    target_pair = fk.tensor((human[a] - human[b]) @ base_rotation / source_width)
    target_normal = []
    for body, contact in contacts:
        rotation = posed.xmat[shadow.body(body).id].reshape(3, 3)
        target_normal.append(contact[3:] @ rotation.T @ base_rotation)
    target_normal = fk.tensor(target_normal)
    prior = fk.tensor(baseline[6:])
    joint_ids = [np.flatnonzero(model.jnt_qposadr == i)[0] for i in range(6, 26)]
    low, high = (fk.tensor(v) for v in model.jnt_range[joint_ids].T)
    rng = np.random.default_rng(0)
    starts = np.stack([baseline[6:], baseline[6:] + rng.normal(0, .2, 20),
                       baseline[6:] + rng.normal(0, .4, 20), (low.cpu().numpy()+high.cpu().numpy())/2])
    q = fk.tensor(starts).clamp(low, high)

    def residual(value):
        p, n = fk.contacts(torch.cat((root, value)), positions, normals)
        vector = p[:20].reshape(5, 4, 3).diff(dim=1)
        unit = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1.e-12)
        return torch.cat(((unit-target_d).flatten()/np.sqrt(45),
                          .75*((p[a]-p[b])/robot_width-target_pair).flatten()/np.sqrt(len(a)*3),
                          .35*(n[20:]-target_normal).flatten()/np.sqrt(max(1,len(bodies)*3)),
                          .02*(value-prior)/np.sqrt(20)))

    res = torch.vmap(residual)
    jac = torch.vmap(torch.func.jacfwd(residual))
    damping = torch.full((len(q),), .01, device=fk.device, dtype=fk.dtype)
    eye = torch.eye(20, device=fk.device, dtype=fk.dtype)
    before = float(residual(prior).square().sum())
    for _ in range(64):
        r, j = res(q), jac(q)
        jt = j.transpose(-1,-2)
        delta = torch.linalg.solve_ex(jt@j+damping[:,None,None]*eye, -(jt@r[...,None]))[0][...,0]
        trial = (q+delta.clamp(-.2,.2)).clamp(low,high)
        accept = torch.isfinite(trial).all(-1) & (res(trial).square().sum(-1)<r.square().sum(-1))
        q = torch.where(accept[:,None],trial,q)
        damping = torch.where(accept,damping*.5,damping*4).clamp(1.e-8,1.e8)
    costs = res(q).square().sum(-1)
    index = int(costs.argmin())
    fitted = baseline.copy()
    fitted[6:] = q[index].detach().cpu().numpy()
    return fitted, dict(policy=POLICY, iterations=64, starts=len(starts), selected=index,
        initial_cost=before, final_cost=float(costs[index]),
        terms=dict(bone_direction=1., relative_finger_position=.75, contact_normal=.35, joint_prior=.02),
        root_fixed=True, all_twenty_joints_fitted_together=True, soft_reference=True)


def snapshot_type12(package, target):
    """Freeze the exact user-approved selected pose; never infer it by index."""
    package, target = Path(package), Path(target)
    case_id = "t12_disk_s04_seed00"
    case = package/"cases"/f"{case_id}.json"
    arrays = package/"candidates.npz"
    record = json.loads(case.read_text())
    with np.load(arrays, allow_pickle=False) as values:
        ids = values["case_ids"].astype(str).tolist()
        q = values["selected_qpos"][ids.index(case_id)].astype(float)
        is_mppi = np.array_equal(q, values["mppi_qpos"][ids.index(case_id)])
    if not is_mppi or q.shape != (27,) or not np.isfinite(q).all():
        raise ValueError("Approved type12 source must be the selected MPPI pose")
    rotation = Rotation.from_quat(q[3:7], scalar_first=True)
    result = dict(version=1, type_id=12, case_id=case_id, method="graspadmm_mppi",
        policy="native_validation_fallback", selected_stage="mppi", grasp_qpos=q.tolist(),
        primitive=dict(kind="disk", radius_m=.045, height_m=.016, size_scale=1.),
        object_center_palm=(-rotation.inv().apply(q[:3])).tolist(),
        object_rotation_palm=rotation.inv().as_matrix().tolist(),
        source_files={str(p.resolve()):digest(p) for p in (case,arrays)}, source_record=record)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2))
    return result


def load_type12(args):
    root = getattr(args,"gesture_reference_root",None)
    if root is None:
        raise ValueError("whole_hand_v1 type12 requires a frozen gesture_reference_root/type12.json")
    path = Path(root)/"type12.json"
    result = json.loads(path.read_text())
    if result.get("case_id") != "t12_disk_s04_seed00" or result.get("selected_stage") != "mppi":
        raise ValueError("Wrong approved type12 reference")
    return result, dict(path=str(path.resolve()),sha256=digest(path),**result)


def make_reference(hand, q, template, source):
    hand.set(np.r_[np.zeros(6),q[6:]])
    ids, points = landmarks(hand.model)
    positions = hand.points(ids,points)
    vector=np.diff(positions.reshape(5,4,3),axis=1)
    unit=vector/np.linalg.norm(vector,axis=-1,keepdims=True)
    straight=np.sum(unit[:,0]*unit[:,-1],axis=-1)>.9
    joint_weights=np.ones((5,4))
    for finger in range(1,5):
        if straight[finger]:joint_weights[finger,2:]=25.
    normal = np.einsum("nij,nj->ni",hand.data.xmat[hand.bodies].reshape(-1,3,3),hand.normals)
    return dict(version=1,policy=POLICY,template=template,hand_qpos=np.asarray(q).tolist(),
        orientation_source='approved_snapshot' if int(template.partition('_')[0])==12 else 'initialization_candidate_pending_review',
        landmark_positions_palm=positions.tolist(),hand_cbody=[hand.model.body(int(b)).name for b in hand.bodies],
        contact_local=hand.local.tolist(),contact_normals_local=hand.normals.tolist(),
        contact_normals_palm=normal.tolist(),source=source,
        joint_weights=joint_weights.flatten().tolist(),
        weights=dict(joint=.05,landmark=.1,normal=.1,orientation=.05,skin=.2),
        joint_scale_rad=.5,position_scale_m=.02,orientation_scale_rad=.5,
        acceptance="pending_user_gesture_review")


def reference_diagnostics(q, reference):
    from .graspadmm_scene import from_upstream_qpos
    value = np.asarray(q,float)
    if value.shape == (27,):
        value = from_upstream_qpos(value)
    target = np.asarray(reference["hand_qpos"])
    angle = (Rotation.from_euler("XYZ",value[3:6])*Rotation.from_euler("XYZ",target[3:6]).inv()).magnitude()
    return dict(reference_policy=reference["policy"],semantic_status="pending_user_review",
        joint_rms_rad=float(np.sqrt(np.mean((value[6:]-target[6:])**2))),
        max_joint_delta_rad=float(np.max(np.abs(value[6:]-target[6:]))),
        orientation_error_deg=float(np.degrees(angle)))


def attach_reference(data, reference, q):
    from .graspadmm_scene import from_upstream_qpos
    result = copy.deepcopy(data)
    ref = copy.deepcopy(reference)
    # The orientation/position target is established by initialization, while
    # canonical finger shape is independent of the optimizer's output.
    local_q=from_upstream_qpos(np.asarray(q))
    ref["hand_qpos"][:3] = local_q[:3].tolist()
    if ref.get('orientation_source')!='approved_snapshot':
        ref["hand_qpos"][3:6] = local_q[3:6].tolist()
    result["gesture_reference"] = ref
    result.pop("reference_orientation_wxyz",None)
    return result


def install_soft_reference(env, reference):
    """Add the reference potential's forces on each optimization substep only."""
    model,data=env._model,env._data
    target=np.asarray(reference["hand_qpos"],float)
    target_r=Rotation.from_euler("XYZ",target[3:6])
    weights=reference["weights"]
    prefix=env._cfg.hand_prefix
    names=reference["hand_cbody"]
    bodies=np.array([model.body(prefix+name).id for name in names])
    local=np.asarray(reference["contact_normals_local"])
    target_normals=np.asarray(reference["contact_normals_palm"])
    lm_bodies=[];lm_local=[]
    for f,name in enumerate(("thumb","index","middle","ring","pinky"),1):
        ids=[model.body(prefix+f"right_finger{f}_link{k}").id for k in (2,3,4)]
        site=model.site(prefix+f"right_{name}_tip")
        lm_bodies.extend([*ids,ids[-1]])
        lm_local.extend([np.zeros(3),np.zeros(3),np.zeros(3),site.pos])
    lm_target=np.asarray(reference["landmark_positions_palm"])
    original=env.step_sim

    def step(substep):
        for _ in range(substep):
            q=env.get_hand_qpos()
            rotation=Rotation.from_quat(q[3:7],scalar_first=True)
            force=np.zeros(model.nv)
            scale=.1
            force[6:26]-=scale*weights["joint"]*np.asarray(reference.get('joint_weights',np.ones(20)))*(q[7:]-target[6:])/reference["joint_scale_rad"]**2
            force[3:6]-=scale*weights["orientation"]*rotation.inv().apply((rotation*target_r.inv()).as_rotvec())/reference["orientation_scale_rad"]**2
            for body,point,goal in zip(lm_bodies,lm_local,lm_target):
                world=data.xpos[body]+data.xmat[body].reshape(3,3)@point
                desired=q[:3]+rotation.apply(goal)
                jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
                mujoco.mj_jac(model,data,jp,jr,world,int(body));jp[:,:6]=0
                force-=scale*weights["landmark"]*jp.T@(world-desired)/(len(lm_bodies)*reference["position_scale_m"]**2)
            for body,normal,goal in zip(bodies,local,target_normals):
                current=data.xmat[body].reshape(3,3)@normal
                desired=rotation.apply(goal)
                jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
                mujoco.mj_jacBody(model,data,jp,jr,int(body));jr[:,:6]=0
                force+=scale*weights["normal"]*jr.T@np.cross(current,desired)/max(1,len(bodies))
            data.qfrc_applied[:]+=force
            original(1)
            data.qfrc_applied[:]-=force

    env.step_sim=step
