"""AdamW's bounded mesh/wrench contact subproblem, with measured SQP input.

The surface solver is shared. AdamW supplies rollout x+dual; SQP supplies
current measured x and zero dual. Equal subproblem inputs give equal outputs;
independent closed-loop trajectories do not generally have equal contacts.
"""
from dataclasses import dataclass
import mujoco
import numpy as np
from retarget.trigrasp import quat_wxyz_to_matrix
from scipy.spatial.transform import Rotation


@dataclass
class ContactPlan:
    points: np.ndarray
    diagnostics: list
    snapshot: dict


def optimize_contact_plan(model, index, solver, qpos, targets, *, friction,
                          center, object_radius, hand_points_object=None, query_enabled=None):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    count = len(targets.dense_actions)
    rows, diagnostics, snapshots = [], [], []
    rotation = data.xmat[index.object_body].reshape(3, 3)
    for stage in range(count + 1):
        contact_stage = max(stage-1, 0) if stage < count else count-1
        template = np.asarray(targets.contact_position_object[contact_stage], np.float32)
        body = np.asarray(targets.query_body[contact_stage], int)
        local = np.asarray(targets.query_local[contact_stage], float)
        weight = (np.asarray(targets.contact_mask[contact_stage], np.float32)
                  * np.asarray(targets.contact_weight[contact_stage], np.float32))
        hand = data.xpos[body] + np.einsum('nij,nj->ni', data.xmat[body].reshape(-1, 3, 3), local)
        measured = ((hand-data.xpos[index.object_body]) @ rotation).astype(np.float32)
        x = measured if hand_points_object is None else hand_points_object[stage]
        enabled = stage > 0 if query_enabled is None else query_enabled[stage]
        if solver is None or not enabled or not np.any(weight > 0):
            points, diagnostic = template.copy(), dict(method='fixed_template')
        else:
            extra = {}
            if hasattr(solver, 'solve_batch'):
                pose = np.asarray(targets.object_qpos[min(stage,count-1)],float)
                gravity = solver.mass*(rotation.T@np.asarray(model.opt.gravity))
                goal = np.r_[rotation.T@(pose[:3]-data.xpos[index.object_body]),
                             Rotation.from_matrix(rotation.T@quat_wxyz_to_matrix(pose[3:7])).as_rotvec()]
                extra['context'] = dict(external=np.r_[gravity,np.cross(center,gravity)],goal=goal)
            points, diagnostic = solver.solve(template, template, x, weight, friction[stage],
                                               center=center, object_radius=object_radius,**extra)
        diagnostic.update(contact_input='measured' if hand_points_object is None else 'provided',
                          surface_subproblem_shared_with_adamw=True,
                          full_adamw_alternation=False)
        rows.append(points); diagnostics.append(diagnostic)
        snapshots.append(dict(c=points, z=points, c_before_surface_polish=template,
                              template=template, body=body, hand_local=local,
                              template_hand_local=local, mask=weight, x=x,
                              dual=np.zeros_like(points)))
    return ContactPlan(np.asarray(rows), diagnostics,
                       {key: np.stack([s[key] for s in snapshots[1:]]) for key in snapshots[0]})
