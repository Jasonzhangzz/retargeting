"""Shared metrics for current native CARD CEM feasibility checks."""
import torch
from audit_reach_trajectory_cem import (WindowCost, FloatingCartHand, ReachObjective,
    PalmAlignedNetwork, unbounded_wrist_alignment, load_sessions, attach_ik,
    attach_card_targets, windows, prepare_windows, CardSettings, cem, SPEC)
from moxian_web.cart_wrist import so3_exp,state_velocity
from moxian_web.cart_palm_alignment import rotation_vector

# Jitter is zero in the current objective; names retain audit-result compatibility.
VARIANTS = dict(full=(1,1), no_smooth=(0,1), no_jitter=(1,0), contact_only=(0,0))

def selected_cost(parts, variant):
    smooth, jitter = VARIANTS[variant]
    return parts['contact'] + smooth*parts['motion_jitter'] + jitter*parts['contact_jitter']


def metrics(evaluator, states, variant):
    parts, states, gaps = evaluator.evaluate_states(states)
    active = gaps[:,0][evaluator.mask]*1000
    frames = []
    for t in range(len(states)):
        g=gaps[t,0][evaluator.mask[t]]*1000
        frames.append(dict(frame=t, anchors=len(g), mean_mm=float(g.mean()) if len(g) else None,
            max_mm=float(g.max()) if len(g) else None, all_within_1mm=bool((g<1).all()) if len(g) else None))
    p={k:float(v[0]) for k,v in parts.items()}
    # Physical motion is reported independently of the optimized cost. Never
    # infer smooth motion from small contact gaps alone.
    dt=evaluator.dt[1:]
    velocity=state_velocity(states[1:,0],states[:-1,0],dt)
    step=velocity*dt[:,None]
    second=(velocity[1:]-velocity[:-1])/(.5*(dt[1:]+dt[:-1]))[:,None]/120**2
    motion=dict(finger_step_rad=step[:,:20].abs().flatten().cpu().tolist(),
        wrist_step_mm=(step[:,20:23].norm(dim=-1)*1000).cpu().tolist(),
        wrist_rotation_step_rad=step[:,23:].norm(dim=-1).cpu().tolist(),
        finger_second_rad=second[:,:20].abs().flatten().cpu().tolist(),
        wrist_second_mm=(second[:,20:23].norm(dim=-1)*1000).cpu().tolist(),
        rotation_second_rad=second[:,23:].norm(dim=-1).cpu().tolist())
    return dict(**p, objective=float(selected_cost(parts,variant)[0]),
        mean_mm=float(active.mean()),p95_mm=float(torch.quantile(active,.95)),max_mm=float(active.max()),
        within_1mm=float((active<1).float().mean()), anchors=len(active), gaps_mm=active.cpu().tolist(),
        frames=frames,states=states[:,0].cpu().tolist(),motion=motion)
