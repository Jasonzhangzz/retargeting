"""Compare recorded IK, held states and policy rollouts using native geometry.

Read-only: never preprocess data, update either model or launch training/Tube.
Reports gradient scales separately from scalar loss scales.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moxian_web.cart_data import load_sessions
from moxian_web.cart_initialization import attach_ik
from moxian_web.cart_training import windows, predict_trajectory, quality_summary
from moxian_web.cart_fast import PreparedBatch, prepare_windows, group_normalizers
from moxian_web.cart_objective import CartHand, CartObjective, policy_settings
from moxian_web.cart_collision import CollisionModel, frozen
from moxian_web.cart_network import CartNetwork
from moxian_web.cart_pretrained_sdf import restore_geometry
from moxian_web.cart_contacts import link_owner
from moxian_web.cart_contact_field import ContactTeacher


def scalar_parts(parts):
    return {key: float(value.detach()) for key, value in parts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/moxian/cart'))
    parser.add_argument('--dataset', default='20261118T135558_05a2628d')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--load-policy', action='store_true')
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--optimize-steps', type=int, default=0)
    parser.add_argument('--optimization-geometry', choices=('learned', 'native-contact'), default='learned',
                        help='Native-contact uses fresh mesh witnesses with FK gradients; SDF penetration stays learned')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(714)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    sessions, _ = load_sessions(args.root, [args.dataset], 'validation')
    attach_ik(args.root, sessions)
    all_windows = windows(sessions, 16)
    chosen = [all_windows[i] for i in np.linspace(0, len(all_windows)-1, args.windows).astype(int)]
    prepare_windows(sessions, chosen, args.device)
    batch = PreparedBatch(chosen, args.device, group_normalizers(chosen))
    hand = CartHand(args.device)
    collision = CollisionModel(hand, restore_geometry(hand, config.get('sdf_backend', {'kind':'link_sdf_v1'})))
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    collision.model.load_state_dict(state['collision'])
    objective = CartObjective(hand, policy_settings(), collision)
    objective.cache_rows = True
    model = CartNetwork(hand.low, hand.high, config['architecture_name'],
                        position_mean=config['position_mean'], position_scale=config['position_scale']).to(args.device)
    if args.load_policy:
        if config.get('policy_output') != 'normalized_delta_q_v1':
            raise ValueError('Audit requires a delta policy')
        model.load_state_dict(state['model'])
    model.eval()
    starts = [len(w['rows'])-w['length']-10 for w in chosen]
    row_steps = [[w['rows'][i+t+1] for w, i in zip(chosen, starts)] for t in range(16)]
    previous_steps = [[w['rows'][i+t] for w, i in zip(chosen, starts)] for t in range(16)]
    rows = [row for step in row_steps for row in step]
    previous = [row for step in previous_steps for row in step]
    weights = [batch.normalizers[t] for t in range(16) for _ in chosen]
    ik = hand.tensor([row['ik_q_rad'] for row in rows])
    prior_ik = hand.tensor([row['ik_q_rad'] for row in previous])
    initial = torch.stack([w['initial_q'] for w in chosen])
    held = initial.repeat(16, 1)
    with torch.no_grad():
        trajectory = predict_trajectory(model, objective, batch)
        predicted = torch.cat(trajectory.predictions)

    report = dict(dataset=args.dataset, checkpoint=str(args.checkpoint), checkpoint_epoch=state.get('epoch'),
                  policy_loaded=args.load_policy, windows=len(chosen), frames=len(rows),
                  settings=asdict(objective.settings), cases={})
    def evaluate(q, refs, *, audit):
        collision.auditing = audit
        with torch.no_grad():
            loss, parts, quality = objective.batch(q, rows, refs, previous_rows=previous,
                    audit=audit, normalizers=weights)
        collision.auditing = False
        return dict(total=float(loss), parts=scalar_parts(parts), metrics=quality_summary(quality))

    for name, q, refs in [('per_frame_ik', ik, prior_ik), ('held_initial_ik', held, held),
                           ('policy_rollout', predicted, torch.cat([x[2] for x in trajectory.pending]))]:
        report['cases'][name] = dict(learned=evaluate(q, refs, audit=False), native=evaluate(q, refs, audit=True),
                mean_q_error_to_ik_rad=float((q-ik).abs().mean()),
                at_joint_limit_fraction=float(((q-hand.low<1.e-5)|(hand.high-q<1.e-5)).float().mean()))
        print(json.dumps(dict(case=name, result=report['cases'][name])), flush=True)

    q = ik.detach().clone().requires_grad_()
    with frozen(collision.model):
        loss, parts, _ = objective.batch(q, rows, prior_ik, previous_rows=previous,
                            normalizers=weights, collect_metrics=False)
        settings=objective.settings
        multipliers = dict(direction=1., smooth=1., contact=settings.contact, normal=settings.normal,
                           penetration=settings.penetration, contact_keep=settings.contact_keep)
        gradients = {}
        for key, multiplier in multipliers.items():
            gradient = torch.autograd.grad(parts[key]*multiplier, q, retain_graph=True)[0]
            gradients[key] = gradient.detach()
        report['weighted_q_gradients'] = {key: dict(l2=float(g.norm()), max_abs=float(g.abs().max()),
                contribution=float(parts[key].detach())*multipliers[key]) for key, g in gradients.items()}
        g = gradients['contact'] + gradients['penetration']
        report['direction_vs_contact_penetration_gradient_cosine'] = float(torch.nn.functional.cosine_similarity(
                        gradients['direction'].flatten(), g.flatten(), dim=0))

    # Directly test whether the learned contact field sends joint gradients in
    # a native-mesh descent direction. Use several perturbation sizes so mesh
    # witness switches / FP32 noise are visible rather than hidden.
    direction = gradients['contact']
    direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1.e-12)
    report['contact_gradient_native_directional_difference'] = []
    for eps in (1.e-4, 1.e-3, 1.e-2):
        plus = evaluate((ik+eps*direction).clamp(hand.low,hand.high), prior_ik, audit=True)
        minus = evaluate((ik-eps*direction).clamp(hand.low,hand.high), prior_ik, audit=True)
        report['contact_gradient_native_directional_difference'].append(dict(eps_rad=eps,
            native_contact_plus=plus['parts']['contact'], native_contact_minus=minus['parts']['contact'],
            descent_improves_native_contact=minus['parts']['contact'] < plus['parts']['contact']))

    # Compare source geometry to the retargeted robot before network learning.
    source = hand.tensor([row['joints_m'] for row in rows])
    robot = hand.frames(ik)[2]
    report['source_vs_robot_ik'] = dict(source_tip_radius_mean_mm=float(source[:,[4,8,12,16,20]].norm(dim=-1).mean()*1000),
        robot_tip_radius_mean_mm=float(robot[:,[4,8,12,16,20]].norm(dim=-1).mean()*1000),
        tip_position_mean_mm=float((source[:,[4,8,12,16,20]]-robot[:,[4,8,12,16,20]]).norm(dim=-1).mean()*1000))

    # A positional lower bound: revolute finger joints cannot move a point
    # farther from the first joint pivot than their complete downstream length.
    p,r,*_ = hand.frames(ik)
    unreachable=[]; all_points=0
    for i,row in enumerate(rows):
        for c in row['contacts']['object_contacts']:
            owner=link_owner(c['region']);base=1+4*((owner-1)//4) if owner else 0
            if not owner:continue
            chain=list(range(base,owner+1)); reach=0.
            for a,b in zip(chain,chain[1:]):reach+=float((p[i,a]-p[i,b]).norm())
            vertices=hand.tensor(hand.surface.meshes[owner].vertices)
            reach+=float(vertices.norm(dim=-1).max())
            distance=float((hand.tensor(c['position_m'])-p[i,base]).norm())
            unreachable.append(max(0.,distance-reach)*1000); all_points+=1
    report['object_target_reach_lower_bound'] = dict(samples=all_points,
        provably_unreachable_fraction=float(np.mean(np.asarray(unreachable)>.5)) if unreachable else None,
        p95_lower_bound_mm=float(np.quantile(unreachable,.95)) if unreachable else None)

    # q and mesh FK can agree while the named landmarks differ. Compare the
    # original solver's proximal origins against CART's abduction origins.
    from retarget_reference.kinematics import HandKinematics
    from retarget.baseline.geort.adapter import anatomical_basis
    upstream=HandKinematics('WujiHand2','right')
    zero=upstream.forward(np.zeros(20))[0]
    basis=anatomical_basis(zero)
    upstream_points=np.asarray([upstream.forward(v)[0] for v in ik.detach().cpu().numpy()])
    upstream_points=(upstream_points-upstream_points[:,:1])@basis
    landmark_error=np.linalg.norm(upstream_points-robot.detach().cpu().numpy(),axis=-1)*1000
    report['ik_vs_cart_landmark_definition'] = dict(mean_error_by_landmark_mm=landmark_error.mean(0).tolist(),
        max_mm=float(landmark_error.max()), note='Each model uses its own neutral canonical basis; q order is identical')

    teacher=ContactTeacher(hand)
    @torch.no_grad()
    def contact_consistency(q):
        p,r,*_=hand.frames(q);xs=[];ds=[];ls=[]
        for i,row in enumerate(rows):
            for c in row['contacts']['object_contacts']:
                owner=link_owner(c['region']);inverse=r[i,owner].T
                xs.append(inverse@(hand.tensor(c['position_m'])-p[i,owner]))
                ds.append(inverse@hand.tensor(c['normal']));ls.append(owner)
        if not xs:return {}
        x=torch.stack(xs);d=torch.stack(ds);links=torch.tensor(ls,device=q.device)
        th,tn,tv=teacher.label(x,d,links,torch.ones_like(links,dtype=torch.bool))
        ah=torch.empty_like(th);av=torch.zeros_like(tv)
        for owner in set(ls):
            ids=links==owner
            hit,_,valid=hand.nearest(x[ids],owner,palmar=True,escape=d[ids])
            ah[ids]=hit;av[ids]=valid
        predicted,_=collision.model.contact(x,d,links,torch.ones_like(links,dtype=torch.bool))
        both=tv&av
        def quantile(error):return float(torch.quantile(error,.95)*1000) if error.numel() else None
        return dict(queries=len(x),teacher_valid_fraction=float(tv.float().mean()),audit_valid_fraction=float(av.float().mean()),
            teacher_audit_hit_difference_p95_mm=quantile((th[both]-ah[both]).norm(dim=-1)),
            field_vs_teacher_hit_p95_mm=quantile((predicted[tv]-th[tv]).norm(dim=-1)),
            field_vs_audit_hit_p95_mm=quantile((predicted[av]-ah[av]).norm(dim=-1)))
    report['contact_consistency_at_ik']=contact_consistency(ik)

    if args.optimize_steps:
        q=torch.nn.Parameter(ik.detach().clone());optimizer=torch.optim.Adam([q],lr=.005)
        progress=[]
        with frozen(collision.model):
            for step in range(args.optimize_steps):
                optimizer.zero_grad()
                loss,parts,_=objective.batch(q, rows, prior_ik, previous_rows=previous,
                    normalizers=weights,collect_metrics=False)
                if args.optimization_geometry == 'native-contact':
                    # The detached native witnesses are recomputed at every
                    # step. This is a diagnostic reference optimizer, never a
                    # policy training path or an offline trajectory asset.
                    collision.auditing=True
                    _,native_parts,_=objective.batch(q, rows, prior_ik, previous_rows=previous,
                        normalizers=weights,audit=True,collect_metrics=False)
                    collision.auditing=False
                    loss=loss+objective.settings.contact*(native_parts['contact']-parts['contact'])+objective.settings.normal*(native_parts['normal']-parts['normal'])
                    parts=dict(parts,contact=native_parts['contact'],normal=native_parts['normal'])
                loss.backward();optimizer.step()
                with torch.no_grad():q.clamp_(hand.low,hand.high)
                if step%25==0 or step==args.optimize_steps-1:
                    progress.append(dict(step=step+1,total=float(loss.detach()),parts=scalar_parts(parts)))
        report['direct_q_optimization'] = dict(steps=args.optimize_steps, geometry=args.optimization_geometry, progress=progress,
                        native=evaluate(q.detach(),prior_ik,audit=True), learned=evaluate(q.detach(),prior_ik,audit=False))
        report['direct_q_optimization']['contact_consistency']=contact_consistency(q.detach())

    report['elapsed_s']=time.monotonic()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(args.output),elapsed_s=report['elapsed_s'])),flush=True)


if __name__ == '__main__':
    main()
