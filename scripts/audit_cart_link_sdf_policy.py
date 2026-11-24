"""Audit policy geometry gradients on real observations without optimizer steps.

Checks the deployed distance graph, not agreement with native-mesh derivatives
or geometry acceptance. Native labels/projections are forbidden after setup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(checkpoint, recording, device):
    import numpy as np
    import torch
    from moxian_web.cart_collision import CollisionModel, frozen
    from moxian_web.cart_fast import PreparedBatch, prepare_windows
    from moxian_web.cart_link_sdf import LinkSDFGeometry, geometry_spec
    from moxian_web.cart_link_sdf_objective import LinkSDFObjective
    from moxian_web.cart_loss_scaling import normalized_policy_cost
    from moxian_web.cart_training import predict_trajectory, windows
    from moxian_web.cart_wrist import FloatingCartHand, CartWristNetwork
    from moxian_web.wuji_surface import COMPLETE_PAD_POLICY

    torch.set_num_threads(1)
    torch.manual_seed(714)
    hand = FloatingCartHand(device, contact_surface_policy=COMPLETE_PAD_POLICY)
    geometry = LinkSDFGeometry(hand, geometry_spec(hand))
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    geometry.load_state_dict(saved['collision'], strict=True)
    collision = CollisionModel(hand, geometry)
    objective = LinkSDFObjective(hand, collision)
    rows = [json.loads(line) for line in (recording / 'cart_targets.jsonl').open()]
    caches = list(recording.glob('card_cost_targets_*.npz'))
    if len(caches) != 1:
        raise ValueError('Provide a recording with one verified CARD target cache')
    with np.load(caches[0], allow_pickle=False) as cache:
        for i, row in enumerate(rows):
            row['_card_targets'] = {
                k: torch.tensor(cache[k][i], device=device)
                for k in ('tips', 'normals', 'directions', 'anchor_position',
                          'anchor_normal', 'force', 'u', 'mask')}
    with np.load(recording / 'cart_ik.npz', allow_pickle=False) as ik:
        for i, row in enumerate(rows):
            row['ik_q_rad'] = ik['q'][i].tolist()
            row['ik_valid'] = bool(ik['valid'][i])

    # Pick actual adjacent observations exercising both anchor and pair losses.
    choices = [i for i in range(1, len(rows))
               if 0 < rows[i]['timestamp_ms'] - rows[i-1]['timestamp_ms'] <= 250
               and rows[i]['stream_epoch'] == rows[i-1]['stream_epoch']
               and rows[i]['contacts']['self_pairs']]
    index = max(choices, key=lambda i: int((rows[i]['_card_targets']['mask'] &
                    rows[i-1]['_card_targets']['mask']).sum()))
    current, previous_rows = [rows[index]], [rows[index-1]]
    span = hand.high - hand.low
    candidates = hand.low + (.1 + .8 * torch.rand(8, 20, device=device)) * span
    with frozen(geometry), torch.no_grad():
        terms = geometry.self_terms(candidates)
        chosen = candidates[terms[:, 1].argmax()]
    wrist = chosen.new_tensor([.001, -.002, .003, .02, -.03, .01])
    # An external collision may legitimately be inactive on the original IK
    # pose. Move the robot wrist (never the observed points) to put one actual
    # active taxel inside a finger link. Native labels are allowed only here,
    # to construct an explicit penetrating diagnostic fixture.
    with torch.no_grad():
        owner = torch.arange(1, 21, device=device).repeat_interleave(256)
        lower, upper = collision.oracle.lower[owner], collision.oracle.upper[owner]
        probes = lower + torch.rand(len(owner), 3, device=device)*(upper-lower)
        native, _ = collision.oracle.label(probes, owner)
        predicted = geometry(probes, owner)
        inside = torch.minimum(-native, -predicted)
        selected = inside.argmax()
        assert inside[selected] > .005, 'No >5mm interior fixture found'
        probe, owner = probes[selected], owner[selected]
        p, r = hand.frames(torch.cat((chosen, wrist))[None])[:2]
        world = p[0, owner] + r[0, owner] @ probe
        active, _ = collision.oracle.active_clouds((p, r), current)
        point = active[0][(active[0]-world).norm(dim=-1).argmin()]
        wrist[:3] += point-world
    state = torch.cat((chosen, wrist))[None].requires_grad_()
    perturbation = .003 * torch.randn_like(state)
    perturbation[:, 20:23] = .0001
    reference = (state.detach() + perturbation).requires_grad_()
    direction = torch.randn_like(state)
    direction[:, 20:23] *= .01
    direction /= direction.norm()
    old_direction = torch.randn_like(state)
    old_direction[:, 20:23] *= .01
    old_direction /= old_direction.norm()

    forbidden_calls = []
    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        raise AssertionError('Native nearest query/projection in policy graph')
    collision.oracle.label = forbidden
    hand.nearest = forbidden
    hand.penetration = forbidden
    geometry.contact = forbidden
    names = ('contact_distance', 'contact_keep', 'contact_self_distance',
             'collision_self', 'collision_object', 'penetration_cap')
    report = dict(optimizer_updates=0, geometry_optimizer_updates=0,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  geometry_update=saved['update'], geometry_accepted=saved['accepted'],
                  observation_index=index, device=str(device),
                  penetrating_fixture=dict(target_link=int(owner),
                      native_depth_mm=float(-native[selected]*1000),
                      predicted_depth_mm=float(-predicted[selected]*1000),
                      robot_wrist_pose=wrist.tolist()),
                  self_queries_per_pose=len(geometry.q_collision.cap_sources),
                  active_taxels=sum(bool(a and b) for a, b in zip(
                      current[0]['tactile_state'], current[0]['tactile_valid'])),
                  components={})

    def evaluate(q, old):
        return objective.batch(q, current, old, previous_rows=previous_rows,
                               collect_metrics=False)[1]

    with frozen(geometry):
        parts = evaluate(state, reference)
        epsilon = 1.e-3
        with torch.no_grad():
            plus = evaluate(state.detach()+epsilon*direction,
                            reference.detach()+epsilon*old_direction)
            minus = evaluate(state.detach()-epsilon*direction,
                             reference.detach()-epsilon*old_direction)
        for name in names:
            value = parts[name]
            grad, old_grad = torch.autograd.grad(value, (state, reference),
                                                 retain_graph=True, allow_unused=True)
            assert grad is not None and torch.isfinite(grad).all(), name
            assert grad[:, :20].norm() > 0, name
            if name not in ('collision_self',):
                assert grad[:, 20:].norm() > 0, name
            if name == 'collision_self':
                assert torch.count_nonzero(grad[:, 20:]) == 0
            if name == 'contact_keep':
                assert old_grad is not None and torch.isfinite(old_grad).all()
                assert old_grad[:, :20].norm() > 0 and old_grad[:, 20:].norm() > 0
            ad = (grad*direction).sum()
            if old_grad is not None:
                ad = ad+(old_grad*old_direction).sum()
            fd = (plus[name]-minus[name])/(2*epsilon)
            relative = (ad-fd).abs()/torch.maximum(ad.abs(), fd.abs()).clamp_min(1.e-3)
            assert relative < .03, (name, float(ad), float(fd), float(relative))
            report['components'][name] = dict(
                value=float(value.detach()), finger_gradient_norm=float(grad[:, :20].norm()),
                wrist_gradient_norm=float(grad[:, 20:].norm()),
                previous_state_gradient_norm=float(old_grad.norm()) if old_grad is not None else 0.,
                directional_ad=float(ad), directional_fd=float(fd), relative_error=float(relative))

        # Real EMF input, 16-step unroll, geometry-only losses into policy heads.
        sessions = [(json.loads((recording/'manifest.json').read_text()), rows,
                     json.loads((recording/'cart_preprocess.json').read_text()))]
        available = windows(sessions, 16, 16)
        scored = sorted(available, key=lambda w: sum(len(r['contacts']['self_pairs'])
                        for r in w['rows']), reverse=True)[:2]
        prepare_windows(sessions, scored, device)
        model = CartWristNetwork(hand.low, hand.high).to(device)
        trajectory = predict_trajectory(model, objective, PreparedBatch(scored, device))
        _, terms, _, _ = trajectory.loss(objective, collect_metrics=False)
        loss = normalized_policy_cost(sum(terms[n] for n in names), 1.e8)
        # Only the last frame's contact/collision cost must still reach frame 1.
        q, row, old, acceleration, prior_rows = trajectory.pending[-1]
        _, final_parts, _ = objective.batch(q, row, old, acceleration=acceleration,
                                           previous_rows=prior_rows, collect_metrics=False)
        early = torch.autograd.grad(sum(final_parts[n] for n in names),
                                    trajectory.predictions[0], retain_graph=True)[0]
        assert torch.isfinite(early).all() and early.norm() > 0
        loss.backward()
        heads = [decoder[-2] for decoder in model.decoders] + [model.wrist_decoder[-2]]
        norms = [float(head.weight.grad.norm()) for head in heads]
        assert all(np.isfinite(v) and v > 0 for v in norms), norms
        assert all(p.grad is None for p in geometry.parameters())
        report['rollout'] = dict(length=16, parallel_windows=len(scored),
            loss=float(loss.detach()), finger_and_wrist_head_gradient_norms=norms,
            final_geometry_loss_to_first_state_gradient_norm=float(early.norm()),
            geometry_weights_frozen=True)
    report['native_policy_queries'] = len(forbidden_calls)
    report['passed'] = True
    report['scope'] = 'Autograd/FD and BPTT connectivity; does not establish mesh prediction accuracy'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    start = time.perf_counter()
    report = audit(args.checkpoint, args.recording, args.device)
    report['wall_s'] = time.perf_counter()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
