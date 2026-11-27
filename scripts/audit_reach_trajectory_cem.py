"""CARD CEM over 16 action steps, with the current production temporal objective.

This is a finite-budget feasibility experiment, not a global convergence proof.
It uses the production CARD CEM sampler and production policy transition, but
optimizes each recorded window independently instead of fitting network weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import warp as wp
from dataclasses import replace
from moxian_web.cart_data import load_sessions
from moxian_web.cart_initialization import attach_ik
from moxian_web.cart_card_cost import attach_card_targets
from moxian_web.cart_fast import PreparedBatch, prepare_windows
from moxian_web.cart_training import windows, predict_trajectory
from moxian_web.cart_wrist import FloatingCartHand, state_velocity, relative_rotation
from moxian_web.cart_palm_alignment import PalmAlignedNetwork, unbounded_wrist_alignment
from moxian_web.cart_reach_loss import ReachObjective, SPEC, command_motion_jitter
from retarget.baseline.card.math import CardSettings, cem


class WindowCost:
    def __init__(self, model, objective, window):
        self.model, self.objective = model, objective
        self.device = objective.hand.device
        offset = len(window['rows']) - window['length'] - 10
        self.rows = window['rows'][offset + 1:offset + 17]
        self.initial = model.initial_state(window['initial_q'][None])
        stamps = torch.tensor([r['timestamp_ms'] for r in window['rows'][offset:offset + 17]],
                              device=self.device, dtype=torch.float64)
        self.dt = (stamps.diff() / 1000).float()
        self.mask = torch.stack([r['_card_targets']['mask'] for r in self.rows]).bool()
        force = torch.stack([r['_card_targets']['force'] for r in self.rows]).clamp_min(0)
        self.rho = (1 + .5 * force / force.masked_fill(~self.mask, 0).amax(-1, keepdim=True).clamp_min(1e-12)).clamp(1, 1.5)

    def rollout(self, actions):
        q = self.initial.expand(len(actions), -1)
        states = []
        for delta in actions.transpose(0, 1):
            q = self.model.apply_delta(q, delta)
            states.append(q)
        return torch.stack(states)

    def convert_wrist_actions(self, actions, states):
        actions=actions.clone()
        previous=torch.cat((self.initial.expand(1,len(actions),-1),states[:-1]))
        actions[...,20:23]=((states[...,20:23]-previous[...,20:23])/self.model.wrist_step[:3]).transpose(0,1)
        actions[...,23:26]=(relative_rotation(states[...,23:26],previous[...,23:26])/self.model.wrist_step[3:]).transpose(0,1)
        return actions

    def evaluate(self, actions):
        states = self.rollout(actions)
        return self.evaluate_states(states)

    def evaluate_states(self, states):
        """Score explicit trajectories too, for independent native CARD controls."""
        length, batch, _ = states.shape
        sq, mask, rho = self.objective.distances(states.flatten(0, 1),
                                                [r for r in self.rows for _ in range(batch)])
        sq = sq.reshape(length, batch, 16)
        contact = (1600 * self.rho[:, None].square() * sq * self.mask[:, None]).sum(-1).mean(0)
        stamps=torch.tensor([r['timestamp_ms'] for r in self.rows],device=states.device,dtype=torch.float64)
        motion=command_motion_jitter(states,stamps[:,None].expand(-1,batch))
        gap = (sq + 1e-12).sqrt()
        speed = (gap[1:] - gap[:-1]) / self.dt[1:, None, None].clamp_min(.001)
        second = (speed[1:] - speed[:-1]) / (.5 * (self.dt[2:] + self.dt[1:-1])).clamp_min(.001)[:, None, None] / 120**2
        retained = self.mask[2:] & self.mask[1:-1] & self.mask[:-2]
        jitter = SPEC['contact_jitter_weight'] * ((second / .001).square() * retained[:, None]).sum(-1).mean(0)
        parts = dict(contact=contact, motion_jitter=motion, contact_jitter=jitter)
        parts['total'] = sum(parts.values())
        return parts, states, gap

    def parity(self, actions):
        parts, states, _ = self.evaluate(actions)
        batch = len(actions)
        previous = torch.cat((self.initial.expand(1, batch, -1), states[:-1]))
        v = state_velocity(states, previous, self.dt[:, None])
        controls = []
        for t in range(16):
            a = torch.zeros_like(v[t]) if t == 0 else (v[t] - v[t-1]) / (.5 * (self.dt[t] + self.dt[t-1]))
            controls.append((v[t], a, torch.full((batch,), t > 0, device=self.device, dtype=torch.bool)))
        tr = SimpleNamespace(batch=[None]*batch, predictions=list(states), commands=list(states),
                             pending=[(states[t], [self.rows[t]]*batch) for t in range(16)], controls=controls)
        _, reference = self.objective.trajectory_loss(tr)
        for name in ['total', 'contact', 'motion_jitter', 'contact_jitter']:
            torch.testing.assert_close(parts[name].mean(), reference[name], rtol=2e-5, atol=2e-6)
        return {k: float(v.mean()) for k, v in parts.items()}

    def metrics(self, actions):
        parts, states, gap = self.evaluate(actions)
        active = gap[:, 0][self.mask] * 1000
        return {**{k: float(v[0]) for k, v in parts.items()}, 'mean_mm': float(active.mean()),
                'p95_mm': float(torch.quantile(active, .95)), 'max_mm': float(active.max()),
                'within_1mm': float((active < 1).float().mean()), 'anchors': active.numel(),
                'states': states[:, 0].cpu().tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('output/moxian/cart'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', type=int, default=4)
    p.add_argument('--population', type=int, default=1024)
    p.add_argument('--iterations', type=int, default=128)
    p.add_argument('--initializers', nargs='+', default=['policy', 'zero'])
    p.add_argument('--sigmas', nargs='+', type=float, default=[.15])
    p.add_argument('--warm-start-results', type=Path)
    p.add_argument('--unbounded-wrist', action='store_true')
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists(): raise ValueError('Use a new evidence directory')
    torch.set_num_threads(1); torch.manual_seed(714); wp.init()
    stream = torch.cuda.Stream(); torch.cuda.set_stream(stream); wp.set_stream(wp.stream_from_torch(stream))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False); config = saved['config']
    hand = FloatingCartHand('cuda:0'); objective = ReachObjective(hand)
    model = PalmAlignedNetwork(hand.low, hand.high, 'attention', alignment=config['palm_alignment'],**{k:v for k,v in config.get('policy_network',{}).items() if k!='parameters'}).to(hand.device)
    model.load_state_dict(saved['model']); model.eval(); model.set_attention_precision('bfloat16')
    transition=model if not args.unbounded_wrist else PalmAlignedNetwork(hand.low,hand.high,'attention',
        alignment=unbounded_wrist_alignment(config['palm_alignment'])).to(hand.device)
    sessions, _ = load_sessions(args.root, config['validation_ids'], 'validation', require_complete=True)
    attach_ik(args.root, sessions); attach_card_targets(args.root, sessions, hand.device)
    ws = windows(sessions, 16, 16); prepare_windows(sessions, ws, hand.device)
    chosen64 = np.linspace(0, len(ws)-1, min(64, len(ws))).astype(int)
    chosen = chosen64[np.linspace(0, len(chosen64)-1, args.cases).astype(int)]
    warm = {} if args.warm_start_results is None else {
        (r['window'], r['initialization']): r
        for r in map(json.loads, args.warm_start_results.read_text().splitlines())}
    settings = replace(CardSettings(), population=args.population, iterations=args.iterations,
                       translation_sigma_m=0., rotation_sigma_rad=0., joint_sigma_rad=.15, seed=714)
    manifest = dict(objective=SPEC, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    checkpoint_update=saved.get('selection', {}).get('update', saved.get('updates')), windows=chosen.tolist(),
                    settings=settings.payload(), sigma_stages=args.sigmas,
                    warm_start_sha256=None if args.warm_start_results is None else hashlib.sha256(args.warm_start_results.read_bytes()).hexdigest(),
                    parameterization='16x26 bounded actions; exact PalmAlignedNetwork.apply_delta',
                    horizon=16, disturbance=0., domain=('same initial IK and fingers; unbounded wrist pose and increments, same temporal costs' if args.unbounded_wrist else 'same initial IK, joint limits, MCP wrist residual and per-step action limits as validation'),
                    sampler='production retarget.baseline.card.math.cem; dummy base fixed; flattened action vector in joint slot',
                    scope='heldout finite-budget trajectory optimization; no learned weight updates, no global optimality guarantee',
                    gpu=torch.cuda.get_device_name(), source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__, 'moxian_web/cart_reach_loss.py', 'moxian_web/cart_palm_alignment.py', 'retarget/baseline/card/math.py']})
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)); records=[]
    with torch.no_grad():
        for wi in chosen:
            batch = PreparedBatch([ws[int(wi)]], hand.device); recorded=[]
            decode = model.decode_observation_context
            def capture(*a, **kw):
                delta=decode(*a, **kw);recorded.append(delta.clone());return delta
            model.decode_observation_context=capture
            tr=predict_trajectory(model, objective, batch, step_noise_rad=0.)
            model.decode_observation_context=decode
            policy_actions=torch.stack(recorded, 1)
            evaluator=WindowCost(transition, objective, ws[int(wi)])
            if args.unbounded_wrist:policy_actions=evaluator.convert_wrist_actions(policy_actions,torch.stack(tr.predictions))
            torch.testing.assert_close(evaluator.rollout(policy_actions), torch.stack(tr.predictions), rtol=2e-5, atol=2e-6)
            evaluator.parity(torch.cat((policy_actions, (policy_actions+.01*torch.randn_like(policy_actions)).clamp(-1,1))))
            for initialization in args.initializers:
                actions=policy_actions if initialization=='policy' else torch.zeros_like(policy_actions)
                if (int(wi), initialization) in warm:
                    record=warm[(int(wi), initialization)]
                    actions=hand.tensor(record['actions'])[None]
                    if args.unbounded_wrist:actions=evaluator.convert_wrist_actions(actions,hand.tensor(record['after']['states'])[:,None])
                before=evaluator.metrics(actions);tick=time.perf_counter()
                def score(t,r,j):return evaluator.evaluate(j.reshape(-1,16,26))[0]['total']
                initial=(hand.tensor([0.,0.,0.]),torch.eye(3,device=hand.device),actions.flatten())
                stages=[]
                lower=torch.full_like(actions,-1);upper=torch.ones_like(actions)
                if args.unbounded_wrist:lower[...,20:]=-torch.inf;upper[...,20:]=torch.inf
                for stage, sigma in enumerate(args.sigmas):
                    best,diag=cem(initial,None,lower.flatten(),upper.flatten(),score,
                                  replace(settings,joint_sigma_rad=sigma),
                                  torch.Generator(device=hand.device).manual_seed(714+int(wi)+stage*1000))
                    stages.append(dict(sigma=sigma,solver=diag));initial=best
                result=best[2].reshape(1,16,26);after=evaluator.metrics(result);evaluator.parity(result)
                record=dict(window=int(wi),initialization=initialization,before=before,after=after,
                            seconds=time.perf_counter()-tick,solver=diag,stages=stages,actions=result[0].cpu().tolist())
                assert after['total'] <= before['total']+1e-5
                records.append(record)
                with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps({k:v for k,v in record.items() if k not in ['actions','solver','before','after']}|dict(
                    before_total=before['total'],after_total=after['total'],before_mm=before['mean_mm'],after_mm=after['mean_mm'],max_mm=after['max_mm'])),flush=True)
    (args.out/'summary.json').write_text(json.dumps(dict(complete=True,results=[{k:v for k,v in r.items() if k!='actions'} for r in records]),indent=2))


if __name__=='__main__':main()
