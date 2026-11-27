"""Four exact trajectory costs from a common independently solved CARD seed.

Direct physical poses, no policy delta parameterization, no learned weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from dataclasses import replace
import torch
import warp as wp
from audit_reach_metrics import (VARIANTS, selected_cost, metrics, WindowCost,
    FloatingCartHand, ReachObjective, PalmAlignedNetwork, unbounded_wrist_alignment,
    load_sessions, attach_ik, attach_card_targets, windows, prepare_windows,
    CardSettings, cem, so3_exp, rotation_vector, SPEC)


class PoseCost(WindowCost):
    def __init__(self, model, objective, window, seed):
        super().__init__(model,objective,window)
        self.seed=seed
        self.scale=torch.cat(((objective.hand.high-objective.hand.low)*.5,seed.new_tensor([.02]*3+[.2]*3)))
    def rollout(self, parameters):
        delta=parameters.transpose(0,1)*self.scale
        seed=self.seed[:,None]
        return torch.cat((seed[...,:23]+delta[...,:23],rotation_vector(so3_exp(delta[...,23:])@so3_exp(seed[...,23:]))),-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','checkpoint','out','seed-results']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument('--population',type=int,default=1024)
    p.add_argument('--iterations',type=int,default=128)
    p.add_argument('--sigmas',type=float,nargs='+',default=[.15,.03,.006])
    p.add_argument('--windows',type=int,nargs='+',default=[0,43,86,129])
    p.add_argument('--candidate-results',type=Path,nargs='*',default=[])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'manifest.json').exists():raise ValueError('Use new output directory')
    torch.set_num_threads(1);torch.manual_seed(714);wp.init()
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream);wp.set_stream(wp.stream_from_torch(stream))
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=saved['config']
    hand=FloatingCartHand('cuda:0');objective=ReachObjective(hand)
    model=PalmAlignedNetwork(hand.low,hand.high,'attention',alignment=unbounded_wrist_alignment(cfg['palm_alignment'])).to(hand.device)
    sessions,_=load_sessions(args.root,cfg['validation_ids'],'validation',require_complete=True)
    attach_ik(args.root,sessions);attach_card_targets(args.root,sessions,hand.device)
    ws=windows(sessions,16,16);prepare_windows(sessions,ws,hand.device)
    seeds={r['window']:r for r in map(json.loads,args.seed_results.read_text().splitlines()) if r['variant']=='native'}
    assert set(seeds)=={0,43,86,129}
    if not set(args.windows)<=set(seeds):raise ValueError('Unknown seed window')
    candidates=[(str(path),r) for path in args.candidate_results
                for r in map(json.loads,path.read_text().splitlines())]
    settings=replace(CardSettings(),population=args.population,iterations=args.iterations,translation_sigma_m=0.,rotation_sigma_rad=0.,joint_sigma_rad=.15,seed=714)
    manifest=dict(variants=args.variants,weights=VARIANTS,objective=SPEC,windows=sorted(args.windows),
        initialization=('lowest full-production-cost existing candidate, rescored with native geometry; refinement is additional budget, not the equal-budget ablation' if candidates else 'identical per-frame native CARD contact solutions for every variant; includes seed as incumbent'),
        candidate_files={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in args.candidate_results},
        parameterization='16 independent q26 physical poses; joint offsets normalized by half mechanical range, wrist translation .02m, SO3 left increments .2rad',
        domain='mechanical finger bounds only; wrist unrestricted; same physical domain as native CARD',
        settings=settings.payload(),sigmas=args.sigmas,optimizer_updates=0,training_started=False,
        seed_file_sha256=hashlib.sha256(args.seed_results.read_bytes()).hexdigest(),
        sources={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,'scripts/audit_reach_temporal_ablation.py','scripts/audit_reach_trajectory_cem.py','moxian_web/cart_reach_loss.py']})
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2));records=[]
    with torch.no_grad():
        for wi in sorted(args.windows):
            seed=hand.tensor(seeds[wi]['after']['states'])
            ev=PoseCost(model,objective,ws[wi],seed)
            seed_scores=[]
            if candidates:
                options=[('native_seed',seed)]+[(path+':'+r['variant'],hand.tensor(r['after']['states'])) for path,r in candidates if r['window']==wi]
                seed_scores=[dict(source=name,total=float(ev.evaluate_states(q[:,None])[0]['total'][0])) for name,q in options]
                chosen=min(range(len(options)),key=lambda k:seed_scores[k]['total'])
                seed=options[chosen][1];ev=PoseCost(model,objective,ws[wi],seed)
            zero=torch.zeros((1,16,26),device=hand.device)
            torch.testing.assert_close(ev.rollout(zero)[:,0],seed,rtol=2e-5,atol=2e-6)
            ev.parity(zero)
            lo=torch.full_like(zero,-torch.inf);hi=torch.full_like(zero,torch.inf)
            lo[...,:20]=(hand.low-seed[:,:20])/ev.scale[:20]
            hi[...,:20]=(hand.high-seed[:,:20])/ev.scale[:20]
            for variant in args.variants:
                before=metrics(ev,ev.rollout(zero),variant);tick=time.perf_counter();stages=[]
                def score(t,r,j):return selected_cost(ev.evaluate(j.reshape(-1,16,26))[0],variant)
                initial=(hand.tensor([0.,0.,0.]),torch.eye(3,device=hand.device),zero.flatten())
                for stage,sigma in enumerate(args.sigmas):
                    best,diag=cem(initial,None,lo.flatten(),hi.flatten(),score,replace(settings,joint_sigma_rad=sigma),
                        torch.Generator(device=hand.device).manual_seed(714+wi+stage*1000))
                    initial=best;stages.append(dict(sigma=sigma,solver=diag))
                params=best[2].reshape(1,16,26);after=metrics(ev,ev.rollout(params),variant);ev.parity(params)
                assert after['objective']<=before['objective']+1e-5
                record=dict(window=wi,variant=variant,before=before,after=after,seconds=time.perf_counter()-tick,stages=stages,seed_candidates=seed_scores)
                records.append(record)
                with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(dict(window=wi,variant=variant,seconds=record['seconds'],before_objective=before['objective'],after_objective=after['objective'],mean_mm=after['mean_mm'],max_mm=after['max_mm'])),flush=True)
    (args.out/'summary.json').write_text(json.dumps(dict(complete=True,training_started=False,results=records),indent=2))

if __name__=='__main__':main()
