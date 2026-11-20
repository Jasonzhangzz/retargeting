"""Fit the q-only collision potential and its Jacobian before policy training."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from moxian_web.cart_objective import CartHand
from moxian_web.cart_collision import CollisionModel
from moxian_web.cart_q_collision import attach_field,field_spec,QCollisionLearner


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--root',required=True)
    parser.add_argument('--train-id',default='20261118T134537_42b050e5');parser.add_argument('--validation-id',default='20261118T135558_05a2628d')
    parser.add_argument('--samples',type=int,default=32768);parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--resume',type=Path);parser.add_argument('--lr',type=float,default=5.e-4)
    parser.add_argument('--warm-start-v1',type=Path)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);root=Path(args.root)
    torch.set_num_threads(1);torch.manual_seed(20261120);started=time.time()
    h=CartHand(args.device);c=CollisionModel(h);spec=field_spec(h);attach_field(c,spec)
    learner=QCollisionLearner(c,batch_size=args.batch_size)
    def ik(key):
        with np.load(root/'recordings'/key/'cart_ik.npz') as f:return h.tensor(f['q'])
    train=ik(args.train_id);validation=ik(args.validation_id)
    def draw(n,poses,uniform_fraction):
        q=h.low+torch.rand(n,20,device=h.device)*(h.high-h.low)
        k=int(n*(1-uniform_fraction));ids=torch.randint(len(poses),(k,),device=h.device)
        q[:k]=(poses[ids]+torch.randn(k,20,device=h.device)*.12).clamp(h.low,h.high)
        return q
    reports=[]
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if saved['spec']!=spec:raise ValueError('Pretraining geometry changed')
        learner.model.load_state_dict(saved['model']);learner.load_state_dict(saved['learner'])
        holdout=saved['holdout_q'].to(h.device);reports=saved['validation']
        if 'torch_rng' in saved:torch.set_rng_state(saved['torch_rng'])
    elif args.warm_start_v1:
        saved=torch.load(args.warm_start_v1,map_location='cpu',weights_only=True)
        if saved['spec']['kind']!='q_collision_sobolev_v1' or saved['spec']['geometry']!=spec['geometry'] or saved['spec']['allowed']!=spec['allowed']:
            raise ValueError('Unexpected warm-start geometry')
        state=learner.model.state_dict()
        for k,v in saved['model'].items():
            if k in ('network.6.weight','network.6.bias'):
                state[k][:2].copy_(v);state[k][2].copy_(v[1])
                if k.endswith('bias'):state[k][2]-=3
            else:state[k].copy_(v)
        learner.model.load_state_dict(state)
        poses=saved['learner']['replay']['q'][:saved['learner']['size']]
        for start in range(0,len(poses),16):
            learner.add(poses[start:start+16].to(h.device))
            if (start+16)%512==0:print(json.dumps(dict(phase='labels',completed=start+16,total=len(poses))),flush=True)
        holdout=saved['holdout_q'].to(h.device)
    else:
        for start in range(0,args.samples,16):
            learner.add(draw(min(16,args.samples-start),train,.65))
            if (start+16)%512==0:
                print(json.dumps(dict(phase='labels',completed=min(start+16,args.samples),total=args.samples,elapsed_s=time.time()-started)),flush=True)
        holdout=draw(256,validation,.5)
    for group in learner.optimizer.param_groups:group['lr']=args.lr
    for step in range(learner.steps,args.steps):
        losses=learner.update()
        if (step+1)%100==0:print(json.dumps(dict(phase='fit',step=step+1,elapsed_s=time.time()-started,**losses)),flush=True)
        if (step+1)%1000==0 or step+1==args.steps:
            report=learner.validate(holdout);reports.append(dict(step=step+1,**report))
            print(json.dumps(dict(phase='validation',step=step+1,**report)),flush=True)
            checkpoint=dict(spec=spec,model=c.model.q_collision.state_dict(),learner=learner.state_dict(),
                validation=reports,train_recording=args.train_id,validation_recording=args.validation_id,
                holdout_q=holdout.cpu(),seed=20261120,torch_rng=torch.get_rng_state())
            torch.save(checkpoint,out/'last.pth.tmp');(out/'last.pth.tmp').replace(out/'last.pth')
            (out/'validation.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(dict(phase='finished',elapsed_s=time.time()-started)),flush=True)


if __name__=='__main__':main()
