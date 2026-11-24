"""Fresh wrist policy with SDF/FK collision and smooth geometric contact.

Reuse only the reference run's recording IDs and hyperparameters. New runs load
no weights or optimizer state. Complete 30000 geometry updates, select/revalidate
the best geometry, and only then start policy training. --resume preserves all
training state of this same run, with the ordinary strict configuration checks.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--reference-config',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    base=args.root.parent;base.mkdir(parents=True,exist_ok=True)
    def status(stage,**values):
        data=dict(stage=stage,pid=os.getpid(),time=time.time(),run_id=args.run_id,**values)
        path=base/'controller_status.json';temp=path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(data,allow_nan=False,indent=2));temp.replace(path)
        print(json.dumps(data),flush=True)
    with (base/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('loading_runtime')
            from moxian_web.cart_training import train
            from moxian_web.cart_wrist import VERSION
            from scripts.continue_cart_after_geometry import resume_arguments
            source=args.root/'models'/args.run_id/'config.json' if args.resume else args.reference_config
            config=json.loads(source.read_text())
            if config['objective_version']!=VERSION:raise ValueError('Expected a floating-wrist reference configuration')
            kwargs=resume_arguments(config);kwargs.update(resume=args.resume,device=args.device)
            status('running',geometry_updates=30000,policy_epochs=config['epochs'],
                initialization='resume_same_run' if args.resume else 'fresh_policy_and_neural_SDF_no_checkpoint',
                contact='fixed_geometric_features_with_smooth_softmin',self_collision='neural_SDF_and_exact_FK')
            train(args.root,config['train_ids'],config['validation_ids'],args.run_id,**kwargs)
            status('finished')
        except BaseException as exc:
            decision=args.root/'models'/args.run_id/'geometry_schedule_status.json'
            rejected=decision.exists() and json.loads(decision.read_text()).get('stage')=='rejected'
            status('rejected' if rejected else 'failed',error=str(exc))
            raise


if __name__=='__main__':main()
