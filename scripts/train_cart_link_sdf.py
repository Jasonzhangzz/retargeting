"""Train an independent link-SDF contact policy alongside existing jobs.

Fresh SDF/policy, Muon+SGD, up to 30000 geometry updates then gated policy
epochs inherited from reference (currently 1000). The root must be isolated:
recordings may be shared read-only, training lock/model/source must not be.
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
    args=parser.parse_args();base=args.root.parent;base.mkdir(parents=True,exist_ok=True)
    def status(stage,**values):
        data=dict(stage=stage,pid=os.getpid(),time=time.time(),run_id=args.run_id,**values)
        p=base/'controller_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(p)
        print(json.dumps(data),flush=True)
    if (base/'geort').is_symlink():raise ValueError('Concurrent run requires its own training lock, not a shared geort directory')
    with (base/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('loading_runtime')
            from moxian_web.cart_training import train
            from moxian_web.cart_link_sdf import OBJECTIVE
            from scripts.continue_cart_after_geometry import resume_arguments
            config=json.loads((args.root/'models'/args.run_id/'config.json' if args.resume else args.reference_config).read_text())
            kwargs=resume_arguments(config)
            kwargs.update(resume=args.resume,device=args.device,policy_objective=OBJECTIVE,
                geometry_completion=config.get('geometry_gate',{}).get('completion','converged_best') if args.resume else 'converged_best')
            manifest=dict(run_id=args.run_id,authorization='user requests a second FK/link-SDF training; keep current job running',
                schema=1,source=str(Path(__file__).resolve().parents[1]),root=str(args.root.resolve()),
                device=args.device,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                geometry_updates=30000,policy_epochs=config['epochs'],optimizer=kwargs['optimizer_kind'],
                seed=kwargs['seed'],batch_size=kwargs['batch_size'],sequence_length=kwargs['length'],
                train_ids=config['train_ids'],validation_ids=config['validation_ids'],
                initialization='resume_same_run' if args.resume else 'fresh_policy_and_21_independent_SDFs_no_checkpoint',
                contact='fixed_observed_point_link_and_link_pair_labels_distance_zero',
                collision='all_allowed_link_pairs_and_active_taxels_signed_penetration',
                limit_m=.005,geometry_acceptance='distance_sign_first_derivatives_FK_and_dense_native_escape',
                schedule='converged_best; minimum5000; 6 validations without >1pct improvement; all gates pass; maximum30000',
                baseline_run=config['run_id'],baseline_mutation=False)
            if not args.resume:(base/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
            status('running',**{k:v for k,v in manifest.items() if k!='run_id'})
            train(args.root,config['train_ids'],config['validation_ids'],args.run_id,**kwargs)
            status('finished')
        except BaseException as exc:
            decision=args.root/'models'/args.run_id/'geometry_schedule_status.json'
            rejected=decision.exists() and json.loads(decision.read_text()).get('stage')=='rejected'
            status('rejected' if rejected else 'failed',error=str(exc));raise


if __name__=='__main__':main()
