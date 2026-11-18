"""CLI for an independent augmentation campaign; never changes GeoRT models."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
from .templates import SOURCE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    default=SOURCE.parent/'wuji2_primitive_aug_v1'
    for name in ('generate','resume','audit','report','calibrate'):
        p=sub.add_parser(name)
        p.add_argument('--output',type=Path,default=default)
        p.add_argument('--device',default='cuda:0')
        if name in ('generate','calibrate'):
            p.add_argument('--source',type=Path,default=SOURCE)
            p.add_argument('--per-type',type=int,default=50000)
            p.add_argument('--points',type=int,default=1024)
            p.add_argument('--seed',type=int,default=20261117)
            p.add_argument('--batch',type=int,default=64)
            p.add_argument('--iterations',type=int,default=60)
            p.add_argument('--patch-samples',type=int,default=128)
            p.add_argument('--baseline',type=int,default=200)
            p.add_argument('--bootstrap-from',type=Path)
        if name in ('generate','resume'):
            p.add_argument('--max-jobs',type=int,help='Pause cleanly after this many proposal batches')
        if name=='audit':p.add_argument('--repair',action='store_true',help='Remove failed rows atomically; subsequent resume refills quotas')
        p.add_argument('--min-free-gib',type=float,default=5)
    args=parser.parse_args()
    # Restrict CPU BLAS threads before importing the numerical/GPU modules.
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    import torch,warp as wp
    torch.set_num_threads(1);wp.init()
    from .campaign import Campaign
    from .storage import write_json
    if args.command in ('resume','audit','report'):
        c=json.loads((args.output/'config.json').read_text())
        campaign=Campaign(args.output,source=Path(c['source_root']),per_type=c['per_type'],points=c['points'],seed=c['seed'],device=args.device,
            batch=c['batch'],iterations=c['iterations'],patch_samples=c['identity']['patch_witnesses'],
            baseline=c['baseline_samples_per_type'],min_free_gib=args.min_free_gib)
    else:
        campaign=Campaign(args.output,source=args.source,per_type=args.per_type,points=args.points,seed=args.seed,
            device=args.device,batch=args.batch,iterations=args.iterations,patch_samples=args.patch_samples,
            baseline=args.baseline,min_free_gib=args.min_free_gib)
        if args.bootstrap_from:
            import shutil
            other=json.loads((args.bootstrap_from/'config.json').read_text())
            for field in ('identity','iterations','code_sha256','seed','baseline_samples_per_type'):
                if other[field]!=campaign.config[field]:raise ValueError(f'Incompatible bootstrap {field}')
            for folder in ('templates','calibration','baseline'):
                if (args.bootstrap_from/folder).exists():
                    shutil.copytree(args.bootstrap_from/folder,args.output/folder,dirs_exist_ok=True)
    def stop(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        if args.command in ('generate','resume'):campaign.generate(args.max_jobs)
        elif args.command=='calibrate':campaign.calibrate()
        elif args.command=='audit':print(json.dumps(campaign.audit(args.repair),ensure_ascii=False))
        else:print(json.dumps(campaign.report(),ensure_ascii=False))
    except KeyboardInterrupt:
        campaign.checkpoint('paused_signal')
        print('Safely paused; use resume with the same output',flush=True)
        return 130
    except Exception as exc:
        if not campaign.state['phase'].startswith(('paused','incomplete')):campaign.checkpoint('failed')
        write_json(args.output/'last_error.json',dict(error=str(exc),type=type(exc).__name__))
        raise
    finally:
        campaign.lock.close()


if __name__=='__main__':sys.exit(main())
