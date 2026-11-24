"""Reviewed schedule-only migration of an existing link-SDF warmup.

Stop only the matching v10r5 controller at a saved checkpoint before applying.
The original source snapshot is retained. Network, loss, optimizer, replay,
recording split and random states are restored from the same warmup checkpoint.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.cart_geometry_schedule import COMPLETION,CONVERGED_COMPLETION,EARLY_STOP


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def prepare_config(config,old_driver,new_driver,old_schedule,new_schedule):
    if old_driver==new_driver or old_schedule==new_schedule:raise ValueError('Revised driver and schedule must both be installed')
    if config.get('objective_version')!='card_style_link_sdf_contacts_v6_wrist':raise ValueError('Only the new link-SDF experiment may be migrated')
    if config['geometry']['files']['cart_training.py']!=old_driver:raise ValueError('Archived driver does not match run')
    gate=config['geometry_gate']
    if gate.get('completion')!=COMPLETION or gate.get('maximum_updates')!=30000 or not gate.get('required'):
        raise ValueError('Expected the original full-budget geometry schedule')
    if gate.get('schedule_sha256')!=old_schedule:raise ValueError('Archived schedule does not match run')
    result=copy.deepcopy(config);result['geometry']['files']['cart_training.py']=new_driver
    result['geometry_gate'].update(completion=CONVERGED_COMPLETION,early_stop=dict(EARLY_STOP),schedule_sha256=new_schedule)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--previous-source',type=Path,required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();source=Path(__file__).resolve().parents[1]
    base=args.root.parent;folder=args.root/'models'/args.run_id;path=folder/'config.json'
    config=json.loads(path.read_text());previous=args.previous_source
    # Compare every Python source file; only reviewed orchestration may differ.
    allowed={'moxian_web/cart_training.py','moxian_web/cart_geometry_schedule.py',
        'scripts/train_cart_link_sdf.py','scripts/continue_cart_after_geometry.py',
        'scripts/enable_cart_sdf_early_stop.py'}
    changed=[]
    for p in previous.rglob('*.py'):
        rel=str(p.relative_to(previous))
        if '__pycache__' in p.parts:continue
        current=source/rel
        if not current.exists() or sha256(p)!=sha256(current):changed.append(rel)
    if set(changed)-allowed:raise ValueError('Non-schedule code changed: '+str(set(changed)-allowed))
    updated=prepare_config(config,sha256(previous/'moxian_web/cart_training.py'),sha256(source/'moxian_web/cart_training.py'),
        sha256(previous/'moxian_web/cart_geometry_schedule.py'),sha256(source/'moxian_web/cart_geometry_schedule.py'))
    if (folder/'resume.pth').exists():raise ValueError('Policy has already started; refusing geometry-only migration')
    receipt=dict(time=time.time(),run_id=args.run_id,authorization='user permits early termination after neural SDF convergence',
        previous_source=str(previous),source=str(source),changes=changed,early_stop=dict(EARLY_STOP),
        maximum_updates=30000,network_loss_optimizer_data_unchanged=True,checkpoint='warmup.pth')
    if args.check_only:print(json.dumps(dict(receipt,check='passed')),flush=True);return
    with (base/'controller.lock').open('a') as control, (base/'geort/training.lock').open('a') as training:
        fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(training,fcntl.LOCK_EX|fcntl.LOCK_NB)
        backup=folder/'config.before_early_stop.json'
        if backup.exists():raise ValueError('An early-stop migration backup already exists')
        if not (folder/'warmup.pth').exists():raise ValueError('A complete warmup checkpoint is required')
        receipt['checkpoint_sha256']=sha256(folder/'warmup.pth')
        def write(p,d):
            tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
        write(backup,config);write(path,updated);write(base/'early_stop_migration.json',receipt)
        manifest=base/'MANIFEST.json'
        if manifest.exists():
            old=json.loads(manifest.read_text());write(base/'MANIFEST.before_early_stop.json',old)
            write(manifest,dict(old,source=str(source),schedule=CONVERGED_COMPLETION,early_stop=dict(EARLY_STOP),
                authorization_revision=2,maximum_geometry_updates=30000,resume_from_saved_state=True))
    print(json.dumps(dict(receipt,migration='applied')),flush=True)


if __name__=='__main__':main()
