"""Resume geometry to its full budget, then train policy only after validation.

Run from the same source checkout and Python environment as the existing job.
The caller must stop the old controller first; the normal training lock prevents
concurrent jobs. The initial migration permits only a reviewed training-driver
change, verified against an archived copy, and the new schedule configuration.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_config(config, current_geometry, schedule_sha256, previous_training_source=None):
    """Strictly allow the driver/schedule migration, preserving all model code."""
    from moxian_web.cart_geometry_schedule import COMPLETION, SELECTION
    updated = copy.deepcopy(config)
    recorded = copy.deepcopy(config['geometry'])
    current_driver = current_geometry['files']['cart_training.py']
    old_driver = recorded['files']['cart_training.py']
    if old_driver != current_driver:
        if previous_training_source is None or sha256(previous_training_source) != old_driver:
            raise ValueError('Archived training source does not match the run fingerprint')
        recorded['files']['cart_training.py'] = current_driver
    if recorded != current_geometry:
        raise ValueError('Geometry/model/data code changed beyond the training driver; refusing migration')
    gate = config['geometry_gate']
    if not gate['required'] or gate['maximum_updates'] != 30000:
        raise ValueError('This controller requires the production 30000-update geometry gate')
    expected = dict(completion=COMPLETION, checkpoint_selection=SELECTION, schedule_sha256=schedule_sha256)
    if any(key in gate and gate[key] != value for key, value in expected.items()):
        raise ValueError('An existing geometry schedule differs from this implementation')
    updated['geometry'] = current_geometry
    updated['geometry_gate'].update(expected)
    return updated


def resume_arguments(config):
    tracking = config['tracking']
    return dict(epochs=config['epochs'], seed=config['seed'], batch_size=config['batch_size'],
        length=config['sequence_length'], architecture=config['architecture_name'], resume=True,
        warmup_steps=config['sdf_warmup_steps'], micro_batch_size=config['micro_batch_size'],
        initial_noise_rad=config['initial_noise_rad'], attention_precision=config['attention_precision_request'],
        validation_interval=config['validation_interval'], validation_window_budget=config['validation_window_budget'],
        window_stride=config['window_stride'], policy_objective=config['objective_version'],
        policy_loss_scale=config['policy_loss_scaling']['scale'], optimizer_kind=config['optimizer_kind'],
        wandb_project=tracking['project'], wandb_entity=tracking['entity'], wandb_base_url=tracking['base_url'],
        wandb_mode=tracking['mode'], geometry_completion=config.get('geometry_gate',{}).get('completion','full_budget_best'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing cart data root')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--previous-training-source', type=Path,
                        help='Archived pre-migration cart_training.py; required for the first migration')
    parser.add_argument('--check-only', action='store_true', help='Verify configuration without changing files or starting training')
    args = parser.parse_args()
    from moxian_web.cart_training import train, implementation_identity
    from moxian_web.cart_geometry_schedule import COMPLETION
    from moxian_web.geort_service import safe_id
    from retarget.baseline.geort.training import atomic_json
    run_id = safe_id(args.run_id)
    folder = args.root / 'models' / run_id
    config_path = folder / 'config.json'
    config = json.loads(config_path.read_text())
    if config['run_id'] != run_id:
        raise ValueError('Run ID mismatch')
    schedule_path = Path(__file__).resolve().parents[1] / 'moxian_web/cart_geometry_schedule.py'
    updated = prepare_config(config, implementation_identity(), sha256(schedule_path), args.previous_training_source)
    kwargs = resume_arguments(updated)
    if args.check_only:
        print(json.dumps(dict(check='passed', run_id=run_id, geometry_updates=30000,
            selection=updated['geometry_gate']['checkpoint_selection'], policy_epochs=config['epochs'],
            source_migration_required=updated != config, resume_arguments=kwargs)), flush=True)
        return
    with (folder / 'geometry_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Hold the ordinary training lock during configuration migration too.
        with (args.root.parent / 'geort/training.lock').open('a') as training_lock:
            fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if updated != config:
                # A policy checkpoint must never silently receive a new training implementation.
                if (folder / 'resume.pth').exists():
                    raise ValueError('Policy has already started; refusing geometry-only schedule migration')
                backup = folder / 'config.before_geometry_schedule.json'
                if backup.exists():
                    raise ValueError('Migration backup already exists; inspect it before retrying')
                atomic_json(backup, config)
                atomic_json(folder / 'geometry_schedule_migration.json', dict(time=time.time(),
                    previous_driver_sha256=config['geometry']['files']['cart_training.py'],
                    driver_sha256=updated['geometry']['files']['cart_training.py'],
                    schedule_sha256=sha256(schedule_path), optimizer_and_weights='resume_unchanged'))
                atomic_json(config_path, updated)
        status_path = folder / 'geometry_controller_status.json'
        atomic_json(status_path, dict(stage='running', pid=os.getpid(), started_at=time.time(),
            completion=COMPLETION, maximum_updates=30000, policy_epochs=config['epochs']))
        try:
            train(args.root, config['train_ids'], config['validation_ids'], run_id, device=args.device, **kwargs)
        except BaseException as exc:
            decision_path = folder / 'geometry_schedule_status.json'
            decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
            atomic_json(status_path, dict(stage='rejected' if decision.get('stage') == 'rejected' else 'failed',
                error=str(exc), ended_at=time.time(), pid=os.getpid()))
            traceback.print_exc()
            raise
        atomic_json(status_path, dict(stage='finished', ended_at=time.time(), pid=os.getpid()))


if __name__ == '__main__':
    main()
