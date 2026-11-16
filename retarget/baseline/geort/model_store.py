"""Publish one GeoRT model after successful training; keep the old one on failure."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import shutil


@contextmanager
def model_lock(root, exclusive=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'model-publish.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def prune_checkpoints(root):
    """Keep the published IK/FK and its one reusable neural-FK cache only.

    Caller holds the model publication lock and the global training lock.
    Recordings, FK labels, configuration and loss traces are untouched.
    """
    from retarget.baseline.geort.adapter import fk_cache_key
    root=Path(root);current=root/'models/current';removed=[]
    config=json.loads((current/'config.json').read_text())
    keep={current/'last.pth',current/'fk.pth'}
    if 'robot' in config:
        keep.add(root/'fk'/fk_cache_key(config)/f'fk_{config["robot_samples"]}_{config["fk_epochs"]}.pth')
    for area in ('models','training','fk'):
        for path in (root/area).rglob('*'):
            if path.suffix in ('.pth','.pt') and path not in keep and path.is_file() and not path.is_symlink():
                path.unlink();removed.append(str(path.relative_to(root)))
    return removed


def prune_pose_caches(root):
    """Keep pose caches referenced by the published model/current reference.

    Caller holds model-publish and training locks; source recordings and source
    geometry caches are untouched. Only hash-named cache directories qualify.
    """
    from .adapter import fk_cache_key,digest
    root=Path(root);config=json.loads((root/'models/current/config.json').read_text())
    if 'robot' not in config:return []
    keep={'fk':fk_cache_key(config),
          'robot_contact_targets':config.get('robot_contact_targets',{}).get('cache_id')}
    current_spec=None
    if config.get('source'):
        from .workspace import training_reference_spec,REVISION
        try:
            current_spec=training_reference_spec(root)
            keep['workspace']=digest([config['source'],REVISION,current_spec])
        except FileNotFoundError:pass
    removed=[]
    for area,key in keep.items():
        parent=root/area
        if parent.is_symlink() or not parent.is_dir():continue
        for folder in parent.iterdir():
            if (folder.name==key or not folder.is_dir() or folder.is_symlink()
                    or len(folder.name)!=64 or any(c not in '0123456789abcdef' for c in folder.name)):
                continue
            if area=='workspace':
                # The live calibration may differ from the training source.
                # Keep any reference using the current labels and policy, and
                # leave an incomplete writer alone until a later cleanup.
                try:
                    if json.loads((folder/'reference_spec.json').read_text())==current_spec:continue
                except (OSError,ValueError):continue
            shutil.rmtree(folder);removed.append(str(folder.relative_to(root)))
    return removed


def publish(root, staged, run_id):
    from retarget.baseline.geort.training import atomic_json
    root=Path(root);staged=Path(staged)
    status=json.loads((staged/'status.json').read_text())
    if status.get('status')!='completed' or not (staged/'last.pth').is_file():
        raise ValueError('只有训练成功的完整 checkpoint 才能替换当前模型')
    models=root/'models';models.mkdir(parents=True,exist_ok=True)
    current=models/'current';backup=root/'training'/'.previous-current'
    with model_lock(root,exclusive=True):
        if current.is_symlink() or models.is_symlink():raise ValueError('模型目录不能是符号链接')
        if backup.exists():shutil.rmtree(backup)
        moved=False
        try:
            if current.exists():os.replace(current,backup);moved=True
            os.replace(staged,current)
            atomic_json(root/'active.json',dict(model='current',run_id=run_id))
        except BaseException:
            if current.exists() and not staged.exists():os.replace(current,staged)
            if moved:os.replace(backup,current)
            raise
        # Model tensors already loaded by a viewer remain valid in memory.
        # The web service notices run_id and reloads the new checkpoint.
        cleanup=[]
        for folder in models.iterdir():
            if folder.name!='current' and folder.is_dir() and not folder.is_symlink():
                try:shutil.rmtree(folder)
                except OSError as exc:cleanup.append(str(exc))
        if backup.exists():
            try:shutil.rmtree(backup)
            except OSError as exc:cleanup.append(str(exc))
        try:
            prune_checkpoints(root)
            removed=prune_pose_caches(root)
            if removed:atomic_json(root/'pose-cache-cleanup.json',dict(run_id=run_id,removed=removed))
        except OSError as exc:cleanup.append(str(exc))
        if cleanup:atomic_json(root/'model-cleanup.json',dict(errors=cleanup))
    return current
