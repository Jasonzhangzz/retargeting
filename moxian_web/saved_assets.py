"""Delete explicitly selected saved assets, excluding live jobs and symlinks."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import shutil


@contextmanager
def asset_lock(root, exclusive=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'assets.lock').open('a') as lock:
        try:
            fcntl.flock(lock, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('数据或模型正在被后台任务使用，请等待完成或先取消任务') from None
        yield


def running_cli(root):
    """Also detect jobs that survived a web restart or are acquiring their lock."""
    root = Path(root).resolve()
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            args = (entry/'cmdline').read_bytes().decode().split('\0')
            if not any(name in args for name in ('retarget.baseline.geort.cli', 'moxian_web.geort_cli', 'moxian_web.cart_cli')):
                continue
            if not any(action in args for action in ('train', 'evaluate', 'preprocess', 'import-raw')):
                continue
            if '--root' in args and Path(args[args.index('--root')+1]).resolve() == root:
                return True
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return False


def delete_saved(root, kind, ids, protected=(), before_delete=None):
    from .geort_service import safe_id
    if kind not in ('recordings', 'models'):
        raise ValueError('无效资产类型')
    if not isinstance(ids, list) or not ids or len(ids) != len(set(map(str, ids))):
        raise ValueError('请选择不重复的录制数据或模型')
    ids = [safe_id(key) for key in ids]
    if set(ids) & set(protected):
        raise ValueError('正在录制或保存的数据不能删除，请先停止并等待保存完成')
    root = Path(root)
    with asset_lock(root, exclusive=True):
        if running_cli(root):
            raise ValueError('后台任务正在使用数据或模型，请先取消或等待完成')
        # Older GeoRT processes use training.lock rather than assets.lock.
        with (root/'training.lock').open('a') as legacy:
            try:
                fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('GeoRT 训练正在运行，请先取消或等待完成') from None
            folders = []
            parent = (root/kind).resolve()
            if (root/kind).is_symlink() or parent.parent != root.resolve():
                raise ValueError('不能删除链接或目录外的数据')
            for key in ids:
                folder = root/kind/key
                if folder.is_symlink() or folder.resolve().parent != parent:
                    raise ValueError('不能删除链接或目录外的数据')
                if not folder.is_dir():
                    raise ValueError('录制数据或模型不存在：'+key)
                folders.append(folder)
            # Validate the entire selection before unloading a model or deleting.
            if before_delete:
                before_delete()
            for folder in folders:
                shutil.rmtree(folder)
    return dict(deleted=ids, kind=kind)
