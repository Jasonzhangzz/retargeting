"""Fixed, discoverable result locations shared by grasp solving and Viser."""
from pathlib import Path
import os
import uuid

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT/'outputs/grasp_results'
METHODS = ('cem_explicit', 'cem', 'adamw')
HANDS = ('wuji1', 'wuji2')


def method_name(optimizer, explicit_pose=True):
    return 'cem_explicit' if optimizer == 'cem' and explicit_pose else optimizer


def _symlink(target, destination):
    """Atomically replace a discovery link without touching saved results."""
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}')
    temporary.symlink_to(target, target_is_directory=True)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def register_results(output, method, hand, *, store=None):
    """Expose a batch (including one still being solved) at the fixed root.

    Existing archives stay in place. A directory link exposes their summary,
    scenes and playback files without copying or rewriting any saved poses.
    """
    if method not in METHODS or hand not in HANDS:
        raise ValueError(f'Unknown grasp result method/hand: {method}/{hand}')
    output = Path(output).expanduser().resolve()
    batch = output.parent.name if output.name in ('results', 'cem', 'adamw') else output.name
    if batch == 'latest' or batch.startswith('.'):
        raise ValueError(f'Reserved result batch name: {batch}')
    folder = Path(store if store is not None else RESULTS_ROOT)/method/hand
    folder.mkdir(parents=True, exist_ok=True)
    link = folder/batch
    if link.absolute() == output:
        pass  # The solver will create the actual batch directly in the store.
    elif link.exists() or link.is_symlink():
        if link.resolve() != output:
            raise ValueError(f'Result batch name already exists: {link}')
    else:
        _symlink(output, link)
    _symlink(batch, folder/'latest')
    return link


def discover_result_sources(*, store=None):
    """List candidate batches, latest first within each method/hand pair.

    The viewer selects the first nonempty batch per pair. An empty new run
    therefore leaves the previous batch viewable until its first result arrives.
    """
    root = Path(store if store is not None else RESULTS_ROOT)
    sources = []
    for method in METHODS:
        for hand in HANDS:
            folder = root/method/hand
            if not folder.is_dir():
                continue
            latest = folder/'latest'
            batches = sorted((p for p in folder.iterdir()
                if p.name != 'latest' and not p.name.startswith('.') and p.is_dir()),
                key=lambda p: (p.lstat().st_mtime_ns, p.resolve() == latest.resolve(), p.name), reverse=True)
            if not batches and latest.is_dir():
                batches = [latest]
            for batch in batches:
                sources.append(dict(method=method, hand=hand, batch=batch.name, path=batch))
    return sources
