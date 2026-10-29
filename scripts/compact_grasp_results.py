#!/usr/bin/env python3
"""Replace discovered archive links with numeric-only batches; keep originals."""
from pathlib import Path
import os
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from retarget.grasp_results import discover_result_sources
    from retarget.grasp_compact import FILE, legacy_rows, write_batch, audit_records
    from scripts.validate_grasp_matrix import audit_batch
    total = 0
    for source in discover_result_sources():
        link = source['path']
        if (link/FILE).is_file():
            continue
        if not link.is_symlink():
            raise ValueError(f'Expected an archive discovery link, preserving original directory: {link}')
        original = link.resolve()
        stamp = link.lstat().st_mtime_ns
        rows = legacy_rows(original)
        audited = audit_batch(original)
        if any(not r.get('saved_data_valid') or not r.get('closure_agrees') for r in audited):
            raise RuntimeError(f'Independent audit failed, archive link unchanged: {link}')
        with tempfile.TemporaryDirectory(prefix='.compact_', dir=link.parent) as workspace:
            workspace = Path(workspace)
            write_batch(rows, workspace, method=source['method'], audited=audited)
            assert all(r['saved_data_valid'] for r in audit_records(workspace))
            # Only replace our directory link; original archive data stays put.
            backup = link.with_name(f'.archive_link_{uuid.uuid4().hex}')
            link.rename(backup)
            try:
                link.mkdir()
                os.replace(workspace/FILE, link/FILE)
            except BaseException:
                if link.is_dir() and not list(link.iterdir()):
                    link.rmdir()
                backup.rename(link)
                raise
            backup.unlink()
            os.utime(link, ns=(stamp, stamp))
        total += len(rows)
        print(f'{source["method"]}/{source["hand"]}/{source["batch"]}: '
              f'{len(rows)} grasps, {(link/FILE).stat().st_size} bytes; original retained at {original}', flush=True)
    print(f'Converted {total} grasps', flush=True)


if __name__ == '__main__':
    main()
