#!/usr/bin/env python3
"""Check the frozen dataset's checksums, pairings, and joint-array dimensions."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    files = json.loads((root / 'MANIFEST.sha256.json').read_text())
    for relative, expected in files.items():
        path = root / relative
        assert path.is_file(), f'Missing: {relative}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Checksum: {relative}'
    manifest = json.loads((root / 'source-manifest.json').read_text())
    count = 0
    for entry in manifest['entries']:
        with np.load(root / 'glove' / entry['file'], allow_pickle=False) as glove, \
             np.load(root / 'wuji2' / entry['file'], allow_pickle=False) as wuji:
            n = entry['samples']
            for z, dof in [(glove, 29), (wuji, 20)]:
                assert z['target_qpos'].shape == (n, dof), entry['type']
                assert np.isfinite(z['target_qpos']).all(), entry['type']
                assert z['target_base_pose'].shape == (n, 7), entry['type']
                assert z['mano_vertices'].shape == (n, 778, 3), entry['type']
                assert z['mano_joints'].shape == (n, 21, 3), entry['type']
                assert len(z['joint_names']) == dof, entry['type']
            for key in ['mano_vertices', 'mano_joints', 'object_pose_world', 'radius_m', 'height_m']:
                assert np.array_equal(glove[key], wuji[key], equal_nan=True), (entry['type'], key)
            count += n
    assert len(manifest['entries']) == 31 and count == 1247
    print(f'Verified {len(files)} file checksums; 31 types, 1,247 paired MANO/glove/Wuji2 poses.')


if __name__ == '__main__':
    main()
