#!/usr/bin/env python3
"""Restore the pinned CPU baseline's margin/gap combination in an isolated MJWarp."""
from pathlib import Path
import argparse,hashlib,json,zipfile


def sha(value):return hashlib.sha256(value).hexdigest()


def run(runtime,wheel):
    path=runtime/'mujoco_warp/_src/collision_core.py'
    with zipfile.ZipFile(wheel) as archive:original=archive.read('mujoco_warp/_src/collision_core.py')
    text=original.decode()
    replacements={
        'margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]':
        'margin = wp.max(geom_margin[margin_id, g1], geom_margin[margin_id, g2])',
        'gap = geom_gap[gap_id, g1] + geom_gap[gap_id, g2]':
        'gap = wp.max(geom_gap[gap_id, g1], geom_gap[gap_id, g2])'}
    for old,new in replacements.items():
        if text.count(old)!=1:raise RuntimeError('Unrecognized official collision implementation')
        text=text.replace(old,new)
    patched=text.encode();actual=path.read_bytes()
    if actual not in (original,patched):raise RuntimeError('Runtime differs from both official and recorded compatibility versions')
    folder=runtime/'compatibility_333';folder.mkdir(exist_ok=True)
    (folder/'collision_core.official.py').write_bytes(original)
    temporary=path.with_suffix('.tmp');temporary.write_bytes(patched);temporary.replace(path)
    receipt=dict(adapter='MuJoCo 3.3.3 geom margin/gap maximum, replacing later additive semantics',
        original_sha256=sha(original),patched_sha256=sha(patched),wheel_sha256=sha(wheel.read_bytes()),
        source='https://github.com/google-deepmind/mujoco/blob/3.3.3/src/engine/engine_collision_driver.c',
        original_formula_locations=dict(margin=1342,gap=1183),
        replacements=replacements,explicit_contact_pairs_unchanged=True,geom_parameter_values_unchanged=True,
        added_objective_terms=[],scope='Only this isolated runtime; no production environment files modified')
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--wheel',type=Path,required=True)
    a=p.parse_args();run(a.runtime,a.wheel)
