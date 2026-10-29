#!/usr/bin/env python3
"""Vendor pinned official WUJI Hand 2.2 and build the grasp interface adapter.

The source MJCF is retained verbatim. Only aliases and six existing wrist DOFs
are added; official finger transforms, inertias, limits and actuators are kept.
Fixed sensor pads and mount remain physical parts of the hand.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.grasp_hand import HAND_BODIES, rigid_mesh_arrays

COMMIT = '5c4dfa16917355d073e2e1d7e3d03897b6cf9508'
BASE = f'https://raw.githubusercontent.com/wuji-technology/wuji-description/{COMMIT}'
SOURCE = 'hand2/hand2_2/body'
DEST = ROOT / 'assets/wuji2'
FINGERS = ('thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky')


def fetch(relative, target):
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urlopen(f'{BASE}/{relative}', timeout=45) as response:
                data = response.read()
            target.write_bytes(data)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


def build(download=False):
    source = DEST / 'official/right.xml'
    if download:
        fetch(f'{SOURCE}/mjcf/right.xml', source)
        fetch('LICENSE', DEST / 'LICENSE')
        assets = ET.parse(source).getroot().findall('./asset/mesh')
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda mesh: fetch(f'{SOURCE}/meshes/right/{mesh.get("file")}',
                DEST / 'meshes/right' / mesh.get('file')), assets))
    tree = ET.parse(source).getroot()
    tree.set('model', 'wujihand2_2-grasp-right')
    tree.find('compiler').attrib.pop('meshdir', None)
    tree.find('compiler').set('autolimits', 'true')
    # The application's native ComFree backend supports Euler integration.
    # Static CEM uses FK only; AdamW and replay share this executor setting.
    tree.find('option').set('integrator', 'Euler')
    for mesh in tree.findall('./asset/mesh'):
        mesh.set('file', 'meshes/right/' + mesh.get('file'))
    aliases = {'r_wrist': 'right_palm_link'}
    for i, finger in enumerate(FINGERS, 1):
        for k, part in enumerate(('proximal', 'proximal_abd', 'middle', 'distal'), 1):
            aliases[f'r_{finger}_{part}'] = f'right_finger{i}_link{k}'
        aliases[f'r_{finger}_tip'] = 'right_' + ('thumb', 'index', 'middle', 'ring', 'pinky')[i-1] + '_tip'
        bodies = [b for b in tree.iter('body') if b.get('name') in
                  [f'r_{finger}_{part}' for part in ('proximal', 'proximal_abd', 'middle', 'distal')]]
        for k, body in enumerate(bodies, 1):
            aliases[body.find('joint').get('name')] = f'right_finger{i}_joint{k}'
    for element in tree.iter():
        # Mesh asset identifiers retain their upstream names.
        for attr in ('name', 'joint', 'body1', 'body2'):
            if element.tag not in ('mesh', 'material') and element.get(attr) in aliases:
                element.set(attr, aliases[element.get(attr)])
    world = tree.find('worldbody')
    mount = world.find('body')
    palm = mount.find('body')
    mount.remove(palm)
    world.remove(mount)
    # Define the task wrist at r_wrist; retain the mounting plate at its
    # exact inverse fixed transform, so the finger FK needs no root offset.
    r = Rotation.from_quat(np.fromstring(palm.get('quat', '1 0 0 0'), sep=' '), scalar_first=True)
    p = np.fromstring(palm.get('pos', '0 0 0'), sep=' ')
    mount.set('pos', ' '.join(map(str, -r.inv().apply(p))))
    mount.set('quat', ' '.join(map(str, r.inv().as_quat(scalar_first=True))))
    palm.attrib.pop('pos', None)
    palm.attrib.pop('quat', None)
    palm.append(mount)
    # Reuse only the six wrist wrapper bodies/controllers of the existing
    # application, never its finger model or first-generation surfaces.
    legacy = ET.parse(ROOT / 'assets/wuji/right.xml').getroot()
    wrapper = copy.deepcopy(legacy.find('./worldbody/body'))
    parent = wrapper
    for _ in range(5):
        parent = parent.find('body')
    parent.remove(parent.find('body'))
    parent.append(palm)
    world.append(wrapper)
    actuator = tree.find('actuator')
    for old in reversed(list(legacy.find('actuator'))[:6]):
        actuator.insert(0, copy.deepcopy(old))
    custom = ET.SubElement(tree, 'custom')
    ET.SubElement(custom, 'numeric', name='grasp_hand_generation', data='2')
    ET.indent(tree)
    # Compile before adding custom backend query geoms.
    raw = ET.tostring(tree, encoding='unicode')
    absolute = ET.fromstring(raw)
    for mesh in absolute.findall('./asset/mesh'):
        mesh.set('file', str(DEST / mesh.get('file')))
    model = mujoco.MjModel.from_xml_string(ET.tostring(absolute, encoding='unicode'))
    rng = np.random.default_rng(20261029)
    queries = DEST / 'qp_right'
    queries.mkdir(exist_ok=True)
    counts = {}
    for name in HAND_BODIES:
        body = model.body(name).id
        v, f = rigid_mesh_arrays(model, body)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        count = 1200 if name == 'right_palm_link' else 384
        # Area-weighted, deterministic samples on the actual full meshes.
        positions, face_ids = trimesh.sample.sample_surface(mesh, count, seed=rng)
        normals = mesh.face_normals[face_ids]
        np.savez_compressed(queries / f'{name}.npz', local_pos=positions.astype(np.float32),
            local_normal=normals.astype(np.float32),
            finger_idx=np.full(count, 0 if name == 'right_palm_link' else int(name[12])-1, np.int32),
            link_id=np.full(count, 0 if name == 'right_palm_link' else int(name[-1]), np.int32))
        counts[name] = count
        owner = next(b for b in tree.iter('body') if b.get('name') == name)
        ET.SubElement(owner, 'geom', name=f'query_{name}', type='querypoint',
            file=f'qp_right/{name}.npz', contype='1', conaffinity='4', condim='3',
            group='3', mass='0', rgba='0 0 0 0')
    # Native SDF/querypoint collisions replace the upstream convex hulls.
    for geom in tree.iter('geom'):
        if geom.get('type') == 'mesh':
            geom.set('contype', '0'); geom.set('conaffinity', '0')
    ET.indent(tree)
    (DEST / 'right.xml').write_text(ET.tostring(tree, encoding='unicode') + '\n')
    files = [source, *sorted((DEST / 'meshes/right').glob('*.STL'))]
    (DEST / 'source.json').write_text(json.dumps(dict(model='WUJI Hand 2.2 right, bare',
        repository='https://github.com/wuji-technology/wuji-description', commit=COMMIT,
        source_path=SOURCE, aliases=aliases, wrist_frame='r_wrist', query_counts=counts,
        query_seed=20261029, sha256={str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in files}), indent=2) + '\n')
    print(f'Built WUJI Hand 2.2: {model.nq} DOFs, {sum(counts.values())} surface queries', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch the pinned official assets first')
    build(parser.parse_args().download)
