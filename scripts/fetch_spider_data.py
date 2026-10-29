#!/usr/bin/env python3
"""Download the pinned, public SPIDER examples and their MJCF dependencies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import posixpath
import time
from urllib.request import urlopen
from urllib.error import HTTPError
import xml.etree.ElementTree as ET

HF_REVISION = '8b0cdf2b24cc86d1c859050c5b0317ddbec59a63'
SOURCE_REVISION = '44717007de41cbef7565dff7ff9f4453557a2d3d'
HF = 'https://huggingface.co/datasets/retarget/retarget_example/resolve/'
SOURCE = 'https://raw.githubusercontent.com/facebookresearch/spider/'
SAMPLES = {
    'humanoid': 'processed/amass/unitree_g1/humanoid/sprint',
    'humanoid_object': 'processed/omomo/unitree_g1/humanoid_object/move_largebox',
    'bimanual': 'processed/arcticv2/xhand/bimanual/s01-box_use_01',
}
SOURCE_FILES = [
    'LICENSE', 'README.md', 'spider/config.py', 'spider/io.py',
    'spider/interp.py','spider/math.py','spider/__init__.py',
    'spider/simulators/mjwp.py', 'spider/simulators/hdmi.py',
    'spider/optimizers/sampling.py', 'spider/preprocess/ik.py',
    'spider/process_datasets/gmr.py', 'spider/process_datasets/arcticv2.py',
    'examples/run_mjwp.py', 'examples/config/override/humanoid.yaml',
    'examples/config/override/humanoid_object.yaml',
    'examples/config/override/arcticv2_fast.yaml',
]


def read_url(url):
    for attempt in range(4):
        try:
            with urlopen(url, timeout=90) as response:
                return response.read()
        except OSError as error:
            if isinstance(error, HTTPError) and error.code == 404:
                raise
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def fetch(root):
    root = root.resolve()
    records = {}

    def download(path, source=False):
        base = SOURCE + SOURCE_REVISION + '/' if source else HF + HF_REVISION + '/'
        rel = 'source/' + path if source else path
        dest = root / rel
        # Keep the remote path confined to the data directory.
        if not dest.resolve().is_relative_to(root):
            raise ValueError(path)
        url = base + path
        try:
            content = read_url(url)
        except HTTPError as error:
            # ARCTICv2 ships objects and scenes, but its robot meshes live in
            # the SPIDER source distribution rather than the HF dataset.
            if error.code != 404 or source or '/assets/robots/' not in path:
                raise RuntimeError(f'Cannot download {url}') from error
            asset = 'spider/assets/robots/' + path.split('/assets/robots/', 1)[1]
            url = SOURCE + SOURCE_REVISION + '/' + asset
            content = read_url(url)
        except OSError as error:
            raise RuntimeError(f'Cannot download {base + path}') from error
        if content.startswith(b'version https://git-lfs.github.com/spec/'):
            raise ValueError(f'LFS pointer instead of data: {path}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        records[rel] = dict(path=rel, url=url, bytes=len(content),
                            sha256=hashlib.sha256(content).hexdigest())
        return dest

    files = []
    for task, folder in SAMPLES.items():
        files += [folder + '/0/trajectory_kinematic.npz']
        files += [folder + ('/0/trajectory_mjwp_fast.npz' if task == 'bimanual'
                           else '/0/trajectory_mjwp.npz')]
        files += [folder + '/scene.xml']
        if task=='humanoid':files += [folder+'/task_info.json']
        if task == 'bimanual':
            files += [folder + '/0/config.yaml', folder + '/task_info.json']
    files += ['processed/amass/assets/robots/unitree_g1/scene.xml',
              'processed/amass/assets/robots/unitree_g1/LICENSE',
              'processed/omomo/assets/robots/unitree_g1/LICENSE']
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, files))
        list(pool.map(lambda path: download(path, True), SOURCE_FILES))

    # Follow actual XML mesh/texture/include paths, retaining the upstream tree.
    pending = [p for p in records if p.endswith('.xml')]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ET.parse(root / path).getroot()
        compiler = tree.find('compiler')
        meshdir = compiler.get('meshdir', '') if compiler is not None else ''
        texdir = compiler.get('texturedir', '') if compiler is not None else ''
        dependencies = set()
        for element in tree.iter():
            filename = element.get('file')
            if filename is None:
                continue
            subdir = meshdir if element.tag == 'mesh' else texdir if element.tag == 'texture' else ''
            dep = posixpath.normpath(posixpath.join(posixpath.dirname(path), subdir, filename))
            if dep not in records:
                dependencies.add(dep)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(download, sorted(dependencies)))
        pending += [p for p in dependencies if p.endswith('.xml')]
    manifest = dict(repository='https://github.com/facebookresearch/spider',
                    source_revision=SOURCE_REVISION,
                    dataset='https://huggingface.co/datasets/retarget/retarget_example',
                    dataset_revision=HF_REVISION, samples=SAMPLES,
                    files=sorted(records.values(), key=lambda item: item['path']))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Downloaded {len(records)} files to {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/spider')
    fetch(parser.parse_args().output)
