#!/usr/bin/env python3
"""Publish every final dense-grid grasp with portable paths and original evidence.

Only selected final results and their required references are exported. This
rewrites dependency paths; final poses, trajectories and force witnesses stay
unchanged. Run from the repository root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(source, destination, source_root):
    source, destination = source.resolve(), destination.resolve()
    source_root = Path(source_root)
    if destination.exists():
        raise FileExistsError(destination)

    def local(value, base=source):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(source_root):
            return ROOT / path.relative_to(source_root)
        return path if path.is_absolute() else base / path

    audit = read(source / 'independent_result_audit.json')
    if not (audit['passed'] and audit['complete']):
        raise ValueError('A complete independently audited source is required')
    entries = read(source / 'manifest.json')['cases']
    destination.mkdir(parents=True)
    paths, new_entries = {}, {}
    final_hashes = []
    for entry in entries:
        old = local(entry['trajectory']).parent
        new = destination / 'cases' / entry['name']
        new.mkdir(parents=True)
        for name in ('scene.xml', 'input_object.obj', 'grasp.npz', 'trajectory.npz',
                     'summary.json', 'projection_force_closure.json', 'patch_contact_coverage.json',
                     'template_contacts.json', 'initialization.json', 'size_ik.json',
                     'object_frame.json'):
            if (old / name).is_file():
                shutil.copy2(old / name, new / name)
        paths[entry['name']] = (old, new)
        updated = dict(entry)
        for field in ('model', 'trajectory', 'summary', 'patch_coverage'):
            if entry.get(field):
                updated[field] = str((new / local(entry[field]).name).relative_to(destination))
        new_entries[entry['name']] = updated
        for name in ('grasp.npz', 'trajectory.npz', 'projection_force_closure.json'):
            final_hashes.append(dict(case=entry['name'], file=name,
                source_sha256=sha(old / name), published_sha256=sha(new / name)))

    def parent_entry(entry):
        result = dict(new_entries[entry['name']])
        for field in ('model', 'trajectory', 'summary', 'patch_coverage', 'continuation'):
            if result.get(field):
                result[field] = str((destination / result[field]).relative_to(ROOT))
        return result

    # First allocate all continuity paths, including parents exported later.
    for entry in entries:
        if entry.get('continuation'):
            new_entries[entry['name']]['continuation'] = 'cases/' + entry['name'] + '/continuation.json'

    for entry in entries:
        old, new = paths[entry['name']]
        initial = read(old / 'initialization.json')
        old_ref = local(initial['reference'], ROOT)
        refdir = new / 'reference'
        refdir.mkdir()
        for name in ('template_contacts.json', 'candidates.npz', 'object_frame.json'):
            if (old_ref.parent / name).is_file():
                shutil.copy2(old_ref.parent / name, refdir / name)
        ref = read(old_ref)
        continuity = read(local(entry['continuation'])) if entry.get('continuation') else None
        evidence = ref.get('canonical_grasp_reference')
        if evidence:
            evidence = dict(evidence)
            if continuity:
                parent = continuity['parent_entry']['name']
                parent_dir = paths[parent][1]
                evidence.update(source=str((parent_dir / 'projection_force_closure.json').relative_to(ROOT)),
                    sha256=sha(parent_dir / 'projection_force_closure.json'),
                    qpos_source=str((parent_dir / 'grasp.npz').relative_to(ROOT)),
                    qpos_sha256=sha(parent_dir / 'grasp.npz'))
            else:
                for field, hash_field in (('source', 'sha256'), ('qpos_source', 'qpos_sha256')):
                    if evidence.get(field):
                        source_file = local(evidence[field], ROOT)
                        target = refdir / ('canonical_' + field + source_file.suffix)
                        shutil.copy2(source_file, target)
                        evidence[field] = str(target.relative_to(ROOT))
                        evidence[hash_field] = sha(target)
            ref['canonical_grasp_reference'] = evidence
        write_json(refdir / 'reference.json', ref)
        initial.update(reference=str((refdir / 'reference.json').relative_to(ROOT)),
            reference_sha256=sha(refdir / 'reference.json'), canonical_grasp_reference=evidence)
        write_json(new / 'initialization.json', initial)
        if continuity:
            continuity['parent_entry'] = parent_entry(continuity['parent_entry'])
            continuity.update(parent_reference=initial['reference'], reference_sha256=initial['reference_sha256'])
            continuity.pop('attempts', None)
            write_json(new / 'continuation.json', continuity)
        tree = ET.parse(new / 'scene.xml')
        for element in tree.iter():
            if 'file' not in element.attrib:
                continue
            asset = local(element.attrib['file'], old)
            target = new / asset.relative_to(old) if asset.is_relative_to(old) else asset
            if not target.is_file():
                raise FileNotFoundError(target)
            element.set('file', os.path.relpath(target, new))
        tree.write(new / 'scene.xml', encoding='unicode')

    for name in ('summary.json', 'resources.json', 'skipped.json', 'status.json',
                 'source_sha256_at_start.json', 'continuation_plan.json',
                 'dense_report.md', 'dense_report.json', 'dense_results.csv', 'remote_job.json'):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    shutil.copytree(source / 'template_initialization', destination / 'template_initialization')
    write_json(destination / 'manifest.json', dict(version=1, task='grasp', method='mppi',
        cases=list(new_entries.values())))
    archive = destination / 'source_audits'
    archive.mkdir()
    lineage = {}
    for i, path in enumerate([source / 'independent_result_audit.json',
            *map(local, read(source / 'continuation_plan.json')['source_audit_sha256'])]):
        target = archive / f'audit_{i}.json'
        shutil.copy2(path, target)
        lineage[str(target.relative_to(ROOT))] = sha(target)
    write_json(destination / 'composition.json', dict(policy='portable dense final results; final selection only',
        source_audit_sha256=lineage))
    preserved = all(row['source_sha256'] == row['published_sha256'] for row in final_hashes)
    write_json(destination / 'materialization.json', dict(source=str(source), source_root=str(source_root),
        cases=len(entries), preserved_final_arrays_and_force_witnesses=preserved, files=final_hashes))
    if not preserved:
        raise ValueError('Copied poses, trajectories or force witnesses failed SHA256 verification')
    print(json.dumps(dict(cases=len(entries), destination=str(destination), preserved=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--source-root', required=True, help='Repository root on the original solve machine')
    args = parser.parse_args()
    export(args.source, args.destination, args.source_root)


if __name__ == '__main__':
    main()
