"""Compose a complete viewer batch, replacing whole template types after audit.

Unchanged cases reference their original immutable artifacts. Corrected types
replace every old size, including changed grids or shape families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retarget.common.results import write_json


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compose(base, correction, output):
    base, correction, output = map(lambda p: Path(p).resolve(), (base, correction, output))
    if output.parent != ROOT / 'output' or output.exists():
        raise ValueError('Composition output must be a new direct child of output/')
    for source in (base, correction):
        audit = read(source / 'independent_result_audit.json')
        if not audit['passed'] or not audit.get('complete', True):
            raise ValueError(f'Incomplete or unaudited source: {source}')
    original = read(base / 'manifest.json')['cases']
    updated = read(correction / 'manifest.json')['cases']
    names = {row['template'] for row in updated}
    if names - {row['template'] for row in original}:
        raise ValueError('Corrected types must exist in the base batch')
    replacements = {name: sorted((row for row in updated if row['template'] == name),
                    key=lambda row: (row['primitive']['kind'], row['primitive']['size_scale']))
                    for name in names}
    # Whole-type replacement also supports an expanded size grid. Preserve
    # existing case identities when dimensions match; name new sizes explicitly.
    planned, emitted = [], set()
    for previous in original:
        name = previous['template']
        if name not in names:
            planned.append((previous, base, previous['name']))
        elif name not in emitted:
            emitted.add(name)
            old = [row for row in original if row['template'] == name]
            for replacement in replacements[name]:
                primitive = replacement['primitive']
                match = next((row for row in old if all(row['primitive'][k] == primitive[k]
                    for k in ('kind', 'radius_m', 'height_m'))), None)
                case_id = match['name'] if match else f"added_{name}_{primitive['kind']}_r{primitive['radius_m']:.7f}_h{primitive['height_m']:.7f}"
                planned.append((replacement, correction, case_id))
    entries, rows, refs, fk_sizes = [], [], [], set()
    for index, (source_entry, source, case_id) in enumerate(planned):
        entry = dict(source_entry, name=case_id, source_batch=str(source),
                     source_case_name=source_entry['name'])
        for field in ('model', 'trajectory', 'summary'):
            entry[field] = str((source / source_entry[field]).resolve())
        if source_entry.get('patch_coverage'):
            entry['patch_coverage'] = str((source/source_entry['patch_coverage']).resolve())
        entries.append(entry)
        row = read(entry['summary'])
        rows.append(dict(row, index=index, case_id=case_id, source_batch=str(source)))
        initial = read(Path(entry['trajectory']).with_name('initialization.json'))
        refs.append(initial['reference'])
        fk_sizes.add(initial['canonical_fk']['batch_size'])
    output.mkdir()
    write_json(output / 'manifest.json', dict(version=1, task='grasp', method='mppi', cases=entries))
    write_json(output / 'summary.json', dict(method='mppi', objective=rows[0]['objective'],
        planned=len(rows), finished=len(rows), solver_completed=sum(r['solver_completed'] for r in rows),
        successes=sum(r['task_success'] for r in rows), native_evaluated=0, cases=rows,
        success_criterion='projected_soft_contact_closure_v2', recomputed_cases=len(updated),
        retained_cases=len(rows)-len(updated)))
    write_json(output / 'template_initialization/manifest.json', dict(
        policy='template_1x_solved_grasp_reference_v2', fits=len({(row['template'],row['primitive']['kind']) for row in rows}),
        source_reference_files=len(set(refs)), cases=len(rows), references=refs,
        fk_backend='torch_cuda_batched_WujiContactKinematics', fk_batch_sizes=sorted(fk_sizes)))
    resources = read(correction / 'resources.json')
    resources.update(runtime_scope=f'Only corrected templates {sorted(names)}; remaining cases retained from audited baseline',
        recomputed_cases=len(updated), retained_cases=len(rows)-len(updated))
    write_json(output / 'resources.json', resources)
    if (correction/'shared_cuda_budget.json').is_file():
        write_json(output/'shared_cuda_budget.json', read(correction/'shared_cuda_budget.json'))
    write_json(output / 'skipped.json', read(base / 'skipped.json'))
    composition = dict(base_batch=str(base), corrected_batch=str(correction),
        replaced_templates=sorted(names), retained_cases=len(rows)-len(updated), recomputed_cases=len(updated),
        policy='All sizes of corrected types replaced; remaining case artifacts unchanged',
        source_provenance_sha256={str(p): sha(p) for p in
            (base/'provenance.json', correction/'provenance.json')},
        source_audit_sha256={str(p): sha(p) for p in
            (base/'independent_result_audit.json', correction/'independent_result_audit.json')})
    write_json(output / 'composition.json', composition)
    write_json(output / 'provenance.json', dict(kind='audited_type_replacement', **composition))
    sources = read(correction / 'source_sha256_at_start.json')
    sources[str(Path(__file__).resolve())] = sha(__file__)
    write_json(output / 'source_sha256_at_start.json', sources)
    write_json(output / 'status.json', dict(state='completed', recomputed_cases=len(updated),
                                           retained_cases=len(rows)-len(updated)))
    print(json.dumps(composition, indent=2))
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('correction', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    compose(args.base, args.correction, args.output)
