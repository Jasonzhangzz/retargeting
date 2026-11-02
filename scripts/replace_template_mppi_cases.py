"""Compose audited results, replacing exact physical instances only."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(row):
 p=row['primitive'];return row['template'],p['kind'],p['radius_m'],p['height_m']


def compose(base,correction,output):
 base,correction,output=map(lambda p:Path(p).resolve(),(base,correction,output))
 if output.exists() or output.parent!=ROOT/'output':raise ValueError('Use a new output directory')
 for root in (base,correction):
  a=read(root/'independent_result_audit.json')
  if not a['passed'] or not a['complete']:raise ValueError(f'Unaudited source {root}')
 original=read(base/'manifest.json')['cases'];updated={key(r):r for r in read(correction/'manifest.json')['cases']}
 if set(updated)-{key(r) for r in original}:raise ValueError('Each corrected physical instance must exist in base')
 entries=[];rows=[];references=[];fk_sizes=set();families=set()
 for index,previous in enumerate(original):
  replacing=key(previous) in updated;source=correction if replacing else base;row=updated[key(previous)] if replacing else previous
  entry=dict(row,name=previous['name'],source_batch=str(source),source_case_name=row['name'])
  for field in ('model','trajectory','summary'):entry[field]=str((source/row[field]).resolve())
  patch=(Path(entry['trajectory']).parent/'patch_contact_coverage.json')
  if replacing and patch.exists():entry['patch_coverage']=str(patch)
  elif row.get('patch_coverage'):entry['patch_coverage']=str((source/row['patch_coverage']).resolve())
  entries.append(entry);record=read(entry['summary']);rows.append(dict(record,index=index,case_id=previous['name']))
  initial=read(Path(entry['trajectory']).with_name('initialization.json'));references.append(initial['reference']);fk_sizes.add(initial['canonical_fk']['batch_size']);families.add((row['template'],row['primitive']['kind']))
 output.mkdir();write_json(output/'manifest.json',dict(version=1,task='grasp',method='mppi',cases=entries))
 summary=read(base/'summary.json');summary.update(cases=rows,planned=len(rows),finished=len(rows),solver_completed=sum(r['solver_completed'] for r in rows),successes=sum(r['task_success'] for r in rows),recomputed_cases=len(updated),retained_cases=len(rows)-len(updated));write_json(output/'summary.json',summary)
 write_json(output/'template_initialization/manifest.json',dict(policy='template_1x_solved_grasp_reference_v2',fits=len(families),cases=len(rows),references=references,source_reference_files=len(set(references)),fk_backend='torch_cuda_batched_WujiContactKinematics',fk_batch_sizes=sorted(fk_sizes)))
 resources=read(correction/'resources.json');resources.update(runtime_scope='Corrected physical instances only; retained cases keep their source resource reports');write_json(output/'resources.json',resources);write_json(output/'skipped.json',read(base/'skipped.json'))
 provenance=dict(kind='audited_exact_case_replacement',base_batch=str(base),corrected_batch=str(correction),replaced_physical_instances=[list(k) for k in updated],retained_cases=len(rows)-len(updated),source_audit_sha256={str(p):sha(p) for p in (base/'independent_result_audit.json',correction/'independent_result_audit.json')},source_provenance_sha256={str(p):sha(p) for p in (base/'provenance.json',correction/'provenance.json')},source_policy='Retained source artifacts and their original solve hashes remain authoritative; validation snapshot follows correction')
 write_json(output/'provenance.json',provenance);write_json(output/'composition.json',provenance);sources=read(correction/'source_sha256_at_start.json');sources[str(Path(__file__).resolve())]=sha(__file__);write_json(output/'source_sha256_at_start.json',sources);write_json(output/'status.json',dict(state='completed',planned=len(rows),finished=len(rows)))
 print(output)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('correction',type=Path);p.add_argument('output',type=Path);a=p.parse_args();compose(a.base,a.correction,a.output)
