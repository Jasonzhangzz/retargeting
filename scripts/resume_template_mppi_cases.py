"""Recover only unfinished cases after a concurrent GPU cohort has exited."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json
from retarget.config.run import load
from retarget.env.grasp.template_mppi_runner import validate
from retarget.env.grasp.template_mppi_parallel import _solve
from retarget.env.grasp.primitives import jobs


def recover(config_path,root,wait_for):
 root=Path(root).resolve();wait_for=Path(wait_for).resolve();started=time.perf_counter()
 old_status=json.loads((root/'status.json').read_text())
 write_json(root/'status.json',dict(state='waiting_for_shared_gpu',reason='recover unfinished cases after concurrent cohort exits',previous_status=old_status))
 while True:
  status=json.loads((wait_for/'status.json').read_text())
  if status['state'] in ('completed','failed','aborted'):break
  time.sleep(2)
 for path,digest in json.loads((root/'source_sha256_at_start.json').read_text()).items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
   raise RuntimeError(f'Source changed since original solve: {path}')
 waited=time.perf_counter()-started
 config=load(config_path);args=validate(config);tasks,_=jobs(root,args.templates,args)
 manifest=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'summary.json').read_text())
 completed={int(r['index']):r for r in summary['cases'] if r['solver_completed']}
 entries={int(r['name'].partition('_')[0]):r for r in manifest['cases']}
 references=json.loads((root/'template_initialization/manifest.json').read_text())['references']
 reference=Path(references[0])
 assert len({str(p) for p in references})==1,'Recovery driver expects one canonical template reference'
 pending=[i for i in range(len(tasks)) if i not in completed]
 print('Recovering',len(pending),'unfinished cases:',pending,flush=True)
 write_json(root/'status.json',dict(state='running',planned=len(tasks),finished=len(completed),recovery=True))
 compute=time.perf_counter()
 for index in pending:
  mesh,name,primitive=tasks[index]
  with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as pool:
   result=pool.submit(_solve,config,root,index,mesh,name,primitive,reference).result()
  if not result['summary']['solver_completed']:
   raise RuntimeError(result['summary'])
  completed[index]=result['summary'];entries[index]=result['entries'][0]
  manifest['cases']=[entries[k] for k in sorted(entries)]
  summary.update(cases=[completed[k] for k in sorted(completed)],finished=len(completed),solver_completed=len(completed),successes=sum(r['task_success'] for r in completed.values()))
  write_json(root/'manifest.json',manifest);write_json(root/'summary.json',summary)
  write_json(root/'status.json',dict(state='running',planned=len(tasks),finished=len(completed),recovery=True))
  print('Recovered case',index,flush=True)
 resources=json.loads((root/'resources.json').read_text())
 resources.update(recovery=dict(previous_status=old_status,wait_for=str(wait_for),wait_seconds=waited,
  compute_seconds=time.perf_counter()-compute,resumed_case_indices=pending,completed_cases_untouched=True))
 write_json(root/'resources.json',resources)
 write_json(root/'status.json',dict(state='completed',planned=len(tasks),finished=len(completed),recovery=True))
 print(str(root),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config',type=Path);parser.add_argument('root',type=Path);parser.add_argument('--wait-for',type=Path,required=True)
 args=parser.parse_args();recover(args.config,args.root,args.wait_for)
