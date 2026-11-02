#!/usr/bin/env python3
"""Compose disjoint audited GPU type shards without copying final artifacts."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retarget.common.results import write_json


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def merge(shards,root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    entries=[];rows=[];nodes=[];groups=[];sources={};parents={};resources=[];names=set()
    for shard in map(lambda p:Path(p).resolve(),shards):
        audit=read(shard/'independent_result_audit.json')
        if not audit['passed'] or not audit['complete']:raise ValueError(f'Unaudited shard {shard}')
        parents[str(shard/'independent_result_audit.json')]=sha(shard/'independent_result_audit.json')
        plan=read(shard/'continuation_plan.json');mapping={n['index']:len(nodes)+i for i,n in enumerate(plan['nodes'])}
        for n in plan['nodes']:
            value=dict(n,index=mapping[n['index']])
            if 'parent_index' in value:value['parent_index']=mapping[value['parent_index']]
            nodes.append(value)
        for group in plan['groups']:
            if group['template'] in names:raise ValueError('A template appears in multiple shards')
            names.add(group['template'])
            groups.append(dict(group,anchor_index=mapping[group['anchor_index']],indices=[mapping[i] for i in group['indices']]))
        for e in read(shard/'manifest.json')['cases']:
            e=dict(e)
            for k in ('model','trajectory','summary','patch_coverage','continuation'):
                if e.get(k):e[k]=str((shard/e[k]).resolve())
            entries.append(e);rows.append(read(e['summary']))
        sources.update(read(shard/'source_sha256_at_start.json'));resources.append(read(shard/'resources.json'))
    order={n['name']:n['index'] for n in nodes};entries.sort(key=lambda e:order[e['name']])
    rowmap={e['name']:read(e['summary']) for e in entries};rows=[rowmap[e['name']] for e in entries]
    write_json(root/'manifest.json',dict(version=1,task='grasp',method='mppi',cases=entries))
    write_json(root/'summary.json',dict(method='mppi',objective=rows[0]['objective'],planned=len(nodes),finished=len(rows),
        solver_completed=sum(r['solver_completed'] for r in rows),successes=sum(r['task_success'] for r in rows),native_evaluated=0,cases=rows))
    write_json(root/'continuation_plan.json',dict(policy='template_adjacent_solved_grasp_v1',subdivisions=4,
        planned=len(nodes),anchors=len(groups),recomputed=len(nodes)-len(groups),nodes=nodes,groups=groups,source_audit_sha256=parents))
    write_json(root/'template_initialization/manifest.json',dict(policy='template_adjacent_solved_grasp_v1',fits=len(groups),cases=len(nodes),
        anchors=len(groups),fk_backend='torch_cuda_batched_WujiContactKinematics'))
    total_free=sum(r['baseline_free_bytes'] for r in resources);peaks=sum(r['peak_incremental_bytes'] for r in resources)
    write_json(root/'resources.json',dict(device=f'{len(resources)} independent A800 device shards',
        memory_scope='sum of per-device peak increments; conservative upper bound on simultaneous total allocation',
        baseline_free_bytes=total_free,budget_bytes=sum(r['budget_bytes'] for r in resources),peak_incremental_bytes=peaks,
        peak_fraction_of_initial_free=peaks/total_free,wall_seconds=max(r['wall_seconds'] for r in resources),per_device=resources))
    write_json(root/'source_sha256_at_start.json',sources)
    write_json(root/'provenance.json',dict(kind='audited_disjoint_gpu_shards',source_audit_sha256=parents))
    write_json(root/'skipped.json',read(Path(shards[0])/'skipped.json'))
    write_json(root/'status.json',dict(state='completed',planned=len(nodes),finished=len(rows)))
    print(root,len(entries),'cases',len(groups),'types',flush=True)
    return entries


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('shards',type=Path,nargs='+')
    a=p.parse_args();merge(a.shards,a.output)
