#!/usr/bin/env python3
"""Strict, reset-consistent real-EMF SDK dataset. Finite source search, no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import numpy as np
import build_wuji_real_sdk_dataset as base
from retarget.learning.wuji_ik.quality_data import passing_runs
from wuji_quality_teacher import solve,valid,retarget_history,QUALITY_KEYS


def save_arrays(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(temp,path)


def certify(root,w,policy_sha,depth=0):
    """Keep qualified label runs with their complete real EMF warm-up context."""
    base.stopped(root);d=solve(root,w);mask=valid(d);accepted=[]
    runs=passing_runs(mask,64)
    if not runs:return accepted
    identity=dict(policy=policy_sha,source_sha256=w['source_sha256'],source_h5=w['source_h5'],side=w['side'],context_start=w['start'],end=w['end'],calibration_sha256=base.digest(w['calibration']['urdf']))
    path=root/'quality_segments'/(base.identity(identity)+'.npz')
    if not path.exists():save_arrays(path,d)
    sha=base.digest(path)
    for a,b in runs:
        dt=np.diff(d['timestamp_ns'][a:b]).astype(float)*1e-9
        activity=float(np.mean(np.linalg.norm(np.diff(d['position_tx_m'][a:b],axis=0),axis=-1)/dt[:,None]))
        spread=float(np.mean(np.linalg.norm(d['position_tx_m'][a:b,:1]-d['position_tx_m'][a:b,1:],axis=-1)))
        accepted.append(dict(start=w['start']+a,end=w['start']+b,context_start=w['start'],activity=activity,spread=spread,quality_file=str(path),quality_sha256=sha,certificate=identity))
    return accepted


def process_collection(task):
    root,row,old_records,policy_sha,target=task;base.stopped(root)
    receipt=root/'collection_receipts'/(base.identity(row['path'])+'.json')
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value.get('policy_sha')==policy_sha and value.get('complete'):return value
        raise ValueError('Collection receipt has incompatible code/policy')
    conf=json.loads((root/'MANIFEST.json').read_text());held={os.path.normpath(p) for p in conf['heldout_episode_paths']}
    records={r['source_h5']:r for r in old_records};rng=np.random.default_rng(int(base.identity(row['path'])[:12],16))
    def score(r):
        values=[x.get('kinematic',{}).get('consistent_frames',0)/max(1,x.get('quality',{}).get('frames',1)) for x in r['sides'].values()]
        return sum(values)
    initial=list(records);rng.shuffle(initial);initial.sort(key=lambda p:score(records[p]),reverse=True)
    remaining=[p for p in row['h5_files'] if p not in records];rng.shuffle(remaining)
    # Inventory and held-out paths share the same recorded source prefixes.
    # Resolving every remote path triggers hundreds of thousands of JFS stat
    # requests before any SDK work. Use recorded paths, plus content exclusion.
    paths=[p for p in initial+remaining if os.path.normpath(str(Path(p).parent)) not in held]
    accepted=[];inspected=[];capacity={'right':0,'left':0};raw_window_count=0;good_sources=0
    for path in paths:
        base.stopped(root)
        record=records.get(path)
        if record is None:record=base.inspect_episode((root,row,path,policy_sha))
        source=Path(path)
        if record.get('source_sha256') and base.digest(source)!=record['source_sha256']:raise ValueError('Raw H5 content changed')
        if record.get('source_sha256') in conf['heldout_source_sha256']:
            inspected.append(dict(source_h5=path,status='heldout_content_excluded'))
            continue
        usable=False;new_sides={};entry=dict(source_h5=path,source_sha256=record.get('source_sha256'),raw_reasons=record.get('reasons',[]),certified_frames={})
        for side,detail in record['sides'].items():
            windows=list(detail['windows']);rng.shuffle(windows)
            certified=[]
            # Inspect additional ranges until this record contributes a useful
            # continuous sample, rather than selecting by network/q error.
            for window in windows:
                full=dict(window,source_h5=path,source_sha256=record['source_sha256'],side=side,calibration=detail['calibration'])
                chunks=certify(root,full,policy_sha);raw_window_count+=1;certified.extend(chunks)
                if sum(x['end']-x['start'] for x in certified)>=min(target,1024):break
            if certified:
                usable=True;new_sides[side]=dict(detail,windows=certified)
                n=sum(x['end']-x['start'] for x in certified);capacity[side]+=n;entry['certified_frames'][side]=n
        if usable:
            accepted.append(dict(record,sides=new_sides,usable_frames=sum(entry['certified_frames'].values())));good_sources+=1
        inspected.append(entry)
        if len(inspected)%4==0:
            base.write(root/'collection_progress'/(base.identity(row['path'])+'.json'),
                dict(collection=row['collection'],device=row['device'],inspected=len(inspected),
                     certified_capacity=capacity,good_original_records=good_sources,complete=False))
            print('QUALITY_RECORDS',row['device'],row['collection'],len(inspected),capacity,flush=True)
        # Cover collections with high-quality data without requiring an absent
        # or persistently unusable hand to contribute labels. Exact hand totals
        # are allocated later across all collections with certified capacity.
        enough=(min(capacity.values())>=target or (len(inspected)>=6 and sum(capacity.values())>=2*target))
        if (good_sources>=3 and enough) or (good_sources and len(inspected)>=6):break
    detail=dict(device=row['device'],collection=row['collection'],path=row['path'],h5_count=row['h5_count'],
        inspected=len(inspected),available_source_files=len(paths),full_inventory_exhausted=len(inspected)==len(paths),
        usable_episodes=len(accepted),candidate_frames=sum(capacity.values()),side_candidate_frames=capacity,
        raw_sdk_windows_evaluated=raw_window_count,status='quality_certified' if accepted else 'no_recorded_EMF' if not paths else 'no_certified_continuous_segment',
        source_audit=inspected)
    value=dict(complete=True,policy_sha=policy_sha,records=accepted,coverage=detail)
    base.write(receipt,value);return value


def collect(root,conf,policy_sha,workers):
    previous=Path(conf['quality_source_campaign']);old=json.loads((previous/'candidate_pool.json').read_text());inventory=json.loads((root/'inventory.json').read_text())
    by_col={}
    for r in old['records']:by_col.setdefault(r['collection_id'],[]).append(r)
    device_counts={d:sum(r['device']==d and r['h5_count']>0 for r in inventory['collections']) for d in {r['device'] for r in inventory['collections']}}
    tasks=[]
    for row in inventory['collections']:
        target=max(1024,int(np.ceil(2*47500/max(1,device_counts[row['device']]))))
        tasks.append((root,row,by_col.get(row['device']+'/'+row['collection'],[]),policy_sha,target))
    values=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(process_collection,task) for task in tasks]
        for f in as_completed(futures):
            value=f.result();values.append(value)
            coverage=[v['coverage'] for v in values];frames=sum(c['candidate_frames'] for c in coverage)
            base.write(root/'DATA_STATUS.json',dict(stage='strict_sdk_screening',complete=False,collections=len(values),expected_collections=len(tasks),quality_candidate_frames=frames,elapsed_seconds=time.perf_counter()-started))
            print('QUALITY_COLLECTION',len(values),value['coverage']['device'],value['coverage']['collection'],value['coverage']['status'],value['coverage']['side_candidate_frames'],flush=True)
    records=[r for value in values for r in value['records']];records.sort(key=lambda r:r['record_id']);coverage=[v['coverage'] for v in values];coverage.sort(key=lambda c:(c['device'],c['collection']))
    unique=[];seen={};duplicates=[]
    for r in records:
        sha=r['source_sha256']
        if sha in seen:
            duplicates.append(dict(excluded_record=r['record_id'],retained_record=seen[sha],source_sha256=sha))
        else:
            unique.append(r);seen[sha]=r['record_id']
    result=dict(complete=True,policy_sha=policy_sha,records=unique,collections=coverage,duplicate_original_contents_excluded=duplicates)
    base.write(root/'candidate_pool.json',result);return result


def select_quality_pool(root,pool,policy_sha,frame_budget=None):
    """Capacity-aware sampling; validation owns complete original records."""
    from retarget.learning.wuji_ik.real_data import allocate_even,diverse_windows
    rng=np.random.default_rng(42);windows=[]
    for r in pool['records']:
        for side,d in r['sides'].items():
            for w in d['windows']:
                windows.append(dict(w,record_id=r['record_id'],collection_id=r['collection_id'],
                    source_h5=r['source_h5'],source_sha256=r['source_sha256'],source_stat=r['source_stat'],
                    device=r['device'],collection=r['collection'],side=side,calibration=d['calibration'],
                    source_seq_step=d['quality']['nominal_source_seq_step']))
    total=sum(w['end']-w['start'] for w in windows)
    budget=total if frame_budget is None else int(frame_budget)
    if not 0<budget<=total:raise ValueError('Frame budget exceeds certified capacity')
    if budget<total:
        # Preserve all eligible collections; balance device/hand subject to actual
        # certified capacity, then collection quotas and motion/spread cycling.
        strata={}
        for w in windows:strata.setdefault((w['device'],w['side']),[]).append(w)
        quotas=allocate_even({k:sum(w['end']-w['start'] for w in v) for k,v in strata.items()},budget)
        sampled=[]
        for key,ws in sorted(strata.items()):
            groups={}
            for w in ws:groups.setdefault(w['collection_id'],[]).append(w)
            counts=allocate_even({k:sum(w['end']-w['start'] for w in v) for k,v in groups.items()},quotas[key])
            for k,n in counts.items():sampled.extend(diverse_windows(groups[k],n,rng,prefix_only=True))
        windows=sampled
    # Select whole records to match ~5% independently in each device/hand and
    # month/task stratum. Records from a collection cannot all enter validation.
    records=sorted({w['record_id'] for w in windows});strata=sorted({(w['device'],w['side']) for w in windows})
    groups=sorted({(w['device'],w['collection'][:6],base.task_class(w['collection'])) for w in windows})
    ri={k:i for i,k in enumerate(records)};si={k:i for i,k in enumerate(strata)};gi={k:i for i,k in enumerate(groups)}
    counts=np.zeros((len(records),len(strata)));hist=np.zeros((len(records),len(groups)));collections={}
    for w in windows:
        i=ri[w['record_id']];n=w['end']-w['start'];counts[i,si[(w['device'],w['side'])]]+=n
        hist[i,gi[(w['device'],w['collection'][:6],base.task_class(w['collection']))]]+=n
        collections.setdefault(w['collection_id'],set()).add(i)
    totals=counts.sum(0);matrix=np.c_[counts,counts.sum(1),hist]
    weights=np.r_[1/np.maximum(.05*totals,1),.5/(.05*budget),.15/np.sqrt(max(1,len(groups)))/np.maximum(.05*hist.sum(0),1)]
    norm=matrix*weights;target=norm.sum(0)*.05;chosen=np.zeros(len(records),bool);residual=-target
    squared=np.einsum('ij,ij->i',norm,norm);members={i:[] for i in range(len(records))}
    for c,ids in collections.items():
        for i in ids:members[i].append(ids)
    while True:
        delta=2*(norm@residual)+squared;delta[chosen]=np.inf
        for i in np.flatnonzero(~chosen):
            if any(all(chosen[j] or j==i for j in ids) for ids in members[i]):delta[i]=np.inf
        i=int(delta.argmin())
        if not np.isfinite(delta[i]) or delta[i]>=-1e-12:break
        chosen[i]=True;residual+=norm[i]
    if np.any(counts[chosen].sum(0)==0):raise ValueError('Validation lacks a device/hand stratum')
    val={r for r,i in ri.items() if chosen[i]};split_rows={'train':0,'val':0,'test':0}
    for w in windows:
        w['split']='val' if w['record_id'] in val else 'train';split_rows[w['split']]+=w['end']-w['start']
        w.setdefault('motion_bin',[0,0]);w['id']=base.identity({k:w[k] for k in ('record_id','source_sha256','side','start','end','context_start','split')})[:24]
    expected={r['collection_id'] for r in pool['records']};covered={w['collection_id'] for w in windows if w['split']=='train'}
    if covered!=expected:raise ValueError('Missing eligible training collection')
    for r in records:
        for side in ['right','left']:
            intervals=sorted((w['start'],w['end']) for w in windows if w['record_id']==r and w['side']==side)
            if any(b>c for (_,b),(c,_) in zip(intervals,intervals[1:])):raise ValueError('Duplicate label source frames')
    value=dict(complete=True,policy_sha=policy_sha,segments=sorted(windows,key=lambda w:w['id']),frames=budget,
        split_rows=split_rows,collection_coverage=pool['collections'],eligible_collections=len(expected),selected_collections=len(covered),
        reserved_validation_episodes=sorted(val),short_exact_budget_tails=sum(w['end']-w['start']<64 for w in windows),
        selection='Capacity-aware device/hand and collection quotas, randomized motion/spread; whole-record validation approximately5%, stratified device/hand and month/task; observed context retained.',
        validation_strata=[dict(device=k[0],side=k[1],frames=int(totals[i]),validation_frames=int(counts[chosen,i].sum()),fraction=float(counts[chosen,i].sum()/totals[i])) for i,k in enumerate(strata)])
    value['signature']=base.identity(value);base.write(root/'selection.json',value);return value


def verify_screening_reuse(root,pool,conf,policy):
    """Reuse only the completed, frozen screening recipe, never relabel policy."""
    proof=json.loads((root/'screening_reuse.json').read_text())
    if not pool.get('complete') or base.digest(root/'candidate_pool.json')!=proof['candidate_pool_sha256']:
        raise ValueError('Completed screening pool changed')
    old=json.loads((root/proof['previous_policy']).read_text())
    for key in ['emf_confidence_min','ik_confidence_min','skeleton_confidence_min','fk_position_max_m','fk_orientation_max_rad','min_certified_frames','max_frames','history','sdk_version','sdk_binaries']:
        if old[key]!=policy[key]:raise ValueError('Screening semantics changed: '+key)
    for name,sha in old['source_sha256'].items():
        if name!='build_wuji_quality_sdk_dataset.py' and policy['source_sha256'][name]!=sha:raise ValueError('Screening dependency changed: '+name)
    import ast
    old_tree=ast.parse((root/proof['previous_builder']).read_text());new_tree=ast.parse(Path(__file__).read_text())
    for name in ['save_arrays','certify','process_collection','collect']:
        def node(tree):return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        if node(old_tree)!=node(new_tree):raise ValueError('Screening implementation changed: '+name)
    if pool['policy_sha']!=base.identity(old):raise ValueError('Screening policy identity changed')
    # Every certificate array and raw source is independently checked by generation.
    base.write(root/'screening_reuse_verified.json',dict(passed=True,optimizer_updates=0,original_policy_sha=pool['policy_sha'],candidate_pool_sha256=proof['candidate_pool_sha256'],screening_functions_unchanged=True,screening_dependencies_unchanged=True))


def generate_one(task):
    root,w,signature=task;path=root/'sdk_segments'/(w['id']+'.npz');receipt=path.with_suffix('.json')
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['label_signature']==signature and base.digest(path)==old['npz_sha256'];return old
    if base.digest(w['quality_file'])!=w['quality_sha256']:raise ValueError('Quality certificate array changed')
    if w['context_start']!=w['certificate']['context_start'] or w['end']>w['certificate']['end']:raise ValueError('Selected crop would change certified history')
    if base.digest(w['source_h5'])!=w['source_sha256']:raise ValueError('Selected source changed')
    with np.load(w['quality_file']) as f:d={k:f[k] for k in f.files}
    d=retarget_history(root,w,d);save_arrays(path,d)
    result=dict(id=w['id'],frames=len(d['q_sdk_rad']),label_signature=signature,npz_sha256=base.digest(path),urdf_sha256=base.digest(w['calibration']['urdf']),source_sha256=w['source_sha256'],source_window=w,
        protocol='Official SDK reset at attention-context start; complete real EMF warm-up retained without q labels; only qualified contiguous labels; native WujiHand2 history/filter retained')
    base.write(receipt,result);return result


def generate(root,plan,signature,workers):
    (root/'sdk_segments').mkdir(exist_ok=True);done=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
        for result in pool.map(generate_one,[(root,w,signature) for w in plan['segments']]):
            done.append(result);base.write(root/'DATA_STATUS.json',dict(stage='strict_sdk_labels',complete=False,frames=sum(r['frames'] for r in done),target_frames=plan['frames'],segments=len(done)))
            if len(done)%25==0:print('STRICT_LABELLED',len(done),sum(r['frames'] for r in done),flush=True)


def prepare(root,output,plan,signature):
    from retarget.learning.wuji_ik.features import urdf_limits,joint_names
    if output.exists():
        meta=json.loads((output/'manifest.json').read_text())
        if meta['signature']!=signature:raise ValueError('Existing prepared cache is incompatible')
        return meta
    pending=output.with_name(output.name+'.building.'+str(os.getpid()));pending.mkdir(parents=True)
    schema={'features':(np.float32,(5,9)),'q_normalized':(np.float32,(20,)),'frame_id':(np.int64,()),
        'position_tx_m':(np.float32,(5,3)),'orientation_xyzw':(np.float32,(5,4)),'confidence':(np.float32,(5,)),
        'timestamp_ns':(np.int64,()),'source_seq':(np.int64,()),'q_sdk_rad':(np.float64,(20,)),
        'skeleton_sdk_m':(np.float32,(21,3)),'human_q_rad':(np.float64,(21,)),
        'ik_confidence':(np.float64,(5,)),'skeleton_confidence':(np.float64,(21,)),
        'fk_position_error_m':(np.float64,(5,)),'fk_orientation_error_rad':(np.float64,(5,))}
    arrays={name:np.lib.format.open_memmap(pending/(name+'.npy'),mode='w+',dtype=dtype,shape=(plan['frames'],)+shape) for name,(dtype,shape) in schema.items()}
    context_rows=sum(w['start']-w['context_start'] for w in plan['segments'])
    contexts=np.lib.format.open_memmap(pending/'context_features.npy',mode='w+',dtype=np.float32,shape=(context_rows,5,9))
    offset=0;context_offset=0;segments=[]
    for w in plan['segments']:
        path=root/'sdk_segments'/(w['id']+'.npz');receipt=json.loads(path.with_suffix('.json').read_text())
        if receipt['label_signature']!=signature or base.digest(path)!=receipt['npz_sha256']:raise ValueError('SDK segment changed')
        n=w['end']-w['start'];c=w['start']-w['context_start']
        with np.load(path) as d:
            for key,array in arrays.items():array[offset:offset+n]=d[key]
            contexts[context_offset:context_offset+c]=d['context_features']
        segments.append(dict(offset=offset,length=n,context_offset=context_offset,context_length=c,
            source_row_start=w['start'],source_context_start=w['context_start'],record=w['id'],
            side=0 if w['side']=='right' else 1,split=w['split'],dataset=w['device'],group_id=w['record_id'],source_id=w['source_h5'],
            collection=w['collection'],calibration_urdf=w['calibration']['urdf'],urdf_sha256=receipt['urdf_sha256'],
            sdk_reset=True,sdk_reset_at='context_start',quality_labels_only=True,motion_bin=w['motion_bin'],activity=w['activity'],spread=w['spread']))
        offset+=n;context_offset+=c
    assert offset==plan['frames'] and context_offset==context_rows
    for a in arrays.values():a.flush()
    contexts.flush();arrays.clear();del contexts
    base.write(pending/'segments.json',segments)
    names=[key+'.npy' for key in schema]+['context_features.npy','segments.json']
    meta=dict(schema='wuji_ik_segments_v1',complete=True,signature=signature,label_signature=signature,rows=plan['frames'],segments=len(segments),
        context_rows=context_rows,supervision_policy='Only hard-quality labels; additional real EMF context has no q target or loss; SDK and model receive identical past EMF',
        split_rows=plan['split_rows'],validation_fraction=plan['split_rows']['val']/plan['frames'],
        limits=np.stack([urdf_limits(root/'assets'/f'robot_{s}.urdf',s) for s in ['right','left']]).tolist(),
        joint_names={s:joint_names(s) for s in ['right','left']},sides=['right','left'],
        label_source='Official SDK2026.8.31 sequential WujiHand2; hard confidence/FK qualified labels with complete observed EMF history',
        eligible_collections=plan['eligible_collections'],source_signature=plan['signature'],source_metadata=str(root/'selection.json'),
        array_sha256={name:base.digest(pending/name) for name in names})
    base.write(pending/'manifest.json',meta)
    for path in pending.iterdir():path.chmod(0o444)
    pending.rename(output);return meta


def validate_quality(root,output,plan,signature):
    meta=json.loads((output/'manifest.json').read_text())
    for name,sha in meta['array_sha256'].items():
        if base.digest(output/name)!=sha:raise ValueError('Prepared array changed: '+name)
    groups={s:{w['record_id'] for w in plan['segments'] if w['split']==s} for s in ['train','val']}
    if groups['train']&groups['val']:raise ValueError('Original episode split leakage')
    extrema={k:[] for k in QUALITY_KEYS};n=0;roundtrip=0.;limits=np.asarray(meta['limits'])
    for w in plan['segments']:
        with np.load(root/'sdk_segments'/(w['id']+'.npz')) as a:
            d={k:a[k] for k in a.files}
        if not valid(d).all():raise ValueError('Final SDK trajectory failed quality')
        if len(d['features'])+len(d['context_features'])>1024:raise ValueError('Context exceeds original continuous segment')
        if not np.isfinite(d['features']).all() or not np.isfinite(d['context_features']).all():raise ValueError('Invalid input features')
        if not np.all(np.diff(d['timestamp_ns'])>0) or not np.all(np.diff(d['timestamp_ns'])<=50_000_000):raise ValueError('Timestamp gap')
        if not np.all(np.diff(d['source_seq'])==w['source_seq_step']):raise ValueError('Missing source sequence')
        side=0 if w['side']=='right' else 1;lo,hi=limits[side,:,0],limits[side,:,1]
        reconstructed=(d['q_normalized']+1)*.5*(hi-lo)+lo
        roundtrip=max(roundtrip,float(abs(reconstructed-d['q_sdk_rad']).max()))
        if not np.isfinite(d['q_normalized']).all() or abs(d['q_normalized']).max()>1.00001:raise ValueError('Invalid normalized q')
        for k in extrema:extrema[k].append((float(d[k].min()),float(d[k].max())))
        n+=len(d['q_sdk_rad'])
    repeats=[]
    for device in sorted({w['device'] for w in plan['segments']}):
        for side in ['right','left']:
            w=max((w for w in plan['segments'] if w['device']==device and w['side']==side),key=lambda w:w['end']-w['start'])
            full=dict(w,start=w['context_start']);fresh_all=solve(root,full,retarget=True)
            start=w['start']-w['context_start'];fresh={k:v[start:] for k,v in fresh_all.items()}
            with np.load(root/'sdk_segments'/(w['id']+'.npz')) as saved:
                np.testing.assert_array_equal(fresh['q_sdk_rad'],saved['q_sdk_rad'])
                np.testing.assert_array_equal(fresh['skeleton_sdk_m'],saved['skeleton_sdk_m'])
                for key in QUALITY_KEYS:np.testing.assert_allclose(fresh[key],saved[key],rtol=0,atol=1e-10)
                np.testing.assert_array_equal(fresh_all['features'][:start],saved['context_features'])
            assert valid(fresh).all();repeats.append(dict(device=device,side=side,frames=len(fresh['q_sdk_rad']),max_q_difference_rad=0.))
    assert n==plan['frames'] and roundtrip<5e-7
    report=dict(complete=True,passed=True,frames=n,split_rows=plan['split_rows'],
        eligible_collections=plan['eligible_collections'],selected_collections=plan['selected_collections'],
        no_episode_split_overlap=True,no_source_frame_overlap=True,normalization_roundtrip_max_rad=roundtrip,
        signature=signature,optimizer_updates=0,context_rows=meta['context_rows'],
        context_policy='Real EMF warm-up is input-only; bad SDK q is never a label; no hidden SDK prehistory',
        quality_gates_passed=True,quality_frames_checked=n,quality_failures=0,
        quality_extrema={k:dict(min=min(v[0] for v in a),max=max(v[1] for v in a)) for k,a in extrema.items()},
        strict_full_pipeline_replays=repeats)
    base.write(root/'data_validation.json',report);return report


def coverage_report(root,plan,pool):
    rows=[]
    for c in pool['collections']:
        selected=[w for w in plan['segments'] if w['device']==c['device'] and w['collection']==c['collection']]
        row={k:c[k] for k in ['device','collection','status','inspected','available_source_files','full_inventory_exhausted','candidate_frames']}
        row.update(train_frames=sum(w['end']-w['start'] for w in selected if w['split']=='train'),val_frames=sum(w['end']-w['start'] for w in selected if w['split']=='val'))
        rows.append(row)
    with (root/'collection_coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(frames=plan['frames'],split_rows=plan['split_rows'],segments=len(plan['segments']),source_collections=len(rows),quality_covered_collections=sum(r['train_frames']>0 for r in rows),
        rejected_collections=[r for r in rows if r['train_frames']==0],
        strata_frames={device+'/'+side+'/'+split:sum(w['end']-w['start'] for w in plan['segments'] if (w['device'],w['side'],w['split'])==(device,side,split)) for device in sorted({w['device'] for w in plan['segments']}) for side in ['right','left'] for split in ['train','val']},
        motion_tertile_frames=np.bincount([w['motion_bin'][0] for w in plan['segments']],weights=[w['end']-w['start'] for w in plan['segments']],minlength=3).astype(int).tolist(),
        source_records={split:len({w['record_id'] for w in plan['segments'] if w['split']==split}) for split in ['train','val']},
        short_exact_budget_tails=plan['short_exact_budget_tails'])
    base.write(root/'dataset_distribution.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();root=a.campaign
    sys.path[:0]=[str(a.reference),str(root/'scripts')];conf=json.loads((root/'MANIFEST.json').read_text())
    for name in ['quality_segments','collection_receipts','input_audits']:(root/name).mkdir(exist_ok=True)
    import wuji_sdk as sdk
    import wuji_quality_teacher as teacher
    from retarget.learning.wuji_ik import quality_data,real_data,features
    policy=dict(emf_confidence_min=.8,ik_confidence_min=.8,skeleton_confidence_min=.8,fk_position_max_m=.01,fk_orientation_max_rad=.7,
        min_certified_frames=64,max_frames=1024,seed=42,frames=conf['frames'],val_fraction=.05,
        history='SDK reset at complete raw EMF context start; retain real past EMF as unsupervised context; only certified continuous runs own q loss; full pipeline replay required',
        selection='Capacity-aware device/hand sampling; original-record isolation; stratify validation by device/hand and month/task; external five episodes excluded',
        source_sha256={str(Path(path).name):base.digest(path) for path in [__file__,base.__file__,teacher.__file__,quality_data.__file__,real_data.__file__,features.__file__]},
        sdk_version=sdk.__version__,sdk_binaries={p.name:base.digest(p) for p in Path(sdk.__file__).parent.glob('*.so')})
    policy_sha=base.identity(policy);base.write(root/'data_policy.json',policy)
    pool_path=root/'candidate_pool.json';pool=json.loads(pool_path.read_text()) if pool_path.exists() else None
    if pool is None:pool=collect(root,conf,policy_sha,a.workers)
    if pool['policy_sha']!=policy_sha:verify_screening_reuse(root,pool,conf,policy)
    plan=select_quality_pool(root,pool,policy_sha,conf['frames'])
    plan['selection']+=' STRICT QUALITY: all supervised frames EMF/IK/skeleton >=0.8; FK position<=10mm and angle<=0.7rad. All real past EMF retained as input-only context; never supervise unqualified SDK q.'
    plan['signature']=base.identity({k:v for k,v in plan.items() if k!='signature'});base.write(root/'selection.json',plan)
    assets={p.name:base.digest(p) for p in (root/'assets').glob('*.urdf')};signature=base.identity(dict(policy=policy_sha,selection=plan['signature'],assets=assets))
    base.write(root/'sdk_provenance.json',dict(sdk_version=sdk.__version__,sdk_file=sdk.__file__,binaries=policy['sdk_binaries'],label_signature=signature))
    coverage_report(root,plan,pool);generate(root,plan,signature,a.workers)
    output=Path(conf['dataset']);prepare(root,output,plan,signature)
    report=validate_quality(root,output,plan,signature)
    base.write(root/'DATA_STATUS.json',dict(stage='complete',**report));print('STRICT_DATASET_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
