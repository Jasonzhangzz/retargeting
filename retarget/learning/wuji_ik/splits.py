"""Stratified group-held-out validation, preserving the original test set."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .data import sha256, write_json


def stratified_validation(source, output, fraction=.05, seed=42):
    if not 0 < fraction < .5: raise ValueError('Invalid validation fraction')
    source, output = Path(source), Path(output)
    parent = json.loads((source/'manifest.json').read_text())
    if not parent.get('complete'): raise ValueError('Incomplete input cache')
    segments = json.loads((source/'segments.json').read_text())
    grouped = {}
    for s in segments: grouped.setdefault(s['group_id'], []).append(s)
    for key, entries in grouped.items():
        if any(s['split']=='test' for s in entries) and any(s['split']!='test' for s in entries):
            raise ValueError('Original test group overlaps train/validation: '+key)
    candidates = sorted(k for k, v in grouped.items() if not any(s['split']=='test' for s in v))
    if len(candidates) < 2: raise ValueError('Not enough independent groups')
    strata = sorted({(s['dataset'], s['side']) for k in candidates for s in grouped[k]})
    lookup = {s:i for i,s in enumerate(strata)}
    counts = np.zeros((len(candidates),len(strata)),np.float64)
    histograms = np.zeros((len(candidates),160),np.float64)
    labels = np.load(source/'q_normalized.npy',mmap_mode='r')
    for i, key in enumerate(candidates):
        for s in grouped[key]:
            counts[i,lookup[(s['dataset'],s['side'])]] += s['length']
            q = labels[s['offset']:s['offset']+s['length']]
            bins = np.clip(np.floor((q+1)*4).astype(np.int64),0,7) + np.arange(20)*8
            histograms[i] += np.bincount(bins.ravel(),minlength=160)
    totals = counts.sum(0); rows = float(totals.sum())
    matrix = np.c_[counts,counts.sum(1),histograms]
    target = matrix.sum(0)*fraction
    # Primary: hold out ~5% within every dataset/side. Secondary: overall count
    # and the distribution of all 20 normalized joint angles (eight bins each).
    weights = np.r_[1/np.maximum(totals*fraction,1)/np.sqrt(len(strata)),
                    .5/max(rows*fraction,1),np.full(160,.2/np.sqrt(20)/max(rows*fraction,1))]
    normalized = matrix*weights; desired = target*weights
    # Seeded deterministic tie-breaking, independent of filesystem iteration.
    ordering = sorted(range(len(candidates)),key=lambda i:hashlib.sha256(f'{seed}:{candidates[i]}'.encode()).hexdigest())
    normalized = normalized[ordering]; ordered_keys = [candidates[i] for i in ordering]
    squared = np.einsum('ij,ij->i',normalized,normalized)
    chosen = np.zeros(len(candidates),bool); residual = -desired.copy()
    while True:
        delta = 2*(normalized@residual)+squared; delta[chosen] = np.inf
        best = int(delta.argmin())
        if not np.isfinite(delta[best]) or delta[best] >= -1e-12: break
        chosen[best] = True; residual += normalized[best]
    # Whole-group swaps improve fractional balance without fragmenting trajectories.
    for _ in range(32):
        selected = np.flatnonzero(chosen)
        if not len(selected): break
        add = 2*(normalized@residual)+squared
        drop = -2*(normalized[selected]@residual)+squared[selected]
        delta = drop[:,None]+add[None,:]-2*(normalized[selected]@normalized.T)
        delta[:,chosen] = np.inf
        a,b = np.unravel_index(delta.argmin(),delta.shape)
        if delta[a,b] >= -1e-12: break
        old = selected[a]; chosen[old] = False; chosen[b] = True
        residual += normalized[b]-normalized[old]
    validation = {k for k, keep in zip(ordered_keys,chosen) if keep}
    if not validation or len(validation)==len(candidates): raise ValueError('Failed to select independent validation groups')
    achieved = np.zeros(len(strata),np.int64); split_rows = {'train':0,'val':0,'test':0}
    result_segments = []
    for original in segments:
        s = dict(original)
        if s['split'] != 'test': s['split'] = 'val' if s['group_id'] in validation else 'train'
        result_segments.append(s); split_rows[s['split']] += s['length']
        if s['split']=='val': achieved[lookup[(s['dataset'],s['side'])]] += s['length']
    if np.any(achieved==0): raise ValueError('A dataset/side stratum is absent from validation')
    original_indices = {k:i for i,k in enumerate(candidates)}
    selected_hist = sum((histograms[original_indices[k]] for k in validation),np.zeros(160)).reshape(20,8)
    pool_hist = histograms.sum(0).reshape(20,8)
    val_rows = int(achieved.sum())
    pose_tv = .5*np.abs(selected_hist/max(val_rows,1)-pool_hist/rows).sum(1)
    report = dict(parent_signature=parent['signature'], fraction=fraction, seed=seed,
        policy='group-held-out; pooled original train+val; original test unchanged; stratify dataset x side plus q histogram',
        pool_rows=int(rows), validation_rows=val_rows, actual_fraction=val_rows/rows,
        strata=[dict(dataset=k[0],side=int(k[1]),pool_rows=int(totals[i]),validation_rows=int(achieved[i]),fraction=float(achieved[i]/totals[i])) for i,k in enumerate(strata)],
        validation_groups=sorted(validation), per_joint_histogram_total_variation=pose_tv.tolist(), split_rows=split_rows)
    signature = hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest()
    if output.exists():
        old=json.loads((output/'manifest.json').read_text())
        if old.get('signature')!=signature: raise ValueError('Existing split cache is incompatible')
        return json.loads((output/'split_report.json').read_text())
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=Path(tempfile.mkdtemp(prefix=output.name+'.building_',dir=output.parent))
    try:
        for name in ('features.npy','q_normalized.npy','frame_id.npy'):
            os.symlink((source/name).resolve(),pending/name)
        write_json(pending/'segments.json',result_segments);write_json(pending/'split_report.json',report)
        metadata=dict(parent);metadata.update(signature=signature,split_rows=split_rows,split_policy=report['policy'],
            parent_signature=parent['signature'],validation_fraction=fraction)
        metadata['array_sha256']=dict(parent['array_sha256']);metadata['array_sha256']['segments.json']=sha256(pending/'segments.json')
        write_json(pending/'manifest.json',metadata);os.replace(pending,output)
    except BaseException:
        shutil.rmtree(pending,ignore_errors=True);raise
    return report
