"""Compare frozen tactile frames on the reference and Robokit CARD backends."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retarget.baseline.card.math import CardSettings
from retarget.baseline.card.retarget import TactileRetarget


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,nargs='+',required=True,help='Frozen topology/frames bundles; names are evaluation labels only')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[7,19])
    ap.add_argument('--frames',type=int,default=4)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    rows=[];manifest={'budget':[1024,16],'native_self_diagnostic_limit_mm':.5,'objective_profile':'paper','frame_labels_used_by_solver':False,'sources':[]}
    with (args.output/'results.ndjson').open('w') as stream:
        for path in args.source:
            raw=path.read_bytes();source=json.loads(raw);frames=source['frames'];half=len(frames)//2
            indices=np.unique(np.linspace(half,len(frames)-1,min(args.frames,len(frames)-half),dtype=int)).tolist()
            manifest['sources'].append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'evaluation_indices':indices})
            for backend in ['torch','robokit']:
                for seed in args.seeds:
                    settings=replace(CardSettings(),cem_backend=backend,seed=seed)
                    solver=TactileRetarget(source['topology'],args.output/'mappings',settings=settings)
                    # Warm each backend outside measured runs, including graph capture.
                    solver.step(frames[indices[0]])
                    for i in indices:
                        solver.reset();f=frames[i]
                        expected=solver.extractor.extract(f)[2]
                        began=time.perf_counter();r=solver.step(f);wall=(time.perf_counter()-began)*1000
                        assert not r['self_contact_pairs'] and not r['solver']['used_history_fallback']
                        assert r['solver']['sampled_candidates']==16384
                        assert all(a['position_m']==b['position_m'] and a['direction']==b['direction'] for a,b in zip(expected,r['contact_anchors']))
                        row=dict(source=path.stem,backend=backend,seed=seed,index=i,wall_ms=wall,
                            timing_ms=r['solver']['timing_ms'],backend_timing_ms=r['solver'].get('backend_timing_ms'),
                            feasible=r['solution_feasible'],contact_rmse_mm=r['quality']['contact_rmse_mm'],
                            direction_deg=r['quality']['contact_normal_mean_deg'],self_penetration_mm=r['collision']['self_mesh']['max_penetration_mm'],
                            anchors=r['quality']['active_anchors'],qpos=r['qpos'],source_targets_preserved=True)
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    print(json.dumps(dict(source=path.stem,backend=backend,seed=seed,completed=len(rows))),flush=True)
    summary={}
    for source in sorted({r['source'] for r in rows}):
        summary[source]={}
        for backend in ['torch','robokit']:
            selected=[r for r in rows if r['source']==source and r['backend']==backend]
            entry={'frames':len(selected),'feasible':sum(r['feasible'] for r in selected)}
            for name,values in {
                'cem_ms':[r['timing_ms']['cem'] for r in selected],
                'total_ms':[r['timing_ms']['total'] for r in selected],
                'contact_rmse_mm':[r['contact_rmse_mm'] for r in selected if r['contact_rmse_mm'] is not None],
                'direction_deg':[r['direction_deg'] for r in selected if r['direction_deg'] is not None],
                'self_penetration_mm':[r['self_penetration_mm'] for r in selected]}.items():
                entry[name]={'mean':float(np.mean(values)),'p50':float(np.median(values)),'p95':float(np.percentile(values,95)),'max':float(np.max(values))} if values else None
            summary[source][backend]=entry
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
