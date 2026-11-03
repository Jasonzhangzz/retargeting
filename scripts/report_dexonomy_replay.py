#!/usr/bin/env python3
"""Export per-case closure labels and replay diagnostics from grasp-type caches."""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);a=p.parse_args();rows=[]
for manifest in sorted(a.batch.glob('*/manifest.json')):
 root=manifest.parent;cases=json.loads(manifest.read_text())['cases'];metrics=json.loads(gzip.decompress((root/'metrics.json.gz').read_bytes()))
 with np.load(root/'poses.npz',allow_pickle=False) as arrays:
  for i,(case,m) in enumerate(zip(cases,metrics)):
   c=m['executed_closure'];g=m['geometry_diagnostics'];d=m['dynamic'];source=Path(case['object']['source_record'])
   reason='passed' if c['certified'] else c.get('reason','wrench_certificate_failed')
   rows.append({'grasp_type':case['grasp_type'],'case':case['name'],'object_id':source.parent.parent.name,
     'scene':source.stem,'source_row':case['object']['source_row'],'scene_scale':m['scene_scale'],
     'scale_x':float(arrays['object_scale'][i,0]),'scale_y':float(arrays['object_scale'][i,1]),'scale_z':float(arrays['object_scale'][i,2]),
     'closure_passed':c['certified'],'failure_reason':reason,'contacts':c.get('contacts',0),
     'certificate_margin':c.get('certified_margin'),'closure_cost':c.get('cost'),
     'grasp_target_closure':m['stage_closure']['grasp']['certified'],'squeeze_target_closure':m['stage_closure']['squeeze']['certified'],
     'object_penetration_m':g['max_hand_object_penetration_m'],'self_penetration_m':g['max_hand_hand_penetration_m'],
     'joint_limit_violation_rad':g['joint_limit_violation_rad'],'translation_m':d.get('translation_m'),'rotation_rad':d.get('rotation_rad'),
     'hold_within_threshold':d.get('hold_within_threshold'),'contact_data':str(Path(root.name)/case['contact_data'])})
with (a.batch/'cases.csv').open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
failed=[r for r in rows if not r['closure_passed']]
reasons={reason:sum(r['failure_reason']==reason for r in failed) for reason in sorted({r['failure_reason'] for r in failed})}
report={'cases':len(rows),'grasp_types':len({r['grasp_type'] for r in rows}),'unique_objects':len({r['object_id'] for r in rows}),
 'closure_passed':sum(r['closure_passed'] for r in rows),'closure_failed':len(failed),'failure_reasons':reasons,
 'contacts_min_median_max':[int(min(r['contacts'] for r in rows)),float(np.median([r['contacts'] for r in rows])),int(max(r['contacts'] for r in rows))],
 'object_penetration_over_2mm':sum(r['object_penetration_m']>.002 for r in rows),'self_penetration_over_0_2mm':sum(r['self_penetration_m']>.0002 for r in rows),
 'joint_limit_violation_over_1e_3rad':sum(r['joint_limit_violation_rad']>.001 for r in rows),'hold_within_threshold':sum(r['hold_within_threshold'] for r in rows),
 'failed_cases':failed,'diagnostic_thresholds_are_not_the_closure_gate':True}
(a.batch/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='failed_cases'},indent=2))
