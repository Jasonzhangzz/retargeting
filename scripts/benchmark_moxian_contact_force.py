"""Frozen-source force-direction ablation; never uses gesture labels in CEM."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.card_math import CardSettings
from moxian_web.card_geometry import closest_surface_points
from moxian_web.wuji_retarget import TactileRetarget


VARIANTS={
    'reach_no_direction':dict(direction_price=0.),
    'card_alignment':dict(),
    'cone_mu1':dict(contact_direction_model='coulomb_force',friction_coefficient=1.),
    'cone_normal_mu1':dict(contact_direction_model='coulomb_force',friction_coefficient=1.,normal_force_weight=1.),
    'cone_mu05':dict(contact_direction_model='coulomb_force',friction_coefficient=.5),
    'cone_normal_mu05':dict(contact_direction_model='coulomb_force',friction_coefficient=.5,normal_force_weight=1.),
}
FIELDS=('vertices_m','joints_m','timestamp_ms','tactile_timestamp_ms','source_contact','geometry_version','binding_version','hand_side','quality','sensor_mounting','calibrated','skeleton_runtime_revision','pose_calibration_version','contact_registration')


def compact(rows):
    by_stamp={}
    for row in rows:
        h=row.get('hand_model',row)
        if h.get('vertices_m'):by_stamp[h['timestamp_ms']]={k:h.get(k) for k in FIELDS}
    return [by_stamp[k] for k in sorted(by_stamp)]


def pad_gap(solver,result,regions=(3,6)):
    triangles=[]
    for region in regions:
        owner=solver.region_owners[region]
        t=solver.tensor(result['link_positions_m'][owner]);r=solver.tensor(result['link_rotations'][owner])
        triangles.append(solver.pad_triangles[region]@r.T+t)
    distances=[]
    for a,b in (triangles,triangles[::-1]):
        points=torch.cat((a.reshape(-1,3),a.mean(1)),0)
        near,_,_=closest_surface_points(points,b)
        distances.append(float((points-near).norm(dim=-1).min()*1000))
    return min(distances)


def metrics(solver,result,dataset):
    anchors=result['contact_anchors'];errors=np.array([a['error_mm'] for a in anchors]);angle=np.array([a['normal_error_deg'] for a in anchors])
    # A projected latent force is always in-cone by construction. Score the
    # ORIGINAL demonstrated direction, and count success only at contact reach.
    reached=errors<=2.;cone45=angle<=45.;force_errors=[];normal=[]
    for a in anchors:
        fit=a['force_support'];estimate=a.get('force_estimate') or {}
        if fit.get('available') and estimate.get('magnitude_relative',0)>0:
            m=estimate['magnitude_relative'];force_errors.append(np.sqrt(fit['vector_error_squared'])/m)
            normal.append(fit['desired_normal_relative']/m)
    by_region={a['region']:a for a in anchors}
    pair_gap=None
    if dataset=='pinch' and 3 in by_region and 6 in by_region:
        pair_gap=float(np.linalg.norm(np.asarray(by_region[3]['robot_position_m'])-by_region[6]['robot_position_m'])*1000)
    return dict(anchors=len(anchors),contact_rate=float(reached.mean()),direction_in_45deg_rate=float(cone45.mean()),
        contact_and_in_cone_rate=float((reached&cone45).mean()),rms_mm=float(np.sqrt(np.mean(errors**2))),
        max_error_mm=float(errors.max()),mean_angle_deg=float(angle.mean()),
        mean_normal_support_fraction=float(np.mean(normal)) if normal else None,
        mean_force_vector_error_fraction=float(np.mean(force_errors)) if force_errors else None,
        self_penetration_mm=result['collision']['self_mesh']['max_penetration_mm'],
        feasible=result['solution_feasible'],elapsed_ms=result['quality']['processing_ms'],
        selected_cop_pair_gap_mm=pair_gap,
        sampled_pad_surface_gap_mm=pad_gap(solver,result) if dataset=='pinch' else None,
        object_contact_ground_truth_available=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pinch',type=Path,required=True);ap.add_argument('--pinch-topology',type=Path,required=True)
    ap.add_argument('--grasp',type=Path,required=True);ap.add_argument('--grasp-topology',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[7,19,37])
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    sources={name:(compact(json.loads(path.read_text())),json.loads(top.read_text())) for name,path,top in (
        ('pinch',args.pinch,args.pinch_topology),('bottle_cap',args.grasp,args.grasp_topology))}
    manifest=dict(variants=VARIANTS,seeds=args.seeds,solver_budget=CardSettings().payload(),
        split='chronological_first_half_normalization_only_second_half_evaluation',
        source_shape_and_contact_targets_frozen=True,source_poses_optimized=False,
        gesture_labels_used_by_solver=False,explicit_self_contact=False,
        normal_target_assumption='desired_resultant_magnitude_not_independent_force_measurement',
        force_units='relative_response_sum',force_calibrated=False,shear_measured=False,
        friction_coefficient_is_measured=False,real_object_geometry_available=False,datasets={})
    for name,(frames,top) in sources.items():
        if len(frames)<4:raise ValueError('At least four distinct captured frames per dataset required')
        payload=dict(topology=top,frames=frames)
        (out/(name+'-source.json')).write_text(json.dumps(payload))
        manifest['datasets'][name]=dict(frames=len(frames),normalization_frames=len(frames)//2,evaluation_frames=len(frames)-len(frames)//2,
            input_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[];began=time.monotonic()
    with (out/'results.ndjson').open('w') as stream:
        for name,(frames,top) in sources.items():
            solver=TactileRetarget(top,out/'mappings');half=len(frames)//2
            force_max=max((a['force'] for f in frames[:half] for a in solver.extractor.extract(f)[2]),default=1.)
            for variant,kwargs in VARIANTS.items():
                for seed in args.seeds:
                    solver.settings=dataclasses.replace(CardSettings(),**kwargs,seed=seed)
                    solver.topology_version=solver.topology()['topology_version']
                    for index,frame in enumerate(frames[half:],half):
                        solver.reset();solver.offline_force_max=force_max
                        _,_,source=solver.extractor.extract(frame)
                        result=solver.step(frame)
                        assert not result['self_contact_pairs'] and not result['solver']['tactile_seed']['applied']
                        assert not result['solver']['used_history_fallback']
                        assert result['initial_qpos']==result['robokit_qpos']
                        assert all(a['position_m']==b['position_m'] and a['direction']==b['direction'] for a,b in zip(source,result['contact_anchors']))
                        record=dict(dataset=name,variant=variant,seed=seed,index=index,timestamp_ms=frame['timestamp_ms'],
                            metrics=metrics(solver,result,name),qpos=result['qpos'],contacts=result['contact_anchors'],
                            solver=result['solver'],source_targets_preserved=True)
                        records.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                    print(json.dumps(dict(dataset=name,variant=variant,seed=seed,completed=len(records),elapsed_seconds=time.monotonic()-began)),flush=True)
    summary={}
    for name in sources:
        summary[name]={}
        for variant in VARIANTS:
            rows=[r['metrics'] for r in records if r['dataset']==name and r['variant']==variant];entry={}
            for key in rows[0]:
                values=[r[key] for r in rows if r[key] is not None]
                entry[key]=dict(mean=float(np.mean(values)),min=float(np.min(values)),max=float(np.max(values))) if values else None
            summary[name][variant]=entry
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(completed=len(records),seconds=time.monotonic()-began,output=str(out))),flush=True)


if __name__=='__main__':main()
