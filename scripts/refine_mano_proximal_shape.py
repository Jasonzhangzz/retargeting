#!/usr/bin/env python3
"""Finite static breadth refinement using saved together data; no pose solver.

Writes a separate candidate and evidence, never replaces the applied profile.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from moxian_web.mano_runtime import ManoSurface
from moxian_web.hand_model import DEFAULT_MANO
from moxian_web.proximal_shape import proximal_pair_gaps,refine_proximal_widths
from scripts.audit_moxian_calibration import restore
from scripts.calibrate_wuji_mano import CHAINS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--home',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    profile=json.loads((args.home/'current.json').read_text());capture=json.loads((args.home/'capture.json').read_text());rows=[r for r in capture['rows'] if r['stage']=='together']
    if len(rows)<6:raise ValueError('缺少独立的并拢标定数据')
    m=ManoSurface(DEFAULT_MANO,'cpu');restore(m,profile);m.enable_official_fk()
    weights=m.model.weights.numpy();base=m.rest_np.copy();jbase=m.rest_j_np.copy();poses,roots,offsets,centres=m.official_parameters(rows)
    train=np.unique(np.linspace(0,len(rows)//2-1,3).astype(int));valid=np.unique(np.linspace(len(rows)//2,len(rows)-1,3).astype(int));evaluations=0
    def gaps(x,indices):
        m.proximal_lateral_scales=np.array(x);m.set_shape(profile['shape'])
        with torch.no_grad():v,j,_,_=m.decode(poses[indices],roots[indices],offsets[indices],centres[indices])
        return np.array([[p['gap_m'] for p in proximal_pair_gaps(a,b,m.faces,weights,CHAINS)] for a,b in zip(v.numpy(),j.numpy())])
    original=np.array(profile.get('proximal_lateral_scales',[0]*4));before=gaps(original,valid)
    fitted=refine_proximal_widths(m,[rows[i] for i in train],max_nfev=40)
    solved=m.proximal_lateral_scales.copy();evaluations=fitted.get('evaluations',0)
    after=gaps(solved,valid)
    accepted=bool(np.mean(abs(after))<np.mean(abs(before))*.8 and after.min()>-.005 and after.max()<.0045 and np.linalg.norm(m.rest_np-base,axis=1).max()<.0035)
    candidate=copy.deepcopy(profile);candidate['proximal_lateral_scales']=solved.tolist()
    report={'accepted':accepted,'optimizer':'shared proximal_shape.refine_proximal_widths; TRF static breadth, pinky fixed','evaluations':evaluations,'max_nfev':40,'train_frames':train.tolist(),'held_out_frames':valid.tolist(),'source_profile_sha256':hashlib.sha256((args.home/'current.json').read_bytes()).hexdigest(),'capture_sha256':hashlib.sha256((args.home/'capture.json').read_bytes()).hexdigest(),'before_gaps_mm':(before*1000).tolist(),'after_gaps_mm':(after*1000).tolist(),'scales':np.exp(solved).tolist(),'rest_joint_change_m':float(abs(m.rest_j_np-jbase).max()),'max_vertex_change_mm':float(np.linalg.norm(m.rest_np-base,axis=1).max()*1000),'pose_optimized':False,'contact_pose_feedback':False,'applied':False,'geometry_version':m.shape_version}
    candidate['proximal_width_report']=report
    (args.output/'proximal_candidate.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2));(args.output/'proximal_report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(args.output/'proximal_geometry.npz',rest_vertices=m.rest_np,rest_joints=m.rest_j_np,weights=weights,parents=m.parents,faces=m.faces,tip_anchors=m.tip_anchors,dimension_parameters=m.dimension_parameters,proximal_lateral_scales=m.proximal_lateral_scales)
    print(json.dumps(report))
if __name__=='__main__':main()
