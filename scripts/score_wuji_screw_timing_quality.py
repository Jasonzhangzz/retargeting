"""Score outputs of complete-trajectory timing against common original F(q)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.score_wuji_pose_comparison import load_reference_modules
from retarget.learning.wuji_ik.batched_lm import FastFingerURDF, RobotRetargetObjective, prepare_robot_targets


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--timings',type=Path,required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args();r=a.root.resolve();w=a.timings
    Human,params,Robot,pre=load_reference_modules(r/'reference');torch.set_num_threads(2);torch.set_grad_enabled(False)
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)
    h=Human('right',r/'assets/builtin_right.urdf',sdk_tx_rounding=True);rob=Robot('WujiHand2','right')
    sfk=FastFingerURDF(r/'assets/builtin_right.urdf',h.joint_names,h.skeleton_links);rfk=FastFingerURDF(rob.urdf_path,rob.joint_names,rob.landmark_names)
    seed=np.concatenate(params('right')['seed']);neutral=h.forward(seed)['skeleton'][:,:3,3].astype(np.float32).astype(float)
    alignment=tensor(pre.estimate_frame(neutral)@pre.OPERATOR2MANO_RIGHT@np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]).T)
    sdk=np.load(r/'sdk/builtin/trajectory_0_right.npz');target=prepare_robot_targets(sfk.forward(tensor(sdk['human_q']))[0],alignment);spatial=RobotRetargetObjective(rfk,*target)
    old=json.loads((r/'viser_viewer/objective_screw_right/metrics.json').read_text())['configurations']['builtin']['rows']
    indices=dict(sdk=0,network=1,causal_prefix=2,frozen_parallel=3,frozen_history_previous=4,frozen_history_network=5,joint_lm=6)
    rows=[]
    for name in json.loads((w/'MANIFEST.json').read_text())['methods']:
        if a.partial and not (w/name/'output.npz').exists():continue
        d=np.load(w/name/'output.npz');qf=d['q_filtered']
        if name=='sdk':q=np.load(r/'oneway_comparison/source_slsqp_stopping/builtin_right.npz')['q_source_raw'][:1133]
        else:q=d['q_raw']
        prev=np.concatenate((rob.bounds.mean(1)[None],q[:-1]))
        parts=spatial(tensor(q))[0];cost=.5*parts.square().sum(1).numpy()+.005*np.square(q-prev).sum(1)
        original=old[indices[name]]['raw_total']
        err=qf-sdk['q_sdk_filtered'];row=dict(method=name,common_F_raw_total=float(cost.sum()),cached_common_F_raw_total=original,
            F_change_percent=float(100*(cost.sum()/original-1)),q_sdk_rmse_rad=float(np.sqrt(np.square(err).mean())),q_sdk_max_rad=float(abs(err).max()))
        rows.append(row)
    report=dict(complete=len(rows)==7,rows=rows,frames=1133,side='right',configuration='builtin',target_scope='same SDK-human targets as original cost comparison',
        note='Single-trajectory BF16 network inference can differ numerically from original five-trajectory batch; fresh network inference is included in timing and propagated through every solver. Do not claim bitwise fixed-quality speedup where output drift exceeds 1e-4 rad.')
    (w/'quality.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
