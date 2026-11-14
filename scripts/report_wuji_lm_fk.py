#!/usr/bin/env python3
"""Post-process frozen results; these CPU FK metrics are outside solver timing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from select_wuji_clean_trajectories import receiver_fk_batch,quaternion_matrix
from benchmark_wuji_parallel_lm import reference_math


def robot_fk_batch(model,q):
    q=np.asarray(q,dtype=np.float64)
    n=len(q);eye=np.eye(3);rotations={model.root:np.broadcast_to(eye,(n,3,3))};positions={model.root:np.zeros((n,3))}
    for joint in model._ordered:
        parent=rotations[joint['parent']]
        position=positions[joint['parent']]+(parent@joint['xyz'][:,None]).squeeze(-1)
        rotation=parent@joint['rotation'];index=joint['q_index']
        if index is not None:
            x,y,z=joint['axis'];k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            angle=q[:,index,None,None];rotation=rotation@(eye+np.sin(angle)*k+(1-np.cos(angle))*(k@k))
        positions[joint['child']]=position;rotations[joint['child']]=rotation
    return np.stack([positions[name] for name in model.landmark_names],1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    sys.path.insert(0,str(root/'reference'))
    from human_ik_reference.kinematics import HumanHandKinematics
    rk,_,_=reference_math(root/'reference');selection=json.loads((root/'selection.json').read_text())['trajectories']
    prediction_subdir=json.loads((root/'metrics.json').read_text())['prediction_subdir'];rows=[];source_order={};startup=[]
    for config in ('builtin','hl_urdf'):
      for side in ('left','right'):
        robot=rk.HandKinematics('WujiHand2',side);human=HumanHandKinematics(side,root/'assets'/f'{config}_{side}.urdf',sdk_tx_rounding=True)
        source_order[side]=dict(names=robot.joint_names,limits_rad=robot.bounds.tolist())
        for item in selection:
          index=item['index'];gt=dict(np.load(root/'gt'/config/f'trajectory_{index}_{side}.npz'))
          pred=dict(np.load(root/prediction_subdir/config/f'trajectory_{index}_{side}.npz'))
          observed_R=quaternion_matrix(gt['orientation_xyzw']);seconds=(gt['timestamp_ns']-gt['timestamp_ns'][0])*1e-9
          gt_robot=robot_fk_batch(robot,gt['q_sdk_rad'])
          np.testing.assert_allclose(gt_robot[0],robot.forward(gt['q_sdk_rad'][0])[0],atol=1e-12,rtol=0)
          for method,key in [('network','q_network_rad'),('native_raw','q_lm_raw_rad'),('native_filtered','q_lm_filtered_rad')]:
            q=pred[key];fk=robot_fk_batch(robot,q);distance=np.linalg.norm(fk-gt_robot,axis=-1);tips=distance[:,[4,8,12,16,20]]
            error=q-gt['q_sdk_rad']
            row=dict(configuration=config,side=side,trajectory=index,method=method,frames=len(q),
                landmark_position_rmse_m=float(np.sqrt(np.mean(distance**2))),landmark_position_max_m=float(distance.max()),
                fingertip_position_rmse_m=float(np.sqrt(np.mean(tips**2))),fingertip_position_max_m=float(tips.max()),
                per_finger_tip_position_rmse_m=np.sqrt(np.mean(tips**2,axis=0)).tolist(),
                max_joint_limit_violation_rad=float(np.maximum(np.maximum(robot.lower-q,q-robot.upper),0).max()))
            rows.append(row)
            startup.append(dict(configuration=config,side=side,trajectory=index,method=method,
                first_second_frames=int((seconds<1).sum()),first_second_q_rmse_rad=float(np.sqrt(np.mean(error[seconds<1]**2))),
                after_first_second_frames=int((seconds>=1).sum()),after_first_second_q_rmse_rad=float(np.sqrt(np.mean(error[seconds>=1]**2))) if (seconds>=1).any() else None))
          for method in ('sdk','native_human_lm'):
            if method=='sdk':p_h,r_h=gt['rx_sdk_tx_position_m'],gt['rx_sdk_tx_rotation']
            else:p_h,r_h=receiver_fk_batch(human,pred['human_q_lm_rad'])
            position_error=np.linalg.norm(p_h-gt['position_tx_m'],axis=-1)
            orientation_error=np.arccos(np.clip((np.trace(r_h.swapaxes(-1,-2)@observed_R,axis1=-2,axis2=-1)-1)*.5,-1,1))
            rows.append(dict(configuration=config,side=side,trajectory=index,method=method,frames=len(p_h),
                human_emf_position_rmse_m=float(np.sqrt(np.mean(position_error**2))),human_emf_position_max_m=float(position_error.max()),
                human_emf_orientation_rmse_rad=float(np.sqrt(np.mean(orientation_error**2))),human_emf_orientation_max_rad=float(orientation_error.max())))
    report=dict(scope='Post-processing only. Robot FK differences use the identical Wuji2 URDF for predicted/SDK q. Human EMF fitting describes the intermediate human reconstruction, not a robot EMF forward model.',metrics=rows,startup=startup)
    (root/'fk_and_startup_metrics.json').write_text(json.dumps(report,indent=2));(root/'joint_order_and_limits.json').write_text(json.dumps(source_order,indent=2))
    print('FK_AND_STARTUP_METRICS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
