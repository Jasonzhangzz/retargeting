"""Read-only SDK oracle: independent raw EMF -> human -> both Wuji models.

Run from the project root: python tests/verify_end_to_end.py.
The SDK and reference receive only the same raw sensor frame and explicit URDF.
Neither receives the other's human q, skeleton, robot q, or solver state.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from emf_retarget import EmfFrame, EmfRetargetPipeline
assert not any(k=='wuji_sdk' or k.startswith('wuji_sdk.') for k in sys.modules)

EXPECTED='a179dcb71d0a5675abe0570d83201db726e545ec7583aba34ae0bcdc54ac7fd5'
EP=Path('/mnt/jfs/x2robot-prod/x2robot_data/collection/18/RP001_VR/XRRD11ZD2604070080/20260921120322-stack_cup_normal_speed_CQQ_rpAAAA/20260921120322-stack_cup_normal_speed_CQQ_rpAAAA@MASTER_SLAVE_MODE@2026_09_21_12_11_08/wuji_glove_offline.h5')


def text(x):return x.decode() if isinstance(x,bytes) else str(x)


def metrics(values):
    x=np.asarray(values,dtype=float)
    return {'n':int(x.size),'max_abs':float(np.max(np.abs(x),initial=0.)),
            'rmse':float(np.sqrt(np.mean(x*x))) if x.size else 0.}


def angular_errors(a,b):
    # Quaternion distance independent of the +/- representation, stable near 0.
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a/np.linalg.norm(a,axis=1)[:,None];b=b/np.linalg.norm(b,axis=1)[:,None]
    b=np.where((np.sum(a*b,axis=1)<0)[:,None],-b,b)
    return 4.*np.arctan2(np.linalg.norm(a-b,axis=1),np.linalg.norm(a+b,axis=1))


def sdk_pose_arrays(values):
    return (np.array([v.pose.position for v in values],dtype=np.float32),
            np.array([[getattr(v.pose.orientation,k) for k in 'xyzw'] for v in values],dtype=np.float32))


def load_sdk():
    spec=importlib.util.find_spec('wuji_sdk')
    package=Path(next(iter(spec.submodule_search_locations)))
    binaries=list(package.glob('wuji_sdk.cpython-310-x86_64-linux-gnu.so'))
    assert len(binaries)==1
    assert hashlib.sha256(binaries[0].read_bytes()).hexdigest()==EXPECTED
    return importlib.import_module('wuji_sdk')


def diagnose_existing(path, sdk):
    """Separate same-input implementation error from upstream sensitivity.

    This diagnostic deliberately replays intermediate skeletons to each engine;
    it is NOT the independent raw-EMF comparison recorded in the source report.
    """
    from retarget_reference import RetargetSession
    source=json.loads(path.read_text(encoding='utf8'))
    output={'scope':'diagnostic replay of saved skeletons; not an independent EMF test',
            'source_report':str(path),'sdk_sha256':EXPECTED,'cases':[]}
    for segment in source['segments']:
        if segment['start']!=0 or segment['side']!='left':continue
        side=segment['side']
        for model,name in [('WujiHand','wuji1'),('WujiHand2','wuji2')]:
            native={key:sdk.RetargetSession.for_hand(getattr(sdk.HandModel,model),
                     side=getattr(sdk.Handedness,side.title())) for key in ('sdk','reference')}
            independent={key:RetargetSession.for_hand(model,side) for key in native}
            errors={key:[] for key in ('same_sdk_skeleton','same_reference_skeleton',
                                       'native_input_sensitivity','reference_input_sensitivity')}
            rows=[]
            for row in segment['rows']:
                nq={key:np.asarray(native[key].step(np.asarray(row['skeleton_'+key],dtype=np.float32)),dtype=float)
                    for key in native}
                rq={key:independent[key].step(np.asarray(row['skeleton_'+key],dtype=np.float32)).astype(float)
                    for key in independent}
                diff={'same_sdk_skeleton':rq['sdk']-nq['sdk'],
                      'same_reference_skeleton':rq['reference']-nq['reference'],
                      'native_input_sensitivity':nq['reference']-nq['sdk'],
                      'reference_input_sensitivity':rq['reference']-rq['sdk']}
                for key,value in diff.items():errors[key].extend(value.tolist())
                rows.append({'index':row['index'],'max_abs':{key:metrics(v)['max_abs'] for key,v in diff.items()}})
            item={'side':side,'model':name,'frames':len(rows),
                  'metrics':{key:metrics(v) for key,v in errors.items()},'rows':rows}
            output['cases'].append(item)
            print(json.dumps({k:v for k,v in item.items() if k!='rows'}),flush=True)
    destination=path.with_name('end-to-end-sensitivity.json')
    destination.write_text(json.dumps(output,indent=2),encoding='utf8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=EP)
    parser.add_argument('--frames',type=int,default=64)
    parser.add_argument('--starts',type=int,nargs='+',default=[0,256])
    parser.add_argument('--output',type=Path,default=ROOT/'end-to-end-verification.json')
    parser.add_argument('--diagnose-existing',action='store_true')
    args=parser.parse_args()
    assert args.frames>0 and all(start>=0 for start in args.starts)
    sdk=load_sdk()
    if args.diagnose_existing:
        diagnose_existing(args.output,sdk)
        return
    report={'scope':'independent full raw EMF -> human q/skeleton/tips -> Wuji1/Wuji2; explicit CQQ URDF',
            'sdk_sha256':EXPECTED,'input':str(args.input),'frames_per_side_per_segment':args.frames,
            'starts':args.starts,'sdk_intermediates_shared_with_reference':False,
            'reference_imported_without_sdk':True,'segments':[],
            'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in ('emf_retarget','human_ik_reference','retarget_reference')
                       for p in sorted((ROOT/folder).rglob('*'))
                       if p.is_file() and '__pycache__' not in str(p)}}
    with h5py.File(args.input,'r',locking=False) as h:
        metadata=json.loads(text(h.attrs['metadata_json']))
        for start in args.starts:
            for side in ('left','right'):
                group=h[side];stop=start+args.frames
                assert stop<=len(group['source_seq']), 'requested segment unavailable'
                urdf=ROOT/'human_ik_reference/assets'/f'{side}_hand.urdf'
                oracle=sdk.WujiGlove.offline_pipeline(sn=metadata['devices'][side]['sn'],hand_side=side,urdf_path=str(urdf))
                robots={name:sdk.RetargetSession.for_hand(getattr(sdk.HandModel,model),
                            side=getattr(sdk.Handedness,side.title()))
                        for name,model in [('wuji1','WujiHand'),('wuji2','WujiHand2')]}
                reference=EmfRetargetPipeline(side)
                errors={key:[] for key in ('human_q_rad','human_confidence','skeleton_position_m',
                     'skeleton_rotation_rad','skeleton_confidence','tip_position_m','tip_rotation_rad',
                     'wuji1_q_rad','wuji2_q_rad')}
                rows=[];t0=time.monotonic()
                positions=group['emf_poses/position'][start:stop]
                orientations=group['emf_poses/orientation_xyzw'][start:stop]
                confidence=group['emf_poses/confidence'][start:stop]
                for j,index in enumerate(range(start,stop)):
                    seq=int(group['source_seq'][index]);timestamp=int(group['source_timestamp_ns'][index]//1000)
                    frame_id=text(group['source_frame_id'][index])
                    raw=EmfFrame(positions[j],orientations[j],confidence[j],seq,timestamp,frame_id)
                    independent=reference.process(raw)
                    frame=sdk.EmfPoseArray(sdk.FrameHeader(seq,timestamp,frame_id),[
                        sdk.EmfPose(sdk.Pose(positions[j,f].tolist(),sdk.Quaternion(*orientations[j,f].tolist())),float(confidence[j,f]))
                        for f in range(5)])
                    angles=oracle.hand_joint_angles().compute(frame)
                    skeleton=oracle.hand_skeleton().compute(frame)
                    tips=oracle.tip_poses().compute(frame)
                    native_q=np.concatenate([np.asarray(v.angles)[:5 if f==0 else 4] for f,v in enumerate(angles.fingers)])
                    skpos,skrot=sdk_pose_arrays(skeleton.joints);tpos,trot=sdk_pose_arrays(tips.poses)
                    native_robot={name:r.step(skpos) for name,r in robots.items()}
                    differences={
                        'human_q_rad':independent.human_q-native_q,
                        'human_confidence':independent.human_confidence-np.array([v.confidence for v in angles.fingers]),
                        'skeleton_position_m':independent.skeleton_position-skpos,
                        'skeleton_rotation_rad':angular_errors(independent.skeleton_orientation_xyzw,skrot),
                        'skeleton_confidence':independent.skeleton_confidence-np.array([v.confidence for v in skeleton.joints]),
                        'tip_position_m':independent.tip_position-tpos,
                        'tip_rotation_rad':angular_errors(independent.tip_orientation_xyzw,trot),
                        **{name+'_q_rad':independent.robot_q[name].astype(float)-np.asarray(value,dtype=float) for name,value in native_robot.items()}}
                    for key,value in differences.items():
                        assert np.isfinite(value).all(),(side,index,key)
                        errors[key].extend(np.asarray(value).ravel().tolist())
                    rows.append({'index':index,'seq':seq,'timestamp_us':timestamp,'frame_id':frame_id,
                        'input_positions':positions[j].tolist(),'input_orientations_xyzw':orientations[j].tolist(),
                        'input_confidence':confidence[j].tolist(),
                        'human_q_reference':independent.human_q.tolist(),'human_q_sdk':native_q.tolist(),
                        'skeleton_reference':independent.skeleton_position.tolist(),'skeleton_sdk':skpos.tolist(),
                        'robot_q_reference':{k:v.tolist() for k,v in independent.robot_q.items()},
                        'robot_q_sdk':{k:np.asarray(v).tolist() for k,v in native_robot.items()},
                        'max_abs':{k:metrics(v)['max_abs'] for k,v in differences.items()}})
                    if j==0 or (j+1)%16==0:
                        print(json.dumps({'start':start,'side':side,'frame':index,'max_abs':rows[-1]['max_abs'],
                                          'elapsed_s':time.monotonic()-t0}),flush=True)
                summary={'start':start,'side':side,'frames':len(rows),'frames_computed':reference.frames_computed,
                         'metrics':{key:metrics(value) for key,value in errors.items()},
                         'elapsed_s':time.monotonic()-t0,'rows':rows}
                report['segments'].append(summary)
                report['completed_frames']=sum(s['frames'] for s in report['segments'])
                args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
                print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
    print(json.dumps({'completed_frames':report['completed_frames'],'report':str(args.output)}),flush=True)


if __name__=='__main__':main()
