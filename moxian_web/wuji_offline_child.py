"""Python 3.12 public SDK offline IK sidecar; no devices or SDK manager."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SDK=ROOT/'.cache/wuji-calibration-native/python312'
ELF_SHA='dddeba3148f6a1e9eae87e5830eea25fc39f5c8bcb7cb958749db6bce82b621c'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--event-fd',type=int,required=True);args=parser.parse_args()
    if sys.version_info[:2]!=(3,12):raise RuntimeError('原生离线IK需要Python3.12')
    elf=SDK/'wuji_sdk/_wuji_sdk.cpython-312-x86_64-linux-gnu.so'
    if hashlib.sha256(elf.read_bytes()).hexdigest()!=ELF_SHA:raise RuntimeError('离线IK原生ELF指纹不匹配')
    sys.path.insert(0,str(SDK));import wuji_sdk as w
    if w.__version__!='2026.7.1':raise RuntimeError('离线IK版本不匹配')
    stream=os.fdopen(args.event_fd,'w',buffering=1);pipeline=None;sequence=0
    for line in sys.stdin:
        try:
            data=json.loads(line)
            if data.get('action')=='load':
                pipeline=w.WujiGlove.offline_pipeline(sn=data['serial'],hand_side=data['side'],urdf_path=data['urdf_path']);sequence=0
                result=dict(ready=True,version=w.__version__)
            else:
                if pipeline is None:raise RuntimeError('模型未加载')
                sequence+=1
                emf=w.EmfPoseArray(header=w.FrameHeader(seq=sequence,timestamp_us=round(data['timestamp_ms']*1000),frame_id='moxian_emf_tx'),
                    poses=[w.EmfPose(pose=w.Pose(position=n['position'],orientation=w.Quaternion(*n['quaternion_xyzw'])),confidence=n.get('confidence',1.)) for n in data['poses']])
                q=pipeline.hand_joint_angles().compute(emf)
                result=dict(angles=[list(f.angles) for f in q.fingers],confidence=[float(f.confidence) for f in q.fingers],timestamp_ms=data['timestamp_ms'])
                if data.get('include_skeleton'):
                    skeleton=pipeline.hand_skeleton().compute(emf)
                    result['skeleton_names']=[j.name for j in skeleton.joints]
                    result['skeleton_positions']=[list(j.pose.position) for j in skeleton.joints]
            stream.write(json.dumps(result,allow_nan=False)+'\n')
        except Exception as exc:stream.write(json.dumps(dict(error=str(exc)))+'\n')

if __name__=='__main__':main()
