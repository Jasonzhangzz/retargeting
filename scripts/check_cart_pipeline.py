"""Non-publishable synthetic EMF fixtures: preprocess, both policies, resume/replay.

Source motion supplies geometry only. Generated EMF is explicitly synthetic and
fixture_only; this is plumbing validation, not real sensor/contact performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def fixtures(root,source,count=16):
    from moxian_web.cart_data import SourceGeometry
    from moxian_web.cart_protocol import PROTOCOL
    from retarget.baseline.geort.training import atomic_json
    source_meta=json.loads((source/'manifest.json').read_text());motion=np.load(source/'motion.npz')
    points=np.load(source/'points.npy')
    geometry=SourceGeometry(json.loads((source/'topology.json').read_text()),json.loads((source/'profile.json').read_text()),'cpu')
    if len(points)<count*2:raise ValueError('Smoke fixture needs two disjoint clips')
    for part,split in enumerate(('train','validation')):
        folder=root/'recordings'/split;folder.mkdir(parents=True)
        ids=np.arange(part*count,(part+1)*count)
        for name in ('profile.json','topology.json'):shutil.copyfile(source/name,folder/name)
        # Explicit fixture time base avoids inheriting source gaps. No real
        # recording is relabelled as synchronized EMF ground truth.
        stamps=np.arange(count,dtype=np.float64)*20+part*1000
        np.save(folder/'points.npy',points[ids]);np.save(folder/'timestamps.npy',stamps)
        np.savez(folder/'motion.npz',qpos=motion['qpos'][ids],joints=motion['joints'][ids])
        frames=[]
        sid=int(np.flatnonzero((np.asarray(geometry.topology['sid_patch_indices'])>=0)&(np.asarray(geometry.topology['sid_patch_indices'])<14))[0])
        with torch.no_grad():
            q=torch.as_tensor(motion['qpos'][ids],device=geometry.model.device)
            p,r=geometry.model.sensor_prediction(q);a,t=geometry.model.alignment()
            rawp=(p-t)@a;rawr=a.T@r
        for k,(i,stamp) in enumerate(zip(ids,stamps)):
            emf=dict(timestamp_ms=float(stamp),coordinate_frame='wrist_local',positions_m=p[k].tolist(),rotations=r[k].tolist(),
                raw_positions_m=rawp[k].tolist(),raw_rotations=rawr[k].tolist(),valid=[True]*5,
                calibration_fingerprint=source_meta['source']['calibration_fingerprint'],
                skeleton_runtime_revision=source_meta['source']['skeleton_runtime_revision'])
            cells=[False]*460;cells[sid]=2<=k<=4
            contact=dict(ready=k!=0,timestamp_ms=float(stamp),cell_contact=cells,normalized=[float(v) for v in cells])
            frames.append(dict(timestamp_ms=float(stamp),qpos=motion['qpos'][i].tolist(),joints_m=motion['joints'][i].tolist(),
                points=points[i].tolist(),source_contact=contact,tactile_timestamp_ms=float(stamp),emf_observation=emf))
        (folder/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
        atomic_json(folder/'manifest.json',dict(source_meta,id=split,protocol=PROTOCOL,capture_mode='cart_v5',split=split,
            session_id='synthetic_fixture_'+split,frames=count,fixture_only=True,emf_synthetic=True,complete=True,
            derived_from_session=source_meta['id'],tactile_recorded=True,tactile_synthetic=True,
            points_sha256=hashlib.sha256((folder/'points.npy').read_bytes()).hexdigest()))


def runtime_probe(root,model):
    from unittest.mock import patch
    from retarget.baseline.geort.runtime import Replay
    from moxian_web import cart_runtime
    from moxian_web.cart_training import Predictor
    replay=Replay(root/'recordings/validation');replay.playing=False
    from moxian_web.cart_ik_view import IKReplay
    with patch('retarget_reference.session.RetargetSession.step',side_effect=AssertionError('Playback must not solve IK')):
        ik=IKReplay(root/'recordings/validation');ik_topology=ik.topology()
        assert ik_topology['frames']==len(replay.stamps)
        with np.load(root/'recordings/validation/cart_ik.npz') as saved:
            for index in (0,len(replay.stamps)//2,len(replay.stamps)-1):
                packet=ik.frame(index)
                assert packet['source_timestamp_ms']==packet['robot_timestamp_ms']==float(replay.stamps[index])
                assert packet['selected']==bool(saved['valid'][index])
                if packet['selected']:np.testing.assert_array_equal(packet['joint_positions_rad'],saved['q'][index])
    with patch.object(cart_runtime,'Predictor',lambda folder,device:Predictor(folder,device,allow_smoke=True)):
        runtime=cart_runtime.CartRetarget(replay.topology,device='cpu',model_path=model)
        ready=[]
        for i in range(len(replay.stamps)):
            replay.index=i;replay.last_index=-1;frame=replay.frame()
            result=runtime.step(frame)
            if i<10:assert result['status']=='buffering'
            else:
                assert result['ready'] and result['recorded_timestamp_ms']==float(replay.stamps[i-9])
                np.testing.assert_allclose(result['source_joints_m'],
                    (runtime.buffer.entries[0][1][0]['joints_m'][runtime.source.indices])@runtime.source.mirror.T,atol=1.e-6)
                ready.append(result['joint_positions_rad'])
        replay.seek(0);frame=replay.frame();assert runtime.step(frame)['status']=='buffering'
        frame['cart_future_frames']=replay.future_frames(frame)
        paused=runtime.step(frame)
        assert paused['ready'] and paused['recorded_timestamp_ms']==float(replay.stamps[1])
        assert paused['cart']['buffer_delay_ms']==0.
        replay.seek(len(replay.stamps)-1);frame=replay.frame();frame['cart_future_frames']=replay.future_frames(frame)
        assert runtime.step(frame)['status']=='buffering'
    return dict(buffered_outputs=len(ready),seek_resets_buffer=True,paused_replay_uses_saved_future=True,tail_extrapolated=False,
        ik_cached_playback=True,ik_playback_solves=False)


def main():
    from moxian_web.cart_data import preprocess
    from moxian_web.cart_training import train,evaluate
    from retarget.baseline.geort.adapter import ROOT
    from retarget.baseline.geort.training import atomic_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--architecture',choices=('both','attention','fsq'),default='both')
    parser.add_argument('--device',default='cpu');parser.add_argument('--length',type=int,default=2)
    parser.add_argument('--initialize-sdf-from',type=Path,help='Also validate importing/restoring an articulated Neural SDF')
    args=parser.parse_args();torch.set_num_threads(1)
    # Compare checkpoint recovery under controlled reductions. Fast BF16 CUDA
    # attention may use nondeterministic backward reductions; two fresh runs
    # can otherwise differ before either one has been interrupted.
    if args.device.startswith('cuda'):torch.use_deterministic_algorithms(True)
    source=args.recording or next((ROOT/'output/moxian/cart/recordings').glob('*/manifest.json')).parent
    root=args.output or ROOT/'output/moxian/cart/smoke_tests'/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:6])
    root.mkdir(parents=True,exist_ok=False);fixtures(root,source,count=max(16,args.length+12))
    for split in ('train','validation'):preprocess(root,split,full=False)
    reports={}
    for architecture in ('attention','fsq') if args.architecture=='both' else (args.architecture,):
        model=train(root,['train'],['validation'],architecture,epochs=1,batch_size=1,length=args.length,smoke=True,
            architecture=architecture,device=args.device,initialize_sdf_from=args.initialize_sdf_from)
        # Reload the entire training state, including the replay and optimizers.
        train(root,['train'],['validation'],architecture,epochs=1,batch_size=1,length=args.length,smoke=True,
            architecture=architecture,resume=True,device=args.device)
        # Interrupt after a committed minibatch, then compare against the
        # uninterrupted run. This exercises order, replay and both optimizers.
        from unittest.mock import patch
        from moxian_web import cart_training
        original=cart_training.save_checkpoint
        def interrupt(path,values):
            original(path,values)
            if Path(path).name=='resume.pth':raise InterruptedError('synthetic interruption')
        interrupted=architecture+'_resume_probe'
        with patch.object(cart_training,'save_checkpoint',interrupt):
            try:
                train(root,['train'],['validation'],interrupted,epochs=1,batch_size=1,length=args.length,smoke=True,
                    architecture=architecture,device=args.device,initialize_sdf_from=args.initialize_sdf_from)
            except InterruptedError:pass
        train(root,['train'],['validation'],interrupted,epochs=1,batch_size=1,length=args.length,smoke=True,
            architecture=architecture,device=args.device,resume=True)
        a=torch.load(model/'resume.pth',map_location='cpu',weights_only=True)
        b=torch.load(root/'models'/interrupted/'resume.pth',map_location='cpu',weights_only=True)
        max_delta=0.
        for group in ('model','collision'):
            for key in a[group]:
                torch.testing.assert_close(a[group][key],b[group][key],rtol=1.e-5 if args.device.startswith('cuda') else 0,atol=1.e-6 if args.device.startswith('cuda') else 0)
                max_delta=max(max_delta,float((a[group][key]-b[group][key]).abs().max()))
        result=evaluate(root,architecture,['validation'],allow_smoke=True)
        result.update(interrupted_resume_exact=max_delta==0.,interrupted_resume_max_delta=max_delta,interrupted_resume_within_tolerance=True)
        result.update(runtime_probe(root,model));reports[architecture]=result
    atomic_json(root/'smoke_report.json',dict(architectures=reports,emf_synthetic=True,published=False,source=str(source),
        deterministic_cuda_check=args.device.startswith('cuda')))
    print(json.dumps(dict(output=str(root),reports=reports),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
