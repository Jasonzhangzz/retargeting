#!/usr/bin/env python3
"""Frozen best/latest checkpoints on the five excluded native EMF episodes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def metric(errors):
    e=np.concatenate(errors,axis=0)
    return dict(frames=len(e),mae_rad=float(np.abs(e).mean()),rmse_rad=float(np.sqrt((e*e).mean())),
        max_abs_rad=float(np.abs(e).max()),per_joint_mae_rad=np.abs(e).mean(0).tolist(),
        per_joint_rmse_rad=np.sqrt((e*e).mean(0)).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--holdout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.campaign;out=a.output;out.mkdir(parents=True,exist_ok=False)
    conf=json.loads((root/'MANIFEST.json').read_text());sys.path[:0]=[str(root/'scripts'),conf['reference']]
    from build_wuji_real_sdk_dataset import sdk_label
    from retarget.learning.wuji_ik.training import load_inference
    from retarget.learning.wuji_ik.metrics import autocast_context
    plan=json.loads((root/'selection.json').read_text());held=json.loads((a.holdout/'selection.json').read_text())['trajectories']
    assert len(held)==5 and {r['episode'] for r in held}<=set(conf['heldout_episode_paths'])
    trained_hash={w['source_sha256'] for w in plan['segments']};trained_records={str(Path(w['source_h5']).parent) for w in plan['segments']}
    provenance=json.loads((root/'sdk_provenance.json').read_text());old_gt=json.loads((a.holdout/'gt_manifest.json').read_text())
    assert provenance['sdk_version']==old_gt['sdk_version'] and provenance['binaries']==old_gt['sdk_binary_sha256']
    streams=[]
    for r in held:
        source=Path(r['cached_h5']);sha=digest(source)
        assert sha==r['h5_sha256'] and sha not in trained_hash and r['episode'] not in trained_records
        for side in ['right','left']:
            cal=r['recording_metadata']['devices'][side];assert cal['urdf_source']=='builtin_default'
            gt_path=a.holdout/'gt/builtin'/f'trajectory_{r["index"]}_{side}.npz'
            with np.load(gt_path) as old:
                n=len(old['q_sdk_rad']);oldq=old['q_sdk_rad'].copy()
            window=dict(id=f'holdout_{r["index"]}_{side}',side=side,source_h5=str(source),source_sha256=sha,
                start=0,end=n,calibration=dict(sn=cal['sn'],urdf=str(root/'assets'/f'builtin_{side}.urdf'),override=False))
            _,data=sdk_label((root,window,provenance['label_signature'],False))
            difference=float(abs(data['q_sdk_rad']-oldq).max())
            assert difference<1e-7, 'Prior GT protocol differs from training SDK teacher'
            assert np.all(np.diff(data['timestamp_ns'])>0) and np.all(np.diff(data['timestamp_ns'])<=50_000_000)
            stream=dict(index=r['index'],collection=r['collection'],side=side,data=data,source_sha256=sha,
                previous_gt_max_difference_rad=difference)
            streams.append(stream);print('SDK_HOLDOUT_VERIFIED',r['index'],side,n,difference,flush=True)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    result=dict(complete=False,optimizer_updates=0,expected_trajectories=5,expected_streams=10,
        expected_frames=sum(len(s['data']['q_sdk_rad']) for s in streams),test_scope='Five complete held-out real episodes; native builtin calibration; full causal context; SDK q teacher; no LM',
        no_episode_or_content_overlap=True,training_data_signature=provenance['label_signature'],
        sdk_version=provenance['sdk_version'],evaluated_at=datetime.now(timezone.utc).isoformat(),checkpoints={})
    for name in ['best','latest']:
        checkpoint=root/'temporal_causal'/f'{name}.pt';ckhash=digest(checkpoint)
        model,normalizer,state=load_inference(checkpoint,'cuda:0','auto')
        assert state['data_signature']==provenance['label_signature'] and state['variant']=='temporal_causal'
        errs=[];delta=[];sides={side:[] for side in ['right','left']};details=[]
        with torch.no_grad():
            for stream in streams:
                data=stream['data'];n=len(data['features']);x=torch.from_numpy(data['features'][None]).to('cuda:0')
                side=torch.tensor([0 if stream['side']=='right' else 1],device='cuda:0');length=torch.tensor([n],device='cuda:0')
                torch.cuda.synchronize();started=time.perf_counter()
                with autocast_context(torch.device('cuda:0'),True):pred=model(x,side,length)
                rad=normalizer.denormalize(pred.float(),side)[0].cpu().numpy();torch.cuda.synchronize();elapsed=time.perf_counter()-started
                error=rad.astype(np.float64)-data['q_sdk_rad'];assert np.isfinite(error).all()
                errs.append(error);delta.append(np.diff(error,axis=0));sides[stream['side']].append(error)
                info={k:v for k,v in stream.items() if k!='data'};info.update(metric([error]),model_seconds=elapsed)
                details.append(info);np.savez_compressed(out/f'{name}_trajectory_{stream["index"]}_{stream["side"]}.npz',q_pred_rad=rad,q_sdk_rad=data['q_sdk_rad'],timestamp_ns=data['timestamp_ns'],error_rad=error)
                print('PREDICTED',name,stream['index'],stream['side'],info['rmse_rad'],flush=True)
        overall=metric(errs);d=np.concatenate(delta);overall.update(delta_mae_rad=float(abs(d).mean()),delta_rmse_rad=float(np.sqrt((d*d).mean())))
        report=dict(checkpoint_path=str(checkpoint),checkpoint_sha256=ckhash,checkpoint_step=state['step'],completed_epochs_in_checkpoint=state['epoch'],
            metrics=dict(overall=overall,**{'side/'+side:metric(e) for side,e in sides.items()}),streams=details,attention_backends=model.core.backend_report())
        result['checkpoints'][name]=report
        (out/'report.json').write_text(json.dumps(result,indent=2));print('CHECKPOINT_COMPLETE',name,json.dumps(overall),flush=True)
        del model,normalizer,state;torch.cuda.empty_cache()
    result['complete']=True;result['passed_protocol']=True
    (out/'report.json').write_text(json.dumps(result,indent=2));print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
