"""Synthetic, non-publishable capture follower interruption/restart smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_cart_pipeline import fixtures
from retarget.baseline.geort.adapter import ROOT
from retarget.baseline.geort.training import atomic_json
from moxian_web.cart_capture_schema import CAPTURE_SCHEMA


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recording',type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args();torch.set_num_threads(1)
    source=args.recording or next((ROOT/'output/moxian/cart/recordings').glob('*/manifest.json')).parent
    root=args.output or ROOT/'output/moxian/cart/capture_smoke_tests'/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:6])
    root.mkdir(parents=True,exist_ok=False);fixtures(root,source,count=16)
    folder=root/'recordings/train';raw_path=folder/'frames.jsonl';raw=raw_path.read_bytes()
    meta=json.loads((folder/'manifest.json').read_text());meta['capture_geometry_schema']=CAPTURE_SCHEMA
    atomic_json(folder/'manifest.json',dict(meta,complete=False,frames=0))
    raw_path.write_bytes(b'')
    command=[sys.executable,'-u','-m','moxian_web.cart_cli','--root',str(root),'capture','--dataset','train','--follow','--device','cpu']
    def launch():
        with (root/'process.log').open('a') as log:
            return subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    proc=launch()
    try:
        # Feed the independent disk follower rather than a latest-frame mailbox.
        with raw_path.open('ab') as stream:
            for line in raw.splitlines(keepends=True):stream.write(line);stream.flush()
        deadline=time.monotonic()+30
        status={}
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError((root/'process.log').read_text()[-3000:])
            try:status=json.loads((folder/'cart_capture_status.json').read_text())
            except (OSError,ValueError):pass
            if status.get('frames',0)>0:break
            time.sleep(.02)
        assert status.get('frames',0)>0,'capture worker did not consume raw frames'
        proc.terminate();proc.wait(timeout=10)
        prefix=sum(1 for _ in (folder/'cart_capture.pending.jsonl').open())
        assert prefix>0 and not (folder/'cart_capture.jsonl').exists()
        with (folder/'cart_capture.pending.jsonl').open('ab') as stream:stream.write(b'{"timestamp_ms":')
        atomic_json(folder/'manifest.json',dict(meta,complete=True,frames=16))
        proc=launch();assert proc.wait(timeout=30)==0,(root/'process.log').read_text()[-3000:]
        rows=[json.loads(line) for line in (folder/'cart_capture.jsonl').open()]
        info=json.loads((folder/'cart_capture_meta.json').read_text())
        assert len(rows)==16 and info['resumed_prefix_frames']==prefix
        assert [r['timestamp_ms'] for r in rows]==[float(i*20) for i in range(16)]
        assert hashlib.sha256(raw_path.read_bytes()).digest()==hashlib.sha256(raw).digest()
        for row in rows:
            assert row['coordinate_frame']=='wrist_local' and row['units']=='m'
            assert len(row['tactile_state'])==460 and set(row['tactile_state'])<={0,1}
            assert np.asarray(row['tactile_positions_m']).shape==(460,3)
            assert np.asarray(row['tactile_normals']).shape==(460,3)
            assert len(row['tactile_signal_positions_m'])==len(row['tactile_signal_sids'])
            assert row['contacts']['self_penetrations']['truncated'] is False
        assert rows[2]['tactile_signal_sids'] and not rows[5]['tactile_signal_sids']
        result=dict(output=str(root),frames=len(rows),resumed_prefix_frames=prefix,raw_unchanged=True,
            binary_taxels=460,all_positions_wrist=True,tactile_synthetic=True,emf_synthetic=True,published=False)
        atomic_json(root/'capture_smoke_report.json',result);print(json.dumps(result),flush=True)
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)


if __name__=='__main__':main()
