"""Replay stored raw observations through the same official calibrated C engine."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.native_skeleton import NativeSkeleton

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--fingerprint',required=True);a=ap.parse_args()
    from qnbot_sdk.glove._resources import builtin_skeleton_package_root
    payload=json.loads(a.input.read_text());engine=NativeSkeleton(builtin_skeleton_package_root(),a.config)
    try:
        if engine.lib.qh_engine_physics_enabled(engine.handle):raise ValueError('离线转换必须关闭动力学')
        for row in payload['rows']:
            result=engine.step(row)
            row['skeleton_original']=row.get('skeleton');row['skeleton']=result['pose'];row['skeleton_targets']=result['targets'];row['calibration_fingerprint']=a.fingerprint
    finally:engine.close()
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False));temp.replace(a.output)
if __name__=='__main__':main()
