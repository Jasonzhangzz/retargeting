#!/usr/bin/env python3
"""Run the audited 31x200 pilot, then the 31x50,000 production campaign."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'data/datasets/wuji2_primitive_aug_v1')
    ap.add_argument('--pilot-output',type=Path,default=ROOT/'data/datasets/wuji2_primitive_aug_pilot_v1')
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    module='retarget.env.grasp.primitive_augmentation'
    for root,count in ((args.pilot_output,200),(args.output,50000)):
        if (root/'config.json').exists():
            command=[sys.executable,'-m',module,'resume','--output',str(root),'--device',args.device]
        else:
            command=[sys.executable,'-m',module,'generate','--output',str(root),'--per-type',str(count),'--device',args.device]
            if count==50000:command+=['--bootstrap-from',str(args.pilot_output)]
        subprocess.run(command,cwd=ROOT,check=True)
        state=json.loads((root/'status.json').read_text())
        if state['phase']!='complete':raise RuntimeError(f'Campaign not complete: {root} ({state["phase"]})')
        if count==200:
            audit=json.loads((root/'audit_report.json').read_text())
            if audit['invalid']:raise RuntimeError('Pilot audit rejected rows; refill and re-audit before production')


if __name__=='__main__':main()
