#!/usr/bin/env python3
"""Run the production MJWarp ADMM controller for the box task."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_humanoid import main as run_humanoid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--fd-batch',type=int,default=64)
    p.add_argument('--fd-step',type=float,default=.01)
    p.add_argument('--rho',type=float,default=1.)
    a=p.parse_args()
    output=a.output.resolve()
    argv=['--task','humanoid-object','--cost','spider','--output-base',str(output.parent),
          '--case-name',output.name,'--device',a.device,'--mjwarp-fd-batch',str(a.fd_batch),
          '--mjwarp-fd-step',str(a.fd_step),'--rho',str(a.rho),'--anchor-weight','0']
    if a.frames is not None:argv+=['--frames',str(a.frames)]
    # Strict mode writes its dense trajectory and tracking summary itself.
    return run_humanoid(argv)


if __name__=='__main__':main()
