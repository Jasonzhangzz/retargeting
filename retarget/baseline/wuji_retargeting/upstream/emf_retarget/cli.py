"""Command-line conversion of recorded EMF poses into independent hand outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from . import __version__
from .pipeline import EmfRetargetPipeline
from .io import iter_emf_frames, output_arrays


def main(argv=None):
    parser=argparse.ArgumentParser(description='EMF receiver poses -> human IK -> Wuji1/Wuji2 joint angles')
    parser.add_argument('input',type=Path,help='native glove .h5 or single-hand .npz')
    parser.add_argument('--output','-o',type=Path,required=True,help='output .npz (angles in radians, positions in metres)')
    parser.add_argument('--side',choices=['left','right','both'],default='both')
    parser.add_argument('--models',nargs='+',choices=['wuji1','wuji2'],default=['wuji1','wuji2'])
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--human-urdf',type=Path,help='single-side override; same expected CQQ joint/link naming')
    parser.add_argument('--report',type=Path,help='JSON manifest, default <output>.json')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    if args.side=='both' and (args.human_urdf is not None or args.input.suffix.lower()=='.npz'):
        parser.error('NPZ input and a URDF override each require --side left or right')
    if args.start<0 or (args.limit is not None and args.limit<=0):parser.error('start >= 0 and limit > 0 required')
    if args.output.suffix.lower()!='.npz':parser.error('--output must end in .npz')
    report_path=args.report or args.output.with_suffix('.json')
    if args.input.resolve() in (args.output.resolve(),report_path.resolve()):parser.error('output must differ from input')
    if report_path.resolve()==args.output.resolve():parser.error('report must differ from output')
    for path in (args.output,report_path):
        if path.exists() and not args.overwrite:parser.error(f'{path} exists; use a new path or --overwrite')
    try:
        t0=time.monotonic();arrays={};summary={}
        for side in (('left','right') if args.side=='both' else (args.side,)):
            pipeline=EmfRetargetPipeline(side,args.models,args.human_urdf);results=[]
            for frame in iter_emf_frames(args.input,side,args.start,args.limit):
                results.append(pipeline.process(frame))
                if len(results)%100==0:print(f'{side}: {len(results)} frames',file=sys.stderr,flush=True)
            arrays.update(output_arrays(results,pipeline))
            model_paths=[pipeline.human.kinematics.urdf_path]+[r.kinematics.urdf_path for r in pipeline.robots.values()]
            summary[side]=dict(frames=len(results),frames_computed=pipeline.frames_computed,
                human_urdf=str(pipeline.human.kinematics.urdf_path),models=list(pipeline.models),
                asset_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_paths})
        report=dict(version=__version__,input=str(args.input.resolve()),output=str(args.output.resolve()),
            start=args.start,limit=args.limit,sides=summary,elapsed_seconds=time.monotonic()-t0,
            units=dict(angles='rad',positions='m',timestamp='us'),
            shapes={key:list(value.shape) for key,value in arrays.items()},
            runtime='independent NumPy human IK and NLopt retarget; no wuji_sdk runtime import',
            model_scope='explicit bundled CQQ human URDF and built-in Wuji1/Wuji2 robot URDFs')
        args.output.parent.mkdir(parents=True,exist_ok=True);report_path.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('wb' if args.overwrite else 'xb') as stream:np.savez_compressed(stream,**arrays)
        with report_path.open('w' if args.overwrite else 'x',encoding='utf8') as stream:json.dump(report,stream,indent=2,ensure_ascii=False)
        print(json.dumps(dict(output=str(args.output),report=str(report_path),frames={s:v['frames'] for s,v in summary.items()})))
        return 0
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f'emf-retarget: {exc}',file=sys.stderr)
        return 2
