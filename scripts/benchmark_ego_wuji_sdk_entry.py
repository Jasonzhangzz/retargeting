"""Time the actual ego_fast_retarget SDK CLI without modifying its sources.

Fixed screw-out record; 2 warmups +10 repeats of full CLI and max_frames=16.
Optional default-YAML glove configuration is tested separately and labeled.
No GPU, body IK, training, rendering, or publication is launched.
"""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    process_begin = time.perf_counter()
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--benchmark-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    r,b,out=a.repo.resolve(),a.benchmark_root.resolve(),a.output.resolve()
    assert not out.exists(), 'Existing output; inspect instead of implicit rerun'
    out.mkdir(parents=True)
    for path in [r,r/'replay',r/'utils']:sys.path.insert(0,str(path))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import yaml
    from wuji_retarget import sdk_retarget as cli
    from wuji_retarget import emf_offline as emf
    import wuji_sdk as sdk
    assert str(Path(cli.__file__).resolve()).startswith(str(r))
    assert str(Path(emf.__file__).resolve()).startswith(str(r))
    assert sdk.__version__=='2026.8.31'
    record=json.loads((b/'selection.json').read_text())['records'][0]
    episode=Path(record['source_h5']).parent
    source_files=[Path(cli.__file__),Path(emf.__file__),r/'utils/wuji_retarget/emf_orientation.py',r/'adapters/wuji_glove_utils.py']
    src=out/'source';src.mkdir()
    for p in source_files:(src/p.name).write_bytes(p.read_bytes())
    yaml_config=yaml.safe_load((r/'replay/calibrated/batch_replay.yaml').read_text())
    cfg={'sdk_version':'2026.8.31'}
    for side in ['left','right']:
        value=yaml_config.get(side+'_glove_urdf')
        cfg[side+'_urdf']=str(Path(value).resolve()) if value else None
        cfg[side+'_model_source']='override' if value else 'builtin_default'
    config_path=out/'current_yaml_glove_config.json';atomic(config_path,cfg)
    cases=[('builtin_full',[]),('builtin_max16',['--max-frames','16']),('current_yaml_glove_full',['--model-config',str(config_path)])]
    manifest=dict(authorization='User explicitly requested testing native SDK through ego_fast_retarget',repo=str(r),episode=str(episode),generation='hand2',cases=[n for n,_ in cases],warmups=2,repeats=10,gpu=False,neural_optimizer_updates=0,original_repo_modified=False,
        sdk_version=sdk.__version__,sdk_path=sdk.__file__,entry=str(Path(cli.__file__)),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},yaml_glove_config=cfg,
        note='current_yaml_glove_full applies the project YAML glove models to the same screw record only for timing; no claim that those models are correct calibration for this recording')
    atomic(out/'MANIFEST.json',manifest);atomic(out/'CONTROL.json',dict(stop_requested=False))
    import_end=time.perf_counter()
    original_solve,original_align=emf.solve_native,emf.align_camera
    current={}
    def solve(*args,**kwargs):
        start=time.perf_counter();result=original_solve(*args,**kwargs);current['native_both_hands_s']=time.perf_counter()-start
        current['native_frames']={side:len(result[0][side+'_hand_q']) for side in ['left','right']}
        current['model_info']=result[1]['models']
        np.savez_compressed(out/'native_last_DIAGNOSTIC.npz',**{side+'_q':result[0][side+'_hand_q'] for side in ['left','right']}) if profile_active else None
        return result
    def align(*args,**kwargs):
        start=time.perf_counter();result=original_align(*args,**kwargs);current['json_alignment_s']=time.perf_counter()-start;current['aligned_frames']=len(result[0]['frame_indices']);return result
    emf.solve_native,emf.align_camera=solve,align
    profile_active=False
    reports={}
    for name,extra in cases:
        output=out/(name+'.npz');samples=[];cold=None
        for rep in range(12):
            if json.loads((out/'CONTROL.json').read_text()).get('stop_requested'):raise InterruptedError('Stop requested')
            current={};begin=time.perf_counter()
            cli.main(['--episode',str(episode),'--out',str(output),'--generation','hand2',*extra])
            end=time.perf_counter()
            row=dict(total_s=end-begin,**{k:v for k,v in current.items() if k!='model_info'})
            row['remaining_config_selection_output_s']=row['total_s']-row['native_both_hands_s']-row['json_alignment_s']
            with np.load(output,allow_pickle=False) as saved:row['output_frames']=int(len(saved['right_hand_q']))
            if rep==0:cold=row
            if rep>=2:samples.append(row)
            atomic(out/'STATUS.json',dict(status='running',case=name,repeat=rep+1,completed=list(reports),neural_optimizer_updates=0))
            print('ENTRY_REPEAT',name,rep,json.dumps(row),flush=True)
        report=dict(samples=samples,first_invocation=cold,median_s={k:float(np.median([s[k] for s in samples])) for k in ['total_s','native_both_hands_s','json_alignment_s','remaining_config_selection_output_s']},native_frames=samples[0]['native_frames'],output_frames=samples[0]['output_frames'])
        with np.load(output,allow_pickle=False) as saved:
            if name=='builtin_full':
                for side in ['left','right']:
                    # Camera-aligned output can have a different frame count;
                    # score native q separately in the profiling diagnostic.
                    report[side+'_camera_output_shape']=list(saved[side+'_hand_q'].shape)
        reports[name]=report;atomic(out/'results.json',dict(complete=False,import_setup_s=import_end-process_begin,cases=reports));print('ENTRY_CASE_COMPLETE',name,json.dumps(report['median_s']),flush=True)
    profile_active=True;current={};profile=cProfile.Profile();profile.enable()
    cli.main(['--episode',str(episode),'--out',str(out/'profile_builtin.npz'),'--generation','hand2'])
    profile.disable();profile_active=False;profile.dump_stats(str(out/'entry_profile.pstats'))
    st=pstats.Stats(profile);stats=[]
    for (file,line,name),(cc,nc,tt,ct,callers) in st.stats.items():
        stats.append(dict(file=file,line=line,name=name,primitive_calls=cc,total_calls=nc,self_s=tt,cumulative_s=ct))
    stats.sort(key=lambda x:x['cumulative_s'],reverse=True);atomic(out/'profile_top.json',dict(scope='diagnostic instrumented run, not used in timings',functions=stats[:100]))
    with np.load(out/'native_last_DIAGNOSTIC.npz') as native:
        matches={}
        for side in ['left','right']:
            expected=np.load(b/'sdk/builtin'/f'trajectory_0_{side}.npz')['q_sdk_filtered'];actual=native[side+'_q'];same=actual.shape==expected.shape;matches[side]=dict(frames=len(actual),shape_matches=same,max_difference_rad=float(abs(actual-expected).max()) if same else None,bitwise_equal=bool(np.array_equal(actual,expected)))
    atomic(out/'results.json',dict(complete=True,import_setup_s=import_end-process_begin,cases=reports,native_builtin_vs_prior_sdk=matches,neural_optimizer_updates=0))
    atomic(out/'STATUS.json',dict(status='complete',complete=True,completed=list(reports),neural_optimizer_updates=0));print('ENTRY_ALL_COMPLETE',flush=True)


if __name__=='__main__':main()
