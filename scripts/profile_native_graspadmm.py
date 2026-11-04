#!/usr/bin/env python3
"""One independent, explicitly profiled original operation; excluded from timing."""
from pathlib import Path
import argparse,collections,copy,inspect,json,sys,time
import native_graspadmm_retest as runner


def run(root,source):
    import torch
    torch.set_default_device('cpu')
    cfg=runner.configuration(source,root,'grasp_admm',root/'phase_diagnostic',1)
    path=json.loads((root/'timing_inputs.json').read_text())[0]['path']
    operation=source.core._single_grasp.__wrapped__
    text,firstline=inspect.getsourcelines(operation)
    pregrasp_line=firstline+next(i for i,s in enumerate(text) if '# Generate pregrasp qpos list' in s)
    totals=collections.defaultdict(float);counts=collections.Counter();stack=[];last=time.perf_counter()
    def classify(frame):
        name=frame.f_code.co_name;file=Path(frame.f_code.co_filename).name
        if file=='qp_single.py' and name=='solve':return 'qp'
        if name=='forward' and isinstance(frame.f_locals.get('self'),source.core.GraspFilter):return 'filter'
        if file=='mujoco_env.py' and name=='__init__':return 'model_scene_prepare'
        if name in ('load_obj_trimesh','create_hand_trimesh'):return 'model_scene_prepare'
        if name in ('project_to_hand','get_closest_triangles','point_to_triangle_simple'):return 'surface_search'
        if name=='save' and 'numpy' in frame.f_code.co_filename:return 'file_write'
        if file=='mujoco_env.py' and name in ('apply_contact_forces_no_normal','apply_contact_forces','step_sim','get_squeeze_qpos'):return 'mujoco_hand_step'
        caller=frame.f_back
        if caller is not None and caller.f_code is operation.__code__ and caller.f_lineno>=pregrasp_line:
            return 'pregrasp'
        return stack[-1][1] if stack else 'other'
    def profile(frame,event,arg):
        nonlocal last
        now=time.perf_counter()
        if stack:totals[stack[-1][1]]+=now-last
        last=now
        if event=='call':
            category=classify(frame);stack.append((frame,category));counts[category]+=1
        elif event=='return':
            if stack and stack[-1][0] is frame:stack.pop()
    start=time.perf_counter();previous=sys.getprofile()
    try:
        sys.setprofile(profile)
        source.core._single_grasp((path,cfg))
    finally:sys.setprofile(previous)
    runner.write(root/'phase_diagnostic.json',dict(input=path,wall_seconds=time.perf_counter()-start,
        exclusive_seconds=dict(totals),call_counts=dict(counts),instrumented=True,
        note='Independent Python return/call profiling; includes profiler overhead, excludes native main timing. Nested filter QP is counted under QP.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    a=p.parse_args();s=runner.load_upstream(a.upstream);runner.seed();run(a.root,s)
