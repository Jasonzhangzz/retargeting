#!/usr/bin/env python3
"""Summarize paired fixed-state/action solver quality and latency."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize(cases,output,closed_loop=None):
    output.mkdir(parents=True,exist_ok=True)
    summaries=[];comparisons=[]
    for case in cases:
        rows=json.loads((case/'results.json').read_text());config=json.loads((case/'config.json').read_text())
        lookup={(r['checkpoint'],r['action_noise_std_rad'],r['solver']):r for r in rows}
        for noise in sorted(set(r['action_noise_std_rad'] for r in rows)):
            for method in config['methods']:
                group=[r for r in rows if r['solver']==method and r['action_noise_std_rad']==noise]
                time=np.array([r['seconds'] for r in group])*1000
                reduction=np.array([r['relative_reduction'] for r in group])
                reference=[lookup[(r['checkpoint'],noise,'adamw8')] for r in group]
                sampling=[lookup[(r['checkpoint'],noise,'sampling64')] for r in group]
                result=dict(task=group[0]['task'],noise_std_rad=noise,solver=method,cases=len(group),
                    median_ms=float(np.median(time)),p95_ms=float(np.percentile(time,95)),
                    median_reduction_percent=float(100*np.median(reduction)),
                    zero_improvement_cases=int(np.sum(reduction<=1.e-6)),
                    beats_adamw8_cases=sum(r['value_after']<a['value_after']-1.e-5 for r,a in zip(group,reference)),
                    beats_sampling64_cases=sum(r['value_after']<a['value_after']-1.e-5 for r,a in zip(group,sampling)),
                    median_speedup_vs_adamw8=float(np.median([a['seconds']/r['seconds'] for r,a in zip(group,reference)])),
                    median_fixed_contact_preparation_ms=float(np.median([r['shared_preparation_seconds'] for r in group])*1000))
                summaries.append(result)
        with np.load(case/'solutions.npz') as arrays:
            solutions={key:arrays[key] for key in arrays.files}
        for i,r in enumerate(rows):
            ref_index=next(j for j,x in enumerate(rows) if x['checkpoint']==r['checkpoint'] and
                           x['action_noise_std_rad']==r['action_noise_std_rad'] and x['solver']=='adamw8')
            q,qr=solutions['next_qpos'][i],solutions['next_qpos'][ref_index]
            c,cr=solutions['controls'][i,:2],solutions['controls'][ref_index,:2]
            record=dict(task=r['task'],checkpoint=r['checkpoint'],noise_std_rad=r['action_noise_std_rad'],
                solver=r['solver'],next_root_difference_vs_adamw8_mm=float(np.linalg.norm(q[:3]-qr[:3])*1000),
                next_joint_rms_vs_adamw8_rad=float(np.sqrt(np.mean((q[7:36]-qr[7:36])**2))),
                first_controls_rms_vs_adamw8_rad=float(np.sqrt(np.mean((c-cr)**2))))
            if 'qpos_dense' in solutions:
                dense=solutions['qpos_dense'][i];ref=solutions['qpos_dense'][ref_index]
                record['predicted_root_difference_rms_vs_adamw8_mm']=float(np.sqrt(np.mean(np.sum((dense[:,:3]-ref[:,:3])**2,axis=1)))*1000)
            comparisons.append(record)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (output/'state_action_comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    with (output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names={'adamw1':'AdamW 1','adamw2':'AdamW 2','adamw3':'AdamW 3','adamw8':'AdamW 8','sampling64':'Sampling 64',
           'lm32':'Local LM 32','lm232':'Full LM 232','sampling64_lm32':'Sampling + LM 32',
           'sampling64_lm32x2':'Sampling + LM 32 x2','sampling64_fit_lm32':'Sampling + secant LM 32'}
    fig,axes=plt.subplots(1,len(cases),figsize=(12,4.8),squeeze=False,constrained_layout=True)
    for ax,case in zip(axes[0],cases):
        config=json.loads((case/'config.json').read_text());task=config['source_summary']['task']
        for j,method in enumerate(config['methods']):
            group=[x for x in summaries if x['task']==task and x['solver']==method]
            for s in group:
                ax.scatter(s['median_ms'],s['median_reduction_percent'],s=60,color=f'C{j}',
                    marker='o' if s['noise_std_rad']==0 else '^',
                    label=names[method] if s['noise_std_rad']==0 else None)
        ax.axvline(100,color='.6',ls='--',lw=1,label='100 ms action budget')
        ax.set(title='AMASS sprint' if task=='humanoid' else 'OMOMO move_largebox',
               xlabel='Median warm action solve (ms)',ylabel='Median actual cost reduction (%)')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Identical SPIDER states / exact dense actions; circles: nominal, triangles: 0.03 rad action noise',fontsize=10)
    fig.savefig(output/'solver_tradeoff.png',dpi=180)
    if closed_loop is not None:
        import mujoco
        root=Path(__file__).resolve().parents[1]
        audit=json.loads(closed_loop.read_text());fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
        for row in audit:
            case=root/'outputs/humanoid'/row['case'];cfg=json.loads((case/'config.json').read_text())
            if not cfg.get('previous_control_correction_retained_at_window_boundary'):
                continue
            col=1 if 'omomo' in cfg['reference'] else 0
            with np.load(case/'rollout.npz') as a:q=a['qpos'].copy();ids=np.r_[0,a['reference_frames']]
            with np.load(cfg['reference']) as a:ref=a['qpos'][ids]
            m=mujoco.MjModel.from_xml_path(cfg['scene']);d=mujoco.MjData(m);r=mujoco.MjData(m);torso=m.body('torso_link').id
            angles=[]
            for actual,target in zip(q,ref):
                d.qpos[:]=actual;r.qpos[:]=target;mujoco.mj_kinematics(m,d);mujoco.mj_kinematics(m,r)
                angles.append(np.degrees(np.arccos(np.clip(d.xmat[torso].reshape(3,3)[:,2]@r.xmat[torso].reshape(3,3)[:,2],-1,1))))
            seconds=np.arange(len(q))*cfg['ref_dt'];fit=cfg['feedforward_method']=='sample-fit'
            if cfg['feedforward_method'].startswith('adamw'):
                label='AdamW '+cfg['feedforward_method'][5:];color='#35936c'
            else:
                label='Sampling + fitted LM' if fit else 'Sampling + FD LM';color='#d18424' if fit else '#2878b5'
            axes[0,col].plot(seconds,q[:,2],label=label,color=color,lw=1.7)
            axes[1,col].plot(seconds,angles,label=label,color=color,lw=1.7)
            if fit:
                axes[0,col].plot(seconds,ref[:,2],ls='--',color='.35',label='Kinematic reference')
                axes[0,col].plot(seconds,ref[:,2]-.18,ls=':',color='#c44e52',label='Height audit boundary')
        for col,title in enumerate(('AMASS sprint','OMOMO move_largebox')):
            axes[0,col].set(title=title,xlabel='Executed time (s)',ylabel='Pelvis height (m)')
            axes[1,col].set(xlabel='Executed time (s)',ylabel='Torso tilt relative to reference (degrees)')
            axes[1,col].axhline(60,ls=':',color='#c44e52',label='Tilt audit boundary')
            for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Offline SPIDER feedforward + measured native feedback; one sampling/LM action block',fontsize=11)
        fig.savefig(output/'closed_loop.png',dpi=180)
    print(output/'summary.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cases',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--closed-loop-audit',type=Path)
    a=p.parse_args();summarize(a.cases,a.output,a.closed_loop_audit)
