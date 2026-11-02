#!/usr/bin/env python3
"""Render paired reports and attach phase/acceptance metadata for replay."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from retarget.common.results import write_json


def render(batch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    batch=Path(batch);r=json.loads((batch/'report.json').read_text());types=r['by_type']
    rows=json.loads((batch/'summary.json').read_text())['jobs']
    manifest=json.loads((batch/'manifest.json').read_text());by_id={row['id']:row for row in rows}
    for case in manifest['cases']:
        case['task_success']=by_id[case['case_id']]['task_success']
        case['evaluation_protocol']=r['protocol']
        if case['phase']=='refinement':
            case.update(stage='接触建立',time_kind='optimization_iteration',initial_index=-1)
        else:
            case.update(stage='GraspADMM评测')
    write_json(batch/'manifest.json',manifest)
    x=np.arange(len(types));fig,ax=plt.subplots(figsize=(17,4.5),layout='constrained')
    for shift,key,label,color in [(-.25,'baseline_success_rate','ADMM','#657a96'),
        (0.,'refined_success_rate','ADMM + local MPPI (raw)','#198a69'),
        (.25,'validated_fallback_success_rate','With native-validation fallback','#e6a442')]:
        ax.bar(x+shift,[t[key] for t in types],width=.24,label=label,color=color)
    ax.set_xticks(x,[str(t['type_id']) for t in types]);ax.set_ylim(0,1);ax.set_ylabel('Success rate')
    ax.set_xlabel('Grasp type');fig.legend(loc='outside lower center',ncols=3,fontsize=9)
    ax.set_title(f"Same {r['trials']} final ADMM grasps / exact original six-direction acceptance")
    fig.savefig(batch/'type_comparison.png',dpi=170);plt.close(fig)
    width=max(t['size_index'] for t in r['by_size'])+1;matrix=np.full((len(types),width),np.nan)
    lookup={t['type_id']:i for i,t in enumerate(types)}
    fig,ax=plt.subplots(figsize=(14,12),layout='constrained')
    for t in r['by_size']:matrix[lookup[t['type_id']],t['size_index']]=t['percentage_point_change']
    colors=plt.colormaps['PiYG'].copy();colors.set_bad('#eeeeee')
    image=ax.imshow(matrix,vmin=-100,vmax=100,cmap=colors,aspect='auto')
    for t in r['by_size']:
        p=t['primitive'];label=f"{p['size_scale']:g}x\n{t['refined_successes']}/{t['trials']} ({t['refined_successes']-t['baseline_successes']:+d})"
        ax.text(t['size_index'],lookup[t['type_id']],label,ha='center',va='center',fontsize=7.5)
    ax.set_yticks(range(len(types)),[str(t['type_id']) for t in types]);ax.set_xticks(range(width),range(1,width+1))
    ax.set_xlabel('Size position in each type grid; labels show actual multiplier, raw MPPI passes and change')
    ax.set_ylabel('Grasp type');ax.set_title('Paired local MPPI refinement: success change by actual size')
    fig.colorbar(image,ax=ax,label='Change in percentage points',shrink=.65)
    fig.savefig(batch/'size_comparison.png',dpi=170);plt.close(fig)
    timings={}
    for name in ['optimization_seconds','geometry_setup_seconds','evaluation_seconds','wall_seconds']:
        values=[t[name] for t in rows if t.get(name) is not None]
        if values:timings[name]=dict(count=len(values),sum=sum(values),median=float(np.median(values)),
            p95=float(np.percentile(values,95)),min=min(values),max=max(values))
    timings['total_sweep_wall_seconds']=r['wall_seconds']
    timings['scope']='One GPU planning worker, one persistent pinned CPU evaluator, sequential cases; source ADMM solve excluded'
    timings['budget']=dict(population=r['settings']['population'],iterations=r['settings']['iterations'],
        stages=r['settings']['stages'],samples_per_trial=r['settings']['population']*r['settings']['iterations']*r['settings']['stages'])
    benchmark=ROOT/'output/graspadmm_performance/benchmark.json'
    if benchmark.exists():
        timings['previous_same_input_native_admm_benchmark']=json.loads(benchmark.read_text())['comparison']
        timings['comparison_caution']='Original ADMM core is CPU; MPPI geometry scoring is RTX 4090 GPU. Costs and budgets differ. No claim of equivalent work or optimizer speedup.'
    write_json(batch/'timings.json',timings)
    table=['| Type | Trials | ADMM | MPPI raw | Change | With validated fallback |',
        '|---|---:|---:|---:|---:|---:|']
    for t in types:
        table.append(f"| {t['type_id']} | {t['trials']} | {t['baseline_successes']} | {t['refined_successes']} | {t['refined_successes']-t['baseline_successes']:+d} | {t['validated_fallback_successes']} |")
    (batch/'comparison.md').write_text('\n'.join([
        '# Paired ADMM → MPPI evaluation','',
        f"Completed {r['trials']} paired trials. Native protocol: `{r['protocol']}`.",'',
        f"ADMM: {r['baseline_successes']}/{r['trials']} ({100*r['baseline_success_rate']:.2f}%). "
        f"Raw local MPPI: {r['refined_successes']}/{r['trials']} ({100*r['refined_success_rate']:.2f}%), "
        f"change {r['percentage_point_change']:+.2f} percentage points.",'',
        f"Transitions: {r['transitions']}. The native-validation fallback policy is separate: "
        f"{r['validated_fallback_successes']}/{r['trials']} ({100*r['validated_fallback_success_rate']:.2f}%).",'',
        'The local MPPI objective differs from ADMM and from historical MPPI guidance; the final acceptance is identical. '
        'Every saved ADMM final pose and actual mesh is reused, without native-evaluation-guided sample selection.',
        '',*table,'','See `timings.json`, `by_size.csv`, `trials.csv` and `audit.json`.','']))
    return timings


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=Path,required=True)
    print(json.dumps(render(parser.parse_args().batch)))
