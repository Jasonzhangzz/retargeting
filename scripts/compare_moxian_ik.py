"""Replay identical raw EMF through public vendor SDKs and local SLSQP.

Run with .venv-moxian-ik/bin/python. No serial access, calibration writes or
MANO correction. Each method owns a persistent solver for the whole clip.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def native_replay(a):
    from qnbot_sdk.glove._resources import builtin_skeleton_package_root
    from moxian_web.official_sdk import calibration_artifact
    from moxian_web.native_skeleton import NativeSkeleton
    from moxian_web.emf_slsqp import SLSQPSkeleton
    from moxian_web.kinematic_policy import build_kinematic_config
    from moxian_web.thumb_coordinates import build_thumb_coordinate_config
    package=Path(builtin_skeleton_package_root());artifact=calibration_artifact(a.calibration_home)
    if artifact is None:raise ValueError('Missing official calibration')
    rows=json.loads(a.input.read_text())['rows']
    with tempfile.TemporaryDirectory(prefix='ik-replay-') as temp:
        home=Path(temp);shutil.copytree(a.calibration_home/'glove',home/'glove')
        if a.native_worker=='cabi17':
            config=home/'runtime/original.yaml';config.parent.mkdir();config.write_bytes(artifact.read_bytes())
            engine=NativeSkeleton(package,config,side=a.side)
            policy={'mode':'official_native_C_ABI','calibration_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
        else:
            if a.side=='left':config,policy=build_thumb_coordinate_config(artifact.read_text(),package,home/'runtime')
            else:config,policy=build_kinematic_config(artifact.read_text(),package,home/'runtime',side=a.side)
            engine=SLSQPSkeleton(package,config,side=a.side);policy=engine.policy(policy)
        try:
            frames=[]
            for row in rows:
                began=time.perf_counter();result=engine.step(row)
                frames.append({'stage':row.get('stage'),'processing_ms':(time.perf_counter()-began)*1000,**result})
            a.output.write_text(json.dumps({'policy':policy,'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
                                           'frames':frames},ensure_ascii=False,allow_nan=False))
        finally:engine.close()


def compare_frames(left,right):
    import numpy as np
    if len(left)!=len(right):raise ValueError('Frame count differs')
    maximum_position=maximum_quaternion=maximum_joint=0.
    for a,b in zip(left,right):
        p,q=a['pose'],b['pose']
        if p['timestamp_ms']!=q['timestamp_ms'] or p['names']!=q['names']:raise ValueError('Not the same source frame/topology')
        maximum_position=max(maximum_position,float(np.max(np.abs(np.array(p['positions_local'])-q['positions_local']))))
        qa,qb=np.array(p['orientations_local_xyzw']),np.array(q['orientations_local_xyzw'])
        maximum_quaternion=max(maximum_quaternion,float(np.max(np.minimum(np.linalg.norm(qa-qb,axis=1),np.linalg.norm(qa+qb,axis=1)))))
        def joints(row):
            if row.get('joint_angles'):return row['joint_angles']['joints']
            return dict(zip(row['pose']['joint_names'],row['pose']['joint_positions_rad']))
        ja,jb=joints(a),joints(b)
        if ja.keys()!=jb.keys():raise ValueError('Joint command topology differs')
        maximum_joint=max(maximum_joint,max(abs(ja[n]-jb[n]) for n in ja))
    return dict(frames=len(left),max_position_coordinate_difference_m=maximum_position,
                max_quaternion_sign_invariant_difference=maximum_quaternion,max_joint_difference_rad=maximum_joint)


def metrics(frames):
    import numpy as np
    counters=Counter();flat={};steps=[];previous=None
    for row in frames:
        stage=row['stage'];count=counters[stage];counters[stage]+=1
        pose=row['pose'];positions=dict(zip(pose['names'],np.array(pose['positions_local'])))
        if stage in ('together','open') and count>=15:
            bends=[]
            for finger in ('index','middle','ring','little'):
                points=[positions[f'{finger}_{s}'] for s in ('proximal','intermediate','distal','tip')]
                v=np.diff(points,axis=0);v/=np.linalg.norm(v,axis=1)[:,None]
                bends.extend(np.rad2deg(np.arccos(np.clip((v[:-1]*v[1:]).sum(1),-1,1))).tolist())
            flat.setdefault(stage,[]).extend(bends)
        # Different staged holds are separated by user movement that was not
        # recorded; never count those transitions as consecutive-frame jumps.
        joints=row.get('joint_angles',{});joints=joints.get('joints') if joints else None
        if joints is None:joints=dict(zip(pose['joint_names'],pose['joint_positions_rad']))
        if previous and previous[0]==stage:
            steps.append(max(abs(joints[n]-previous[1][n]) for n in joints))
        previous=(stage,joints)
    return {'frames':len(frames),'stage_counts':dict(counters),
            'flat_holdout_four_finger_IP_bend_deg':{stage:{'mean':float(np.mean(v)),'max':float(np.max(v)),'samples':len(v)} for stage,v in flat.items()},
            'within_hold_max_joint_step_deg':float(np.rad2deg(max(steps,default=0))),
            'processing_ms':{'median':float(np.median([r['processing_ms'] for r in frames])),
                             'p95':float(np.percentile([r['processing_ms'] for r in frames],95))}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--calibration-home',type=Path,required=True)
    ap.add_argument('--side',choices=('left','right'),required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--native-worker',choices=('cabi17','slsqp'))
    a=ap.parse_args()
    if a.native_worker:native_replay(a);return
    a.output.mkdir(parents=True,exist_ok=True)
    for name,environment in (('sdk17','.venv-moxian-ik'),('sdk18','.venv-moxian-official')):
        cmd=[str(ROOT/environment/'bin/python'),'-I',str(ROOT/'moxian_web/official_worker.py'),
             '--home',str(a.output/('scratch-'+name)),'--calibration-home',str(a.calibration_home.resolve()),
             '--side',a.side,'--replay',str(a.input.resolve()),'--output',str((a.output/(name+'.json')).resolve())]
        subprocess.run(cmd,check=True)
    for name in ('cabi17','slsqp'):
        subprocess.run([str(ROOT/'.venv-moxian-ik/bin/python'),'-I',str(Path(__file__).resolve()),
            '--native-worker',name,'--input',str(a.input.resolve()),'--calibration-home',str(a.calibration_home.resolve()),
            '--side',a.side,'--output',str((a.output/(name+'.json')).resolve())],check=True)
    results={name:json.loads((a.output/(name+'.json')).read_text()) for name in ('sdk17','sdk18','cabi17','slsqp')}
    report={'input':str(a.input.resolve()),'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'side':a.side,
        'public17_vs_native17':compare_frames(results['sdk17']['frames'],results['cabi17']['frames']),
        'public17_vs_public18':compare_frames(results['sdk17']['frames'],results['sdk18']['frames']),
        'methods':{name:{'policy':r['policy'],'metrics':metrics(r['frames'])} for name,r in results.items()},
        'limitations':['Saved eight-action holds, not synchronized external ground truth.',
                      'Flat holdout means last 15 frames of each 30-frame flat hold; thumb excluded.',
                      'SLSQP uses a flat mount fit and extension limits; this is a pipeline comparison, not solver-only ablation.',
                      'Replay timings include public SDK dispatch overhead and are not physical tracking accuracy.',
                      'No MANO conversion, tactile correction, serial I/O or hardware commands.']}
    integrity=a.output/'wheel-integrity.json'
    if integrity.is_file():report['sdk17_wheel_integrity']=json.loads(integrity.read_text())
    (a.output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# Moxian 官方 IK 对照核验','',f'输入：同一组 {len(results["sdk17"]["frames"])} 帧原始 EMF，手侧 {a.side}。',
        f'输入 SHA256：`{report["input_sha256"]}`。','',
        '官方 Python SDK 使用原始官方八动作标定的字节一致副本，直接读取 Skeleton.pose() 与 joint_angles()。',
        '右手旧 C ABI 调用的也是官方原生库；左手旧链路改过拇指坐标／限位，不能据右手实验推断左手也完全原版。','',
        'MANO 骨架绑定、姿态零位、尺寸拟合和触觉修正属于本地链路，不包含在此官方骨架对照中。','',
        '## 同帧输出差异','',
        '| 比较 | 最大坐标差 (m) | 最大关节差 (rad) | 最大四元数差（符号不变） |','|---|---:|---:|---:|']
    for label,key in [('官方 SDK 1.7.0 / 原生 C ABI 1.7.0','public17_vs_native17'),('官方 SDK 1.7.0 / 官方 SDK 1.8.1','public17_vs_public18')]:
        d=report[key];lines.append(f'| {label} | {d["max_position_coordinate_difference_m"]:.12g} | {d["max_joint_difference_rad"]:.12g} | {d["max_quaternion_sign_invariant_difference"]:.12g} |')
    lines+=['','## 平摊留出帧的骨段弯曲','',
        '每个平摊动作后 15 帧，四指 PIP/DIP 相邻骨段夹角；拇指不计。0° 表示几何伸直，非关节指令零位。','',
        '| 方法 | 并拢 mean / max (°) | 张开 mean / max (°) | 回放 median / P95 (ms) |', '|---|---:|---:|---:|']
    for name,r in report['methods'].items():
        m=r['metrics'];f=m['flat_holdout_four_finger_IP_bend_deg'];t=m['processing_ms']
        fmt=lambda key:f'{f[key]["mean"]:.2f} / {f[key]["max"]:.2f}' if key in f else 'N/A'
        lines.append(f'| {name} | {fmt("together")} | {fmt("open")} | {t["median"]:.2f} / {t["p95"]:.2f} |')
    lines+=['','## 判读范围','',
        'SLSQP 同时包含固定安装配准与伸直限位修正，此表比较完整链路，不是只替换 solver 的消融。',
        'SDK 版本结论仅适用于本次相同标定、资源和输入。1.8.1 沿用原标定声明的资源根；资源哈希和原生库哈希见 comparison.json。',
        '平摊帧的“应伸直”来自动作语义，没有外部光学真值；不能由此量化所有姿态的实际精度，也不能把 MANO 的额外误差归咎于官方。','',
        '官方 1.7.0 wheel 的 124 个 skeleton 资源文件与本机逐字节一致，核验记录见 wheel-integrity.json。' if integrity.is_file() else '',
        '','## 公开接口','',
        '官方公开文档提供 Dynamic OpenXR 25 节点骨架与关节命令；截至本次核验未找到公开 EMF→MANO 接口。',
        '- [官方模型说明](https://docs.qnbot.com/zh-CN/docs/vetra-glove-pro/models/retargeting/latest)',
        '- [官方 Python SDK 示例](https://github.com/qnbot-technology/qnbot-sdk/tree/main/examples/glove)',
        '', '网页“官方 SDK 骨架（对照）”使用独立 1.8.1 公共 SDK worker；当前 SLSQP 仍驱动 MANO／Wuji2。',
        '录制分别保存 skeleton 与 skeleton_official；只有 timestamp_ms 相等时才属于同帧比较。','']
    (a.output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:report[k] for k in ('public17_vs_native17','public17_vs_public18')},indent=2))


if __name__=='__main__':main()
