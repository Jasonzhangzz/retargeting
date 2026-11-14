#!/usr/bin/env python3
"""Paired contact-anchor ablation on the already frozen HO3D/Wuji2 cohort."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('output/ho3d_mpc_dexforge_20261112'))
    p.add_argument('--output', type=Path, default=Path('output/ho3d_mpc_contact_20261112'))
    p.add_argument('--contact-weight', type=float, default=1.)
    p.add_argument('--contact-scale', type=float, default=.01)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'benchmark.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((a.source/'manifest.json').read_text())
    cases = [c for c in manifest['cases'] if (Path(c['path'])/'mpc/evaluation.json').exists()]
    protocol = dict(seed=42, samples=128, iterations=6, horizon_seconds=.3, execute_seconds=.03,
                    physics_hz=300, duration_seconds=2.1, contact_weight=a.contact_weight,
                    contact_scale_m=a.contact_scale, reference='ContactAware object-local anchors and hand-local material points',
                    loss='weight * weighted_mean(||hand_world - actual_object_anchor_world||^2 / scale^2)',
                    baseline='Archived original MPC; no baseline outputs overwritten', initial_state='Exact same initial.npz',
                    selection='All five previously completed dynamic trials; no new selection',
                    limitations='One seed, five short clips, MC1 used for development; original four CA failures not retried')
    write(root/'protocol.json', protocol)
    write(root/'hardware.json', dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
          gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
          packages={x: importlib.metadata.version(x) for x in ('numpy', 'torch', 'mujoco', 'mujoco-warp', 'warp-lang', 'scipy')}))
    files = ['retarget/env/dynamic/contact_reference.py', 'retarget/env/dynamic/native_mpc.py',
             'scripts/compare_ho3d_mpc_dexforge.py', 'scripts/test_ho3d_mpc_contacts.py']
    write(root/'implementation.json', {f: digest(f) for f in files})
    state = dict(status='running', pid=os.getpid(), total_cases=len(cases), results=[], started_at=time.time())
    write(root/'status.json', state)
    command = [sys.executable, '-u', 'scripts/compare_ho3d_mpc_dexforge.py']
    for item in cases:
        case = Path(item['path'])
        out = case/'mpc_contact'
        out.mkdir(exist_ok=True)
        receipt = dict(object=item['object'], case=str(case), initial_state_sha256=digest(case/'initial.npz'),
                       scene_sha256=digest(case/'scene.xml'), contact_guidance_sha256=digest(case/'contactaware/contact_guidance.npz'),
                       baseline_rollout_sha256=digest(case/'mpc/rollout.npz'))
        state['current'] = receipt
        write(root/'status.json', state)
        timing = out/'timing.json'
        if timing.exists():
            old = json.loads(timing.read_text())
            if old.get('contact_weight') != a.contact_weight or old.get('contact_scale_m') != a.contact_scale:
                raise ValueError(f'Refusing to overwrite a different contact experiment at {out}')
        if not (out/'rollout.npz').exists():
            with (out/'run.log').open('w') as log:
                subprocess.run(command + ['mpc', '--case', str(case), '--contact-points',
                               '--contact-weight', str(a.contact_weight), '--contact-scale', str(a.contact_scale)],
                               stdout=log, stderr=log, check=True, timeout=900)
        with (out/'evaluate.log').open('w') as log:
            subprocess.run(command + ['evaluate', '--case', str(case), '--method', 'mpc_contact'],
                           stdout=log, stderr=log, check=True, timeout=180)
        import numpy as np
        import mujoco
        from retarget.env.dynamic.contact_reference import load_contact_reference, measure_contact_tracking
        from retarget.env.dynamic.comparison import load_motion
        from retarget.viewer.dynamic import export_viewer
        model = mujoco.MjModel.from_xml_path(str(case/'scene.xml'))
        contact = load_contact_reference(case, model)
        receipt['methods'] = {}
        trajectories = {}
        for method in ('mpc', 'mpc_contact'):
            with np.load(case/method/'rollout.npz') as z:
                trajectories[method] = z['qpos'].copy()
            receipt['methods'][method] = dict(
                evaluation=json.loads((case/method/'evaluation.json').read_text()),
                timing=json.loads((case/method/'timing.json').read_text()),
                contact_tracking=measure_contact_tracking(model, trajectories[method], contact),
            )
        with np.load(case/'initial.npz') as z:
            receipt['initial_contact_tracking'] = measure_contact_tracking(model, z['qpos'][None], contact, start_frame=0)
        if receipt['initial_state_sha256'] != digest(case/'initial.npz') or receipt['baseline_rollout_sha256'] != digest(case/'mpc/rollout.npz'):
            raise AssertionError('Shared initialization or archived baseline changed')
        viewer = root/case.name/'viewer'
        viewer.mkdir(parents=True, exist_ok=True)
        export_viewer(model, load_motion(case/'motion.npz'), trajectories['mpc_contact'][9::10],
                      trajectories['mpc'][9::10], np.load(case/'reference.npy')[1:], viewer)
        html = (viewer/'index.html').read_text().replace('DIAL-MPC', 'MPC 接触点对照').replace('紫：优化结果', '紫：加入参考接触点').replace('灰：IK 控制物理回放', '灰：原 MPC').replace('id="base" type="checkbox"', 'id="base" type="checkbox" checked')
        (viewer/'index.html').write_text(html)
        write(viewer/'result.json', receipt)
        write(root/case.name/'result.json', receipt)
        state['results'].append(receipt)
        state.pop('current', None)
        write(root/'status.json', state)
        print(json.dumps(dict(object=case.name, results={k: v['evaluation']['accepted'] for k, v in receipt['methods'].items()})), flush=True)
    state.update(status='complete', finished_at=time.time())
    write(root/'status.json', state)
    summary = {}
    for method in ('mpc', 'mpc_contact'):
        values = [r['methods'][method] for r in state['results']]
        count = sum(v['evaluation']['accepted'] for v in values)
        summary[method] = dict(successes=count, trials=len(values), success_rate=count/len(values),
                              mean_loop_seconds=float(np.mean([v['timing']['optimization_and_execution_seconds'] for v in values])),
                              mean_plan_seconds=float(np.mean([np.mean(v['timing']['plan_seconds']) for v in values])))
    write(root/'summary.json', dict(protocol=protocol, methods=summary, results=state['results']))
    rows = ['| 物体 | 原 MPC | 加接触点 | 位置 RMSE mm：前 → 后 | 接触点 RMSE mm：前 → 后 | 耗时 s：前 → 后 |', '|---|---|---|---|---|---|']
    for r in state['results']:
        old, new = (r['methods'][m] for m in ('mpc', 'mpc_contact'))
        rows.append(f"| {r['object']} | {'通过' if old['evaluation']['accepted'] else '失败'} | {'通过' if new['evaluation']['accepted'] else '失败'} | {old['evaluation']['object_position_rmse_mm']:.2f} → {new['evaluation']['object_position_rmse_mm']:.2f} | {old['contact_tracking']['weighted_rmse_mm']:.2f} → {new['contact_tracking']['weighted_rmse_mm']:.2f} | {old['timing']['optimization_and_execution_seconds']:.2f} → {new['timing']['optimization_and_execution_seconds']:.2f} |")
    report = '# MPC 显式参考接触点对照\n\n' + '\n'.join(rows)
    report += f"\n\n成功率：{summary['mpc']['successes']}/5 → {summary['mpc_contact']['successes']}/5。平均优化与执行耗时：{summary['mpc']['mean_loop_seconds']:.2f} → {summary['mpc_contact']['mean_loop_seconds']:.2f} s／2.1 s 轨迹。\n"
    report += '\n接触目标：手指局部材料点与实际预测物体上的局部锚点距离平方，按有效置信度加权平均；长度尺度 10 mm、权重 1。保留原物体跟踪、手姿、控制、接触力和穿透目标。每步增加 MJWarp GPU kinematics，避免读取积分前的旧变换。未添加参考法向约束，参考文件不包含该字段。\n'
    report += '\n样本、seed=42、128 候选、6 次迭代、0.3 s 预测窗口、0.03 s 重规划和初始状态完全相同。旧结果为归档对照，耗时不是同一时刻的并发测量。判据保持最大位置误差 <150 mm、最大旋转误差 <40°、平均最大穿透 ≤5 mm、双指载荷中位数 ≥1 N、单指 p95 载荷 ≤100 N。\n'
    report += '\n这是五段各 2.1 s 的小样本对照，包含开发用 MC1，不能代表大规模泛化成功率。四段原 ContactAware 前端失败未重跑。仍使用原凸包碰撞、假设质量／摩擦／重力方向、无支撑桌面；全部物理执行与预测均为 MJWarp。未按此次结果调参。\n'
    report += '\n逐段失败原因：\n'
    for r in state['results']:
        report += f"\n- {r['object']}：{', '.join(r['methods']['mpc_contact']['evaluation']['failure_reasons']) or '通过'}。"
    (root/'REPORT.md').write_text(report+'\n')
    links = ''.join(f'<li><a href="{r["object"]}/viewer/">{r["object"]}</a></li>' for r in state['results'])
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MPC 参考接触点对照</title><style>body{font:17px system-ui;max-width:1000px;margin:40px auto;background:#15201e;color:#dfe8e1}a{color:#8ed2bb}pre{white-space:pre-wrap}</style><h1>MPC 参考接触点对照</h1><p>紫：加入参考接触点；灰：原 MPC；青：MANO；橙：参考物体。</p><ul>'+links+'</ul><p><a href="REPORT.md">报告</a> · <a href="summary.json">详细指标</a></p><pre>'+report.replace('&','&amp;').replace('<','&lt;')+'</pre>')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # A failed subprocess must never leave a stale "running" receipt.
        args = sys.argv[1:]
        output = Path(args[args.index('--output') + 1]) if '--output' in args else Path('output/ho3d_mpc_contact_20261112')
        status = output/'status.json'
        if status.exists():
            value = json.loads(status.read_text())
            if value.get('pid') == os.getpid():
                value.update(status='failed', error=repr(error), finished_at=time.time())
                write(status, value)
        raise
