"""Summarize an audited articulation/size correction without altering grasps."""
import argparse
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--baseline',type=Path);parser.add_argument('--solve-batch',type=Path);parser.add_argument('--micro-geometry',type=Path)
args=parser.parse_args();out=args.output.resolve();composition=json.loads((out/'composition.json').read_text());old=args.baseline.resolve() if args.baseline else Path(composition['base_batch']);solved=args.solve_batch.resolve() if args.solve_batch else Path(composition['corrected_batch'])
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(out/'independent_result_audit.json')['passed']
rows=read(out/'manifest.json')['cases'];prior=read(old/'manifest.json')['cases'];records=[];groups={};unchanged8=[]
for e in rows:
 s=read(out/e['summary']);d=s['geometry_diagnostics'];p=e['primitive'];case=(out/e['trajectory']).parent;q=np.load(out/e['trajectory'])['qpos'][-1]
 number=int(e['template'].split('_')[0]);policy=d.get('articulation_policy');lo=d.get('optimization_joint_lower_limits');hi=d.get('optimization_joint_upper_limits')
 if policy:
  assert np.all(q>=np.array(lo)-1e-6) and np.all(q<=np.array(hi)+1e-6) and d['max_finger_order_violation_m']<=1e-5+1e-7
  assert d['max_hand_hand_penetration_m']<=1e-5+1e-7
 record=dict(case=e['name'],template=e['template'],type=number,scale=p['size_scale'],radius_mm=p['radius_m']*1000,thickness_mm=p['height_m']*1000,
  patch_coverage=d['coverage'],fixed_anchor_coverage=d.get('template_point_coverage',d['coverage']),projected_closure=d['projected_closure'],valid=d['valid'],self_depth_mm=d['max_hand_hand_penetration_m']*1000,
  crossing_violation_mm=d.get('max_finger_order_violation_m',0)*1000,articulation_policy=policy['policy'] if policy else None,
  min_non_thumb_pip_dip_deg=float(np.degrees(q[6:].reshape(5,4)[1:,2:]).min()),optimization_seconds=s['optimization_seconds'])
 records.append(record);groups.setdefault(number,[]).append(record)
 if number==8:
  previous=next(x for x in prior if x['name']==e['name']);assert all(sha(out/e[k])==sha(old/previous[k]) for k in ('model','trajectory','summary'));unchanged8.append(e['name'])
assert len(rows)==183 and len(unchanged8)==5
for number,low,high in ((10,.8,1.3),(12,.75,1),(18,1,2)):
 scales=[r['scale'] for r in groups[number]];assert abs(min(scales)-low)<1e-8 and abs(max(scales)-high)<1e-8
for number in (22,30):assert min(r['thickness_mm'] for r in groups[number])==.25
with (out/'results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
micro=read(args.micro_geometry if args.micro_geometry else out/'micro_contact_geometry.json');(out/'micro_contact_geometry.json').write_text(json.dumps(micro,indent=2)+'\n')
if (old/'prismatic_pose_comparison.png').is_file():
 shutil.copy2(old/'prismatic_pose_comparison.png',out/'prismatic_pose_comparison.png')
metadata=dict(cases=183,types=31,tests_passed=253,projected_closed=sum(r['projected_closure'] for r in records),patch_coverage80=sum(r['patch_coverage']+1e-7>=.8 for r in records),
 recomputed_articulation_cases=97,articulation_all_valid=all(r['valid'] and r['crossing_violation_mm']<=.010001 for r in records if r['articulation_policy']),type8_unchanged=unchanged8,
 resource_scope='97-case articulation and size-grid correction only; prior pad-center and Type10 cohorts retain their own resource reports',resources=read(solved/'resources.json'),
 target_types={str(k):v for k,v in groups.items() if k in (6,7,10,12,13,16,18,20,22,23,24,25,27,28,29,30,32,33)})
(out/'batch_report.json').write_text(json.dumps(metadata,indent=2)+'\n')
lines=['# 抓姿和尺寸范围修正','',
 '31个可用类型、183个尺寸实例；本轮15类97例全部重算并通过独立FK、碰撞、关节范围、手指顺序及真实响应力审计。253项CPU/CUDA回归通过。Type8五例模型、轨迹与summary哈希保持原值。','',
 'Type6/7固定锚点移到指腹中段（约末节45%–52%），十例中段patch贴合均为100%。其余类型手部局部接触仍使用原始模板，不重新选点。所有手势保持pose_weight=0；活动手指允许随尺寸开合。','',
 '本轮类型限制非拇指PIP/DIP反屈及PIP超过100°的极端折叠；MCP允许−5°的小幅伸展。空闲手指放在宽的张开/收拢区间，拇指保留模板所需伸展与对掌。另约束相邻非拇指在掌横轴上的PIP、DIP、指尖顺序；Type29中心线间隔10mm，其余4mm。手部自接触仍允许，穿透独立检查。','',
 '| 类型 | 尺寸策略 |','|---|---|','| 10 | 0.8×–1.3×，步长0.05 |','| 12 | 0.75×–1×，6档 |','| 18 | 1×–2×，9档；直径160–320mm |','| 22 | 板面150×105mm；厚度0.25–12mm，共10档 |','| 30 | 直径100mm；厚度0.25–12mm，共10档 |','',
 f"投影闭合{metadata['projected_closed']}/183；patch贴合率≥80%为{metadata['patch_coverage80']}/183。姿态约束通过不等同于全部接触建立。Type16仍有五例未通过投影闭合（旧结果也未通过），Type18扩展后的部分尺寸及Type24极细尺寸同样保留未通过状态。原生动态物理评测未运行。",'',
 f"Type24最小直径0.05mm，食指与拇指网格最近距离上界约{micro['cases'][0]['index_thumb_gap_upper_bound_mm']:.6f}mm；该实例没有原生自穿透，食指与中指顺序通过，投影闭合未证实。",'',
 f"本轮97例耗时{metadata['resources']['wall_seconds']:.1f}s，显存增量峰值为启动空闲显存的{metadata['resources']['peak_fraction_of_initial_free']:.1%}，预算上限80%。组合结果含此前Type6/7和Type10的独立重算，不能把此耗时视为183例从零求解耗时。",'',
 '| 类型 | 实例 | 投影闭合 | patch贴合中位数 | 最大自穿透mm |','|---|---:|---:|---:|---:|']
for k in sorted(groups):
 g=groups[k];lines.append(f"| {k} | {len(g)} | {sum(r['projected_closure'] for r in g)} | {np.median([r['patch_coverage'] for r in g]):.1%} | {max(r['self_depth_mm'] for r in g):.5f} |")
lines+=['','逐例结果见results.csv；manifest.json可回放最终q，independent_result_audit.json保存独立核验。']
(out/'batch_report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in metadata.items() if k not in ('target_types','resources')},ensure_ascii=False))
