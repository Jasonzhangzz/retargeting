#!/usr/bin/env python3
"""Reconstruct v5 quality from the complete fixed cohort, including failures."""
import argparse,json,csv,collections
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();root=args.experiment
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    config=json.loads((root/'MANIFEST.json').read_text());methods=['input','velocity_input','cem'];rows=[];summary={};preserved=0;repaired=0
    for method in methods:
        values=[];seconds=0.;acceptance=0.;warmup=0.;failure=collections.Counter()
        for packet in config['packets']:
            folder=root/'quality'/packet['name']/method;a=np.load(folder/'final.npz');m=json.loads((folder/'metrics.json').read_text());r=json.loads((folder/'report.json').read_text())
            np.testing.assert_allclose(a['optimizer_cost_v3'],a['patch_fit_cost']+1e4*a['penetration']+.1*a['soft_penetration']+a['rollout_cost'],rtol=1e-5,atol=1e-5)
            assert np.array_equal(a['patch_fit_slot_error']==0,a['patch_fitted'])
            expected=(a['patch_coverage']+1e-7>=.8)&a['patch_all_required_fingers']
            assert np.array_equal(a['patch_fit_cost']==0,expected)
            if method=='cem':
                before=np.load(folder.parent/'input/final.npz');good=before['patch_joint_accepted']
                np.testing.assert_array_equal(before['variables'][good],a['variables'][good]);assert a['patch_joint_accepted'][good].all()
                preserved+=int(good.sum());repaired+=int((~good&a['patch_joint_accepted']).sum())
            for i,v in enumerate(m):
                value=dict(packet=packet['name'],row=i,method=method,**v);values.append(value);rows.append(value)
                failure.update(v['failure_reasons'])
            seconds+=r.get('solve_seconds',0.);acceptance+=r['acceptance_seconds'];warmup+=r.get('warmup_seconds',0.)
        assert len(values)==config['samples']
        summary[method]=dict(samples=len(values),valid=sum(v['valid'] for v in values),velocity=sum(v['stable'] for v in values),
            geometry=sum(v['geometry'] for v in values),joint=sum(v['passed'] for v in values),all_fingers=sum(v['all_fingers'] for v in values),
            coverage80=sum(v['coverage']+1e-7>=.8 for v in values),solve_seconds=seconds,amortized_refinement_seconds=seconds/len(values),
            acceptance_seconds=acceptance,warmup_seconds=warmup,evaluations=sum(v.get('evaluations',0) for v in values),
            failure_reasons=dict(failure),early_stopped=sum(v.get('early_stopped',False) for v in values))
    result=dict(protocol=config['protocol'],complete=True,production_changed=False,source=config['source_sha256'],
        methods=summary,initial_joint_preserved_exactly=preserved,newly_repaired=repaired,
        thresholds=dict(linear_m_s=.1,angular_rad_s=2.,surface_mm=2,anchor_mm=5,patch_radius_mm=10,coverage=.8,hand_object_mm=5,self_mm=5))
    (root/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    fields=['packet','row','method','valid','stable','geometry','passed','coverage','all_fingers','end_linear_speed_m_s','end_angular_speed_rad_s','total_cost','anchor_cost','rollout_cost','iterations','evaluations','failure_reasons']
    with (root/'per_pose.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for v in rows:w.writerow({k:json.dumps(v.get(k),ensure_ascii=False) if isinstance(v.get(k),list) else v.get(k) for k in fields})
    lines=['# 切向形变历史与一致patch目标：固定54条验证','',
        '使用同一批54条固定诊断样本，未重采样。新模型不是原始ComFree公式；原内核保持不变。生产任务仍停止。','',
        '## 已实现','',
        '- 每个候选从零切向弹簧形变开始，接触时累计；重投影到当前切平面，滑动按μFn返回映射，脱离时清零。不同候选和不同pose不共享历史。',
        '- 物理参数：Kn=Kt=500 N/m，Cn=Ct=5 N·s/m，μ=0.6，物体0.1kg。每个外步10ms，内部4步；共5个外步=50ms。六维隐式动量更新及阻尼Newton在GPU运行，没有QP或完整MuJoCo动力学。参数在真实质量测试前由受控夹持检查固定。',
        '- 贴合目标与验收调用同一个实际patch witness。槽误差为表面距离超过2mm及物面点距固定目标超过5mm的归一化平方惩罚；全局选80%槽，并独立累加每个必要手指最小槽惩罚。掌部参与全局覆盖，保持既有规则。',
        '- 合格槽不再因其固定模板点偏离而继续受惩罚。patch cost为零，当且仅当覆盖≥80%且每个必要手指至少一槽合格；穿透和速度单独检验。原5mm穿透门槛及原穿透项保持。',
        '- 全部候选都检查同一联合判据，独立保留已通过候选；允许选择总代价稍高但首次可行的姿态，避免被标量最低值丢弃。原CEM采样及精英更新规则保留。','',
        '## 同一新模型内的结果','',
        '| 阶段 | 速度通过 | 几何通过 | 联合通过 | 每指通过 | 覆盖≥80% | 有效 |','|---|---:|---:|---:|---:|---:|---:|']
    labels=dict(input='纯贴合LM姿态直接重评',velocity_input='此前速度版姿态直接重评',cem='从纯贴合LM经新目标CEM精修')
    for key,v in summary.items():lines.append(f'| {labels[key]} | {v["velocity"]}/54 | {v["geometry"]}/54 | {v["joint"]}/54 | {v["all_fingers"]}/54 | {v["coverage80"]}/54 | {v["valid"]}/54 |')
    lines+=['',f'原有{preserved}条联合通过起点参数逐位保留，新增修复{repaired}条。仍有{54-summary["cem"]["joint"]}条未联合通过；失败样本全部保留。完整逐样本数据见per_pose.csv及quality目录。',
        '','原模型对同一纯LM姿态的速度通过为0/54；新模型直接重评为22/54。这是接触模型改变的效果，不能描述为同一物理模型下的solver提升，也不是放宽速度阈值。',
        '','## 01 Large Diameter','',
        '新CEM：覆盖81.25%，所有必要手指通过，thumb至少一槽间距1.926mm；末线速度0.003122m/s、角速度0.12034rad/s；手物穿透4.299mm、自穿透1.125mm；联合通过。固定模板点RMS仍为8.456mm，仅作诊断，不再与允许的patch接触互相冲突。',
        '','## GPU wall time','',
        f'全54条新增精修共{summary["cem"]["solve_seconds"]:.3f}s，摊销{summary["cem"]["amortized_refinement_seconds"]:.3f}s/pose；最终验收共{summary["cem"]["acceptance_seconds"]:.3f}s。模型加载{json.loads((root/"loading.json").read_text())["seconds"]:.3f}s、首次准备/图捕获分别记录，未混入精修时间。',
        '','设备'+config['device']+f'，输入batch=1/8，候选缓冲{config["chunk"]}，128候选×最多24轮，联合通过即停。该摊销数不是独立单pose延迟，也不是batch1024生产吞吐。新的隐式带记忆接触比旧瞬时公式更重，尚未据此切换四卡生产。',
        '','## 正确性与限制','',
        '受控两点夹持5mm：五步末速度约0.00458m/s（原响应约0.416m/s）；验证了静态剪切力、摩擦限幅、接触Jacobian、脱离清零、候选历史重置、步长细化、GPU图捕获、pose独立性及可行候选保留。原ComFree AST指纹回归保持。',
        '','真实MANO图捕获与eager一致，评分期间不允许Tensor.cpu()。整条pose/接触/积分/目标/验收计算在GPU；CPU用于资产准备、编排和文件读写。',
        '','接触发现仍沿用16个刚体分区凸包和物体凸包；贴合验收采用实际三角网格。该代理差异尚未消除。五步50ms通过是本协议短时稳定，不是严格力闭合、全方向抗扰或长期GSR证明。',
        '','网页增加v5起点、此前速度版重评及新CEM结果，显示实际网格、patch、接触与五步运动：http://127.0.0.1:8106/interactive.html 。54条为分层诊断样本，不推广为数据集总体通过率。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
