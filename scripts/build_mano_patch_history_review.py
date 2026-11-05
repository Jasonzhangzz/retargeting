#!/usr/bin/env python3
"""Display only current v5 results, with the same patch cost and acceptance."""
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.mano_patch_viewer import read_page,variant
from retarget.env.grasp.mano_solver import ManoDecoder

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--previews',type=Path,default=Path('output/mano_comfree_previews_20261105'))
    p.add_argument('--assets',type=Path,default=Path('output/mano_patch_velocity_v4_20261105/assets'));args=p.parse_args()
    target=args.previews/'interactive.html';data,previous=read_page(target)
    staging=args.previews/'interactive.next.html'
    mapping=json.loads((args.assets/'fixed_contact_mapping.json').read_text())
    decoder=ManoDecoder('/home/zz/GraspSONIC/thirdparty/DexForge/assets/mano_v1_2/models/MANO_RIGHT.pkl','cuda:0');decoder.configure_contact_vertices(mapping['mappings'])
    index=json.loads((args.previews/'INDEX.json').read_text());checks=[]
    labels=dict(input='纯贴合 LM 起点（当前模型重评）',velocity_input='此前速度版姿态（当前模型重评）',cem='当前 GPU 版 CEM 最终结果')
    for case,sample in zip(data['cases'],index['samples']):
        case['variants']=[];base=dict(np.load(args.previews/(sample['name']+'.npz')))
        folder=args.experiment/'quality'/sample['name'];provenance=json.loads((folder/'anchor_provenance.json').read_text())
        for method in ['cem','input','velocity_input']:
            path=folder/method;a=dict(np.load(path/'final.npz'));m=json.loads((path/'metrics.json').read_text())[0]
            if method=='cem':
                case['label']=sample['name'][:2]+' · '+sample['type']+' · '+('联合通过' if m['passed'] else '未联合通过')
            v=variant(method,labels[method],base,a,m,mapping['mappings'][sample['type']],decoder,provenance);v['id']='v5_'+method
            v['note']=v['note'].replace('法向只在旧阶段参与验收。','法向不作为当前patch验收门槛。')
            v['note']+='受力使用带切向记忆的物理单位弹簧；5个10ms外步，每外步4个隐式子步。固定模板点误差仅诊断；优化惩罚与表中patch间距及目标邻近阈值一致。'
            for figure in v['figures']:
                for trace in figure['data']:
                    if trace.get('name')=='优化的模板点距离':trace['name']='固定模板点距离（仅诊断）'
                    if trace.get('name')=='验收的 patch 表面间距':trace['name']='目标与验收共用的 patch 间距'
            values=[('末线速度 / m/s',m['end_linear_speed_m_s']),('末角速度 / rad/s',m['end_angular_speed_rad_s']),
                ('一致patch cost',float(a['patch_fit_cost'][0])),('最大切向形变 / mm',float(np.linalg.norm(a['rollout_tangential_displacement_m'][0],axis=-1).max()*1000)),
                ('最大接触法向力 / N',float(a['rollout_normal_force_n'][0].max())),('最大接触切向力 / N',float(np.linalg.norm(a['rollout_tangent_force_n'][0],axis=-1).max()))]
            v['metrics'] += [dict(label=k,value=f'{val:.5f}') for k,val in values]
            if 'rollout_newton_linear_solves' in a:
                v['metrics'].append(dict(label='内部 Newton 解次数',value=str(int(a['rollout_newton_linear_solves'][0]))))
            for i,row in enumerate(v['rows']):row['slot_cost']=f'{a["patch_fit_slot_error"][0,i]:.5f}'
            case['variants'].append(v);checks.append(dict(case=sample['name'],method=method,sha256=hashlib.sha256((path/'final.npz').read_bytes()).hexdigest()))
    data.update(title='MANO 当前 GPU 计算结果',subtitle='6条固定诊断样本，包含成功与失败；切换最终结果、纯贴合起点与此前速度版。左侧初始物体，右侧五步后的物体。',
        eyebrow='CEM / TANGENTIAL MEMORY / ACTUAL GPU OUTPUT',
        views=[dict(label='选中阶段手姿',description='手与初始物体'),dict(label='五步弹簧 rollout 后',description='手保持固定，物体运动50ms')],
        sources=['Kn=Kt=500 N/m；Cn=Ct=5 N·s/m；μ=0.6；五步共50ms。切向形变在候选内累计、滑动限幅、脱离清零。',
                 '贴合cost和验收使用同一patch witness。固定点误差不再参与新cost。',
                 '碰撞仍用分区刚体凸包；显示与贴合验收采用真实三角网格。短时速度通过不代表严格力闭合或长期GSR。生产任务仍停止。'],
        footer='表面距离≤2mm，距固定物面目标≤5mm；10mm同解剖patch；整体覆盖≥80%且每必要手指至少一槽合格；手物/自穿透均≤5mm；末线速度≤0.1m/s、角速度≤2rad/s。',
        columns=[dict(key=k,label=label) for k,label in [('slot','槽'),('finger','手指'),('distance','表面间距 / mm'),('anchor_offset','物面距目标 / mm'),
            ('fixed_anchor','固定点误差（诊断） / mm'),('slot_cost','槽惩罚'),('fit','槽验收'),('face','patch三角形')]])
    spec=importlib.util.spec_from_file_location('builder',ROOT/'skills/inspection-webpages/scripts/build_page.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);builder.build(data,staging)
    page=staging.read_text();extra=previous.split('<label><input type="checkbox" id="hand"')[1].split('<label id="surface-control">')[0]
    page=page.replace('<label id="surface-control">','<label><input type="checkbox" id="hand"'+extra+'<label id="surface-control">')
    page=page.replace(" if(role==='surface'&&$('transparent').checked)t.opacity=.28;",""" const layer=t.meta?.layer;
 if(layer&&['hand','object','mapped','native'].includes(layer)&&!$(layer).checked)t.visible=false;
 if(role==='surface'&&['hand','object'].includes(layer))t.opacity=Number($(layer+'-opacity').value)*($('transparent').checked ? .35 : 1);""")
    page=page.replace("for(const id of ['surface','normals','bones','transparent'])","for(const id of ['surface','normals','bones','transparent','hand','object','mapped','native','hand-opacity','object-opacity'])")
    packed=copy.deepcopy(data);assets={};lookup={}
    for c in packed['cases']:
        for v in c['variants']:
            for f in v['figures']:
                for i,t in enumerate(f['data']):
                    if t.get('type')!='mesh3d':continue
                    key=hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest()
                    if key not in lookup:lookup[key]='a'+str(len(assets));assets[lookup[key]]=t
                    f['data'][i]={'asset':lookup[key]}
    packed['assets']=assets;serialized=json.dumps(packed,ensure_ascii=False,allow_nan=False,separators=(',',':'));target.with_suffix('.data.json').write_text(serialized)
    for ch,escape in [('&',r'\u0026'),('<',r'\u003c'),('>',r'\u003e')]:serialized=serialized.replace(ch,escape)
    start=page.index('const DATA=');end=page.index('\nconst $=',start)
    hydration='\nfor(const c of DATA.cases)for(const v of c.variants)for(const f of v.figures)f.data=f.data.map(t=>t.asset?DATA.assets[t.asset]:t);'
    page=page[:start]+'const DATA='+serialized+';'+hydration+page[end:]
    page=page.replace('<body>','<body><nav style="padding:12px 30px;background:#e3ebe4"><a href="interactive.html">MANO 三维结果</a> · <a href="performance.html">计算耗时分布</a></nav>')
    staging.write_text(page);staging.replace(target)
    (args.previews/'history_viewer_validation.json').write_text(json.dumps(dict(cases=len(data['cases']),variants_each=3,results=checks,
        experiment=str(args.experiment),mesh_reconstruction_verified=True,browser_validated=False),indent=2))

if __name__=='__main__':main()
