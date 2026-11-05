#!/usr/bin/env python3
"""Browse original Shadow GraspADMM results, with independent diagnostic labels."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import viser


def serve(root,host,port):
    manifest=json.loads((root/'manifest.json').read_text());entries=manifest['cases']
    geometry=np.load(root/'geometry.npz');poses=np.load(root/'poses.npz')
    source_seeded=manifest.get('seed_kind')=='shadow_gsr_transfer'
    model_label='MANO 刚体' if 'mano' in manifest['model'].lower() else 'Shadow'
    server=viser.ViserServer(host=host,port=port,label=model_label+' · GraspADMM',verbose=False)
    @server.on_client_connect
    def connect(client):
        client.scene.set_up_direction('+z');gui=client.gui
        gui.configure_theme(control_layout='collapsible',control_width='large')
        gui.add_markdown(f'# {model_label} · GraspADMM\n{len(manifest["scenes"])} 个物体—尺度组合，{len(entries)} 条实际输入。{model_label} / Clarabel / MuJoCo；点击切换查看初始化与最终状态。')
        summary=manifest.get('summary')
        if summary and summary.get('penetration_tolerances_m'):
            gui.add_markdown('此配置允许手物和自穿透各 **2 mm**。原生 GSR 与新增六扭矩测试分开显示。')
        extra_categories=['六扭矩通过','六扭矩失败','碰撞超出2mm'] if any(r.get('wrench') for r in entries) else []
        category=gui.add_dropdown('类别',options=['全部','原生 GSR 通过','原生 GSR 失败','联合覆盖失败',*extra_categories])
        scene=gui.add_dropdown('物体场景',options=['全部',*manifest['scenes']])
        choice=gui.add_dropdown('样本',options=['加载中'])
        if source_seeded:gui.add_markdown(f"来源：**{manifest['source_selected']} 条 Shadow GSR 成功抓姿**，逐条映射到 MANO。")
        stage=gui.add_dropdown('阶段',options=['ADMM 最终姿态','MANO 映射初值','Shadow 成功姿态','三阶段对比'] if source_seeded else ['ADMM 最终姿态','原生初始化','并排对比'])
        nav=gui.add_button_group('浏览',options=['上一条','下一条'])
        fit=gui.add_button('重置视角')
        with gui.add_folder('显示',expand_by_default=True):
            show_object=gui.add_checkbox('显示物体',initial_value=True)
            show_slots=gui.add_checkbox('手部接触槽与物面投影',initial_value=True)
            show_targets=gui.add_checkbox('ADMM 目标点与贴合连线',initial_value=True)
            normals=gui.add_checkbox('显示法向',initial_value=False)
            labels=gui.add_checkbox('显示槽位编号',initial_value=True)
            opacity=gui.add_slider('物体透明度',min=.05,max=1.,step=.05,initial_value=.45)
        gui.add_markdown('绿＝距离≤2 mm且法向≤30°；橙＝距离失败；紫＝法向失败；红＝两项失败。蓝点＝手点物面投影；青点＝ADMM 物面目标。初始化槽位为灰色，未单独判分。单位为米。')
        details=gui.add_markdown('');table=gui.add_markdown('');state=dict(rows=[],entry=None,handles=[])
        def reset_camera(_=None):
            if state['entry'] is None:return
            info=manifest['scenes'][state['entry']['scene']];radius=max(info['radius'],.12)
            factor=1.6 if stage.value=='三阶段对比' else 1.
            client.camera.up_direction=(0,0,1);client.camera.position=(radius*2.5*factor,-radius*3.2*factor,radius*2.0*factor);client.camera.look_at=(0,0,0)
        def render(_=None):
            r=state['entry']
            if r is None:return
            reason=r['failure_reason'].replace(r['input']+' ','')
            i=r['index'];center=np.array(manifest['scenes'][r['scene']]['center']);radius=max(manifest['scenes'][r['scene']]['radius'],.12)
            with client.atomic():
                for h in state['handles']:h.remove()
                state['handles']=[]
                def add(h):state['handles'].append(h)
                def points(name,p,color,size=.004):
                    if len(p):add(client.scene.add_point_cloud(name,np.asarray(p,np.float32),colors=color,point_size=size,point_shape='circle'))
                def lines(name,p,color):
                    if len(p):add(client.scene.add_line_segments(name,np.asarray(p,np.float32),colors=color,line_width=2))
                views=['shadow','initial','final'] if stage.value=='三阶段对比' else (['initial','final'] if stage.value=='并排对比' else ['shadow' if stage.value=='Shadow 成功姿态' else ('initial' if stage.value in ('原生初始化','MANO 映射初值') else 'final')])
                for k,s in enumerate(views):
                    off=-center+np.array([(k-(len(views)-1)/2)*radius*2.6 if len(views)>1 else 0,0,0]);prefix='/pose/'+s
                    if show_object.value:add(client.scene.add_mesh_simple(prefix+'/object',geometry[r['scene']+'_vertices']+off,geometry[r['scene']+'_faces'],color=(105,150,210),opacity=opacity.value,side='double'))
                    if f'{i}_{s}_positions' not in poses:continue
                    mesh_prefix='shadow_hand' if s=='shadow' else 'hand'
                    count=manifest['shadow_hand_geoms'] if s=='shadow' else manifest['hand_geoms']
                    for g in range(count):
                        add(client.scene.add_mesh_simple(prefix+f'/hand_{g}',geometry[f'{mesh_prefix}_{g}_vertices'],geometry[f'{mesh_prefix}_{g}_faces'],position=poses[f'{i}_{s}_positions'][g]+off,wxyz=poses[f'{i}_{s}_wxyz'][g],color=(195,205,216),side='double'))
                    add(client.scene.add_label(prefix+'/title',('Shadow GSR 成功源姿态' if s=='shadow' else (('MANO 映射初值' if source_seeded else '原生初始化') if s=='initial' else 'ADMM 最终姿态')),position=tuple(off+center+[0,0,radius*1.2])))
                    key=f'{i}_hand_slots' if s=='final' else (f'{i}_shadow_slots' if s=='shadow' else f'{i}_initial_slots')
                    if key not in poses:continue
                    slots=poses[key];hp=slots[:,:3]+off
                    d=r.get('diagnostics') or {};dist=np.array(d.get('slot_distances_m',[]));ang=np.array(d.get('slot_angles_deg',[]))
                    colors=np.tile([150,150,150],(len(hp),1)).astype(np.uint8)
                    if s=='final' and len(dist)==len(hp):
                        goodd=dist<=.002;goodn=ang<=30;colors[:]=[235,65,65];colors[goodd&goodn]=[40,220,100];colors[~goodd&goodn]=[255,135,35];colors[goodd&~goodn]=[190,90,240]
                    if show_slots.value:
                        points(prefix+'/slots',hp,colors)
                        if labels.value:
                            for j,p in enumerate(hp):add(client.scene.add_label(prefix+f'/slot_{j}',str(j+1),position=tuple(p)))
                        if s=='final' and f'{i}_hand_projections' in poses:
                            op=poses[f'{i}_hand_projections']+off;points(prefix+'/projections',op,(60,145,255),.0025);lines(prefix+'/gaps',np.stack((hp,op),1),(240,180,60))
                    if normals.value:lines(prefix+'/normals',np.stack((hp,hp+slots[:,3:]*.012),1),(235,90,60))
                    if s=='final' and show_targets.value and f'{i}_target_points' in poses:
                        target=poses[f'{i}_target_points']+off;points(prefix+'/targets',target,(20,235,230));lines(prefix+'/target_gaps',np.stack((hp,target),1),(20,235,230))
            d=r.get('diagnostics') or {};yn=lambda key:'通过' if d.get(key) else ('未通过' if key in d else '未检测')
            details.content=f"**#{i:03d} · {r['scene']}**\n\n以下指标均对应 ADMM 最终姿态。\n\n原生输出：{'有' if r['native_operation_pass'] else '被拒绝'}；六方向 GSR：{'通过' if r['native_gsr_pass'] else '未通过'}。\n\n联合槽覆盖：{d.get('slot_joint_coverage',float('nan')):.0%}。\n\n手物穿透 {d.get('hand_object_depth_m',float('nan'))*1000:.3f} mm；自穿透 {d.get('self_depth_m',float('nan'))*1000:.3f} mm。\n\n原生阶段：`{reason}`\n\n失败姿态显示最后取得的实际最终状态。缺失结果不会用初值替代。"
            if r.get('source_shadow'):
                source=r['source_shadow'];details.content+=f"\n\nShadow 来源 **#{source['index']}：GSR 通过**。映射后接触点相对源位置最大误差 **{source['max_mapping_error_m']*1000:.3f} mm**。\n\n初值经跨模型 IK 映射；其几何可行性另行评估。"
            if r.get('wrench'):
                w=r['wrench'];details.content+=f"\n\n**扩展稳定性**：六外力 {'通过' if w['force_passed'] else '未通过'}；六扭矩 {'通过' if w['torque_passed'] else '未通过'}。扭矩幅度 {w.get('torque_magnitude_nm',float('nan')):.4g} N·m；当前允许手物和自穿透各2 mm。"
            table.content='**最终姿态接触槽**\n\n| 槽位 / link | 距离 mm | 法向误差 ° |\n|---|---:|---:|\n'+'\n'.join(f"| {j+1} / {body} | {dist*1000:.2f} | {ang:.1f} |" for j,(body,dist,ang) in enumerate(zip(r['hand_cbody'],d.get('slot_distances_m',[]),d.get('slot_angles_deg',[]))))
        def show(_=None):
            if not state['rows']:return
            state['entry']=state['rows'][int(choice.value.split(' | ')[0])];render();reset_camera()
        def refresh(_=None):
            def accept(r):
                d=r.get('diagnostics') or {};c=category.value
                return (scene.value=='全部' or r['scene']==scene.value) and (c=='全部' or c=='原生 GSR 通过' and r['native_gsr_pass'] or c=='原生 GSR 失败' and not r['native_gsr_pass'] or c=='联合覆盖失败' and d.get('slot_joint_pass') is False or c=='六扭矩通过' and (r.get('wrench') or {}).get('torque_passed') is True or c=='六扭矩失败' and (r.get('wrench') or {}).get('torque_passed',False) is False or c=='碰撞超出2mm' and d.get('tolerated_collision_pass') is False)
            state['rows']=[r for r in entries if accept(r)]
            choice.options=[f"{j} | #{r['index']:03d} · {r['scene']} · {'GSR ✓' if r['native_gsr_pass'] else 'GSR ×'}" for j,r in enumerate(state['rows'])] or ['无匹配样本']
            choice.value=choice.options[0]
            if state['rows']:show()
            else:
                for h in state['handles']:h.remove()
                state['handles']=[];state['entry']=None;details.content='无匹配样本';table.content=''
        def navigate(event):
            if state['rows']:choice.value=choice.options[(int(choice.value.split(' | ')[0])+(1 if event.target.value=='下一条' else -1))%len(choice.options)]
        category.on_update(refresh);scene.on_update(refresh);choice.on_update(show);stage.on_update(lambda _:(render(),reset_camera()));nav.on_click(navigate);fit.on_click(reset_camera)
        for widget in (show_object,show_slots,show_targets,normals,labels,opacity):widget.on_update(render)
        refresh()
    (root/'server.json').write_text(json.dumps(dict(pid=os.getpid(),host=host,port=server.get_port(),url=f'http://127.0.0.1:{server.get_port()}/')))
    print(f'VISER_READY http://127.0.0.1:{server.get_port()}/',flush=True)
    while True:time.sleep(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8103);a=p.parse_args();serve(a.data,a.host,a.port)
