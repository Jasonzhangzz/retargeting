#!/usr/bin/env python
"""Viser inspection of portable MANO failure samples; never re-score or optimize."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import viser
from retarget.viewer.mano_failures import CATEGORIES, camera, describe, load_case


class FailureViewer:
    def __init__(self, server, root, refinement=None):
        self.server, self.root = server, Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.entries = self.manifest['cases']
        self.refinement=Path(refinement) if refinement else None
        self.refined={}
        if self.refinement and (self.refinement/'quality.json').exists():
            report=json.loads((self.refinement/'quality.json').read_text())
            self.refinement_report=report
            self.refined={e['id']:e for e in report['cases']}

    def connect(self, client):
        client.scene.set_up_direction('+z')
        client.scene.add_grid('/grid', width=1., height=1., cell_size=.05, section_size=.25)
        gui = client.gui
        gui.configure_theme(titlebar_content=None, control_layout='collapsible', control_width='medium')
        gui.add_markdown('# MANO · 接触与闭合诊断\n实际最终姿态；CEM 原始结果与独立 GraspADMM/MANO 精修对照。' if self.refined else '# MANO CEM · 失败诊断\n实际最终姿态；读取正式结果抽样，未重新优化或改变验收。')
        if self.refined:
            result=self.refinement_report
            before=result['counts']['moving']['before'];after=result['counts']['moving']['after']
            gui.add_markdown(f"**ADMM/MANO 机制验收：{'通过' if result['mechanism_passed'] else '未通过'}**。16 条固定样本，50 轮；投影闭合 {before['closure']}/16 → {after['closure']}/16，闭合＋覆盖＋碰撞 {before['joint_success']}/16 → {after['joint_success']}/16。\n\n移动槽联合覆盖均值 {before['mean_coverage']:.1%} → {after['mean_coverage']:.1%}；没有替换生产 CEM。")
        counts = self.manifest['counts']
        gui.add_markdown(f"固定快照 **{self.manifest['samples']:,}** 条；采样时刻 {self.manifest['created_utc']}\n\n" +
            '\n\n'.join(f"{CATEGORIES[k]}：**{counts[k]:,}**" for k in CATEGORIES))
        breakdown = self.root/'diagnostic_breakdown.json'
        if breakdown.exists():
            diagnostic = json.loads(breakdown.read_text())
            gui.add_markdown('此快照每条有 **4 个指定接触槽**，80% 门槛实际要求 **4/4** 合格。\n\n' +
                f"**{diagnostic['counts']['distance_pass_but_joint_coverage_fail']:,}** 条的 4 个槽距离均合格，但法向导致覆盖未达标；接触不足不能一概解释为距离太远。")
        selected = gui.add_dropdown('失败类别', options=['全部', *[CATEGORIES[k] for k in CATEGORIES if counts[k]]])
        choices = gui.add_dropdown('样本', options=['加载中'])
        stage=gui.add_dropdown('结果阶段',options=['CEM 原始结果','ADMM 原固定槽','ADMM 更新接触槽'] if self.refined else ['CEM 原始结果'])
        nav = gui.add_button_group('浏览', options=['上一条', '下一条'])
        front = gui.add_button('正对手掌')
        back = gui.add_button('手背视角')
        with gui.add_folder('显示', expand_by_default=True):
            show_object = gui.add_checkbox('显示物体', initial_value=True)
            slots = gui.add_checkbox('接触槽与物体投影', initial_value=True)
            normals = gui.add_checkbox('手与物体法向', initial_value=True)
            labels = gui.add_checkbox('槽位编号', initial_value=True)
            near = gui.add_checkbox('2 mm 近表面顶点', initial_value=False)
            targets = gui.add_checkbox('ADMM 物面目标与贴合误差', initial_value=True)
            original = gui.add_checkbox('叠加 CEM 原始手部', initial_value=False)
            hand_alpha = gui.add_slider('手透明度', min=.05, max=1., step=.05, initial_value=1.)
            object_alpha = gui.add_slider('物体透明度', min=.05, max=1., step=.05, initial_value=.35)
            gui.add_markdown('绿色＝距离和法向均合格；橙色＝仅距离失败；紫色＝仅法向失败；红色＝两项都失败。蓝点＝物体投影，连线＝槽位到投影的间隙。')
        details = gui.add_markdown('')
        slot_table = gui.add_markdown('')
        state = dict(entry=None, arrays=None, handles=[], rows=[])

        def render():
            a = state['arrays']
            if a is None: return
            with client.atomic():
                for handle in state['handles']: handle.remove()
                handles = state['handles'] = []
                def add(handle): handles.append(handle)
                def points(name, values, color, size):
                    if len(values): add(client.scene.add_point_cloud(name, np.asarray(values,dtype=np.float32), colors=color, point_size=size, point_shape='circle', precision='float32'))
                def lines(name, values, color, width=2):
                    if len(values): add(client.scene.add_line_segments(name, np.asarray(values,dtype=np.float32), colors=color, line_width=width))
                add(client.scene.add_mesh_simple('/pose/hand', a['hand_vertices'], a['hand_faces'], color=(231,179,151), opacity=hand_alpha.value, side='double'))
                add(client.scene.add_mesh_simple('/pose/object', a['object_vertices'], a['object_faces'], color=(100,150,220), opacity=object_alpha.value, visible=show_object.value, side='double'))
                if original.value and stage.value!='CEM 原始结果':
                    raw=load_case(self.root,state['entry'])
                    add(client.scene.add_mesh_simple('/pose/original_hand',raw['hand_vertices'],raw['hand_faces'],color=(130,150,180),opacity=.2,side='double'))
                hp, op = a['hand_points'], a['object_points']
                if slots.value:
                    colors = np.tile([235,65,65], (len(hp),1)).astype(np.uint8)
                    colors[~a['distance_ok'] & a['normal_ok']] = [255,135,35]
                    colors[a['distance_ok'] & ~a['normal_ok']] = [190,90,240]
                    colors[a['fitted']] = [40,220,100]
                    points('/pose/slots',hp,colors,.004)
                    points('/pose/projections',op,(60,145,255),.0025)
                    lines('/pose/gaps',np.stack((hp,op),1),(240,180,60))
                    if labels.value:
                        for k,p in enumerate(hp): add(client.scene.add_label(f'/pose/label_{k}',str(k+1),position=p,font_screen_scale=.7))
                if normals.value:
                    lines('/pose/hand_normals',np.stack((hp,hp+.01*a['hand_normals']),1),(235,90,60))
                    lines('/pose/object_normals',np.stack((op,op+.01*a['object_normals']),1),(65,150,255))
                if near.value: points('/pose/near',a['near_points'],(245,225,35),.002)
                if targets.value and 'object_targets' in a:
                    points('/pose/targets',a['object_targets'],(20,235,230),.004)
                    lines('/pose/target_gaps',np.stack((a['moving_hand_points'],a['object_targets']),1),(20,235,230),3)

        def set_camera(reverse=False):
            if state['arrays'] is not None:
                eye,target,up = camera(state['arrays'],reverse)
                client.camera.up_direction=up
                client.camera.position=eye
                client.camera.look_at=target

        def show():
            if not state['rows']: return
            entry = state['rows'][int(choices.value.split(' | ')[0])]
            a = load_case(self.root, entry)
            state.update(entry=entry,arrays=a)
            description=entry
            extra=''
            if stage.value!='CEM 原始结果' and entry['id'] in self.refined:
                refined=self.refined[entry['id']]
                a=load_case(self.refinement,refined)
                kind='fixed' if stage.value=='ADMM 原固定槽' else 'moving'
                for name in ('hand_points','hand_normals','object_points','object_normals','gap_m','fitted','distance_ok','normal_ok','normal_angle_deg'):
                    a[name]=a[kind+'_'+name]
                metrics=refined['after'][kind];collision=refined['after']['collision']
                description=dict(entry,projected_closure=metrics['closure'],coverage=metrics['coverage'],
                    geometry_accepted=metrics['joint_success'],closure_margin=metrics['closure_margin'],closure_cost=metrics['closure_cost'],
                    object_penetration_m=collision['object_depth_m'],self_penetration_m=collision['self_depth_m'])
                extra=f"\n\n**GraspADMM/MANO：{'完成' if refined['completed'] else '失败'}，{refined['outer_steps']} 轮。**\n\n手步为弹簧位置 LM，没有刚体动力学碰撞响应；未增加法向目标。完整验收同时检查手物穿透≤2 mm。\n\n物面目标自身闭合：{'通过' if refined['after']['targets']['closure'] else '未通过'}；目标到实际移动手点最大距离 {refined['after']['targets']['hand_gap_max_m']*1000:.3f} mm。青色为物面目标与实际手点连线。"
                state['arrays']=a
            details.content = describe(description,a)+extra
            slot_table.content = '| 槽 | 间隙 mm | 法向误差 ° | 合格 |\n|---:|---:|---:|:---:|\n' + '\n'.join(
                f"| {i+1} | {a['gap_m'][i]*1000:.2f} | {a['normal_angle_deg'][i]:.1f} | {'✓' if a['fitted'][i] else '×'} |" for i in range(len(a['gap_m'])))
            render(); set_camera()

        def refresh():
            state['rows']=[e for e in self.entries if selected.value=='全部' or CATEGORIES[e['category']]==selected.value]
            choices.options=[f"{j} | {CATEGORIES[e['category']]} / row {e['row']}" for j,e in enumerate(state['rows'])]
            choices.value=choices.options[0]
            show()

        def navigate(event):
            j=int(choices.value.split(' | ')[0])
            delta=1 if event.target.value=='下一条' else -1
            choices.value=choices.options[(j+delta)%len(choices.options)]

        selected.on_update(lambda _:refresh()); choices.on_update(lambda _:show());stage.on_update(lambda _:show())
        nav.on_click(navigate); front.on_click(lambda _:set_camera()); back.on_click(lambda _:set_camera(True))
        for control in (show_object,slots,normals,labels,near,targets,original,hand_alpha,object_alpha): control.on_update(lambda _:render())
        refresh()
        return dict(state=state,render=render,camera=set_camera,show=show,controls=(selected,choices,near,normals,show_object),stage=stage)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('output/mano_cem_failures_20261104'))
    parser.add_argument('--host',default='0.0.0.0')
    parser.add_argument('--port',type=int,default=8102)
    parser.add_argument('--refinement',type=Path)
    args=parser.parse_args()
    server=viser.ViserServer(host=args.host,port=args.port,label='MANO CEM · 失败诊断',verbose=False)
    viewer=FailureViewer(server,args.data,args.refinement)
    server.on_client_connect(viewer.connect)
    print(f'VISER_READY http://127.0.0.1:{server.get_port()}',flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: server.stop()
