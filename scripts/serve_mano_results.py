#!/usr/bin/env python
"""Browse immutable solved MANO shards without using the optimization GPUs."""
from __future__ import annotations
import argparse,concurrent.futures,json,math,os,random,sys,threading,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import viser
from retarget.viewer.mano_results import DEFAULT_RESULTS,DEFAULT_CACHE,SolvedIndex,ClosureCache,snapshot,write

QUALITY=('全部','几何达标','几何未达标')
def closure_markdown(report):
    if report is None:return '**力闭合：未评估**\n\n点击“计算力闭合”，或开启自动计算。'
    def describe(title,r):
        text='通过' if r.get('certified') else '未证实'
        extra=f"；接触点 {r.get('contacts',0)}"
        if 'certified_margin' in r:extra+=f"；margin {r['certified_margin']:.4g}"
        if r.get('reason'):extra+='；'+r['reason']
        return f"**{title}：{text}**{extra}"
    lines=[describe('映射槽投影闭合',report['projected']),describe('2 mm 近表面点闭合',report['near_surface']),
        '**几何＋投影闭合验收：'+('通过' if report['combined_geometry_and_projected'] else '未通过')+'**']
    model=report['projected'].get('closure_model',{})
    if model:
        lines.append(f"摩擦系数 {model.get('friction','—')}；扭转系数 {model.get('torsional_friction_m','—')} m；最大预载 {model.get('max_preload_n','—')} N")
        lines.append(f"物体质量 {model.get('object_mass_kg','—')} kg；使用原始缩放物体的质量／质心／惯量。")
    if 'audited_max_wrench_residual' in report['projected']:
        lines.append(f"独立检查证书最大归一化残差：{report['projected']['audited_max_wrench_residual']:.5g}")
    lines.append('投影闭合可在手尚未贴合时成立；近表面点为几何查询结果，未执行动力学稳定性测试。')
    return '\n\n'.join(lines)

def pose_markdown(row):
    s=row['snapshot'];i=row['index'];e=row['entry'];m=row['metric'];d=m['geometry_diagnostics']
    v,_=s.object_mesh(i,world=False);extent=np.ptp(v,axis=0)*1000
    lines=[f"**{e['grasp_type']} / shard {s.path.name} / row {i}**",
        '**姿态求解已完成；几何验收：'+('通过' if m['geometry_accepted'] else '未通过')+'**',
        f"固定映射槽贴合率：{d['mapped_coverage']:.1%}；仅距离贴合率：{d['mapped_distance_coverage']:.1%}",
        f"固定槽 {d['mapped_contact_count']} 个；2 mm 近表面顶点 {d['contact_count']} 个",
        f"最大槽间隙：{d['max_mapped_gap_m']*1000:.3f} mm；最大手物穿透：{d['max_hand_object_penetration_m']*1000:.3f} mm",
        f"关节位置 RMSE：{d['joint_rmse_to_shadow_m']*1000:.2f} mm",
        f"物体局部包围盒：{extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm；scene_scale={e['scene_scale']:.6g}",
        f"物体网格闭合：{'是' if d['object_watertight'] else '否'}；自碰撞／关节限位：未评估",
        f"来源物体：{Path(e['object']['source_record']).parts[-3]}；原始行 {e['object']['source_row']}",
        '右手 MANO，固定 beta=0；缓存中含完整手／物体世界位姿。']
    if m.get('optimization'):
        o=m['optimization'];reason={'budget':'达到预算','geometry_accepted':'几何达标早停','plateau':'停滞早停（不代表成功）','geometry_recheck_failed':'几何复核未通过'}.get(o['stop_reason'],o['stop_reason'])
        method=o.get("search_method",m.get("method","mppi")).upper()
        count=o.get("cem_iterations",o.get("mppi_iterations",0))
        lines.append(f"IK {o['ik_iterations']} 步；{method} {count} 轮；热启动：{'是' if o['warm_start_used'] else '否'}；{reason}")
    if d.get('mapping_review_flags'):lines.append('**映射待复核槽：'+', '.join(map(str,d['mapping_review_flags']))+'**')
    return '\n\n'.join(lines)

def palm_camera(row):
    s=row['snapshot'];i=row['index'];j=s.poses['mano_joints'][i]
    x=j[5]-j[17];y=j[9]-j[0];normal=np.cross(x,y);normal/=max(np.linalg.norm(normal),1e-9)
    palm=j[[0,5,9,13,17]].mean(0);ov,_=s.object_mesh(i);target=np.concatenate((s.mano_mesh(i)[0],ov)).mean(0)
    if np.dot(normal,ov.mean(0)-palm)<0:normal=-normal
    span=max(np.ptp(np.concatenate((s.mano_mesh(i)[0],ov)),axis=0).max(),.15)
    up=y/max(np.linalg.norm(y),1e-9)
    return target+normal*span*2.4,target,up

class Viewer:
    def __init__(self,server,index,closure):
        self.server=server;self.index=index;self.closure=closure
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='closure-cpu')
        self.futures={};self.lock=threading.RLock()
    def submit(self,row,force=False):
        key=row['key']
        with self.lock:
            old=self.futures.get(key)
            if old is not None and not old.done():return old
            future=self.executor.submit(self.closure.evaluate,row,force);self.futures[key]=future
            future.add_done_callback(lambda f:self.futures.pop(key,None))
            return future
    def connect(self,client,*,start_timer=True,auto_closure=True):
        client.scene.set_up_direction('+z')
        client.scene.add_grid('/grid',width=1,height=1,cell_size=.05,section_size=.25)
        gui=client.gui
        gui.configure_theme(titlebar_content=None,control_layout='collapsible',control_width='medium')
        gui.add_markdown('# 已求解 MANO 抓取\n只读取已验证完成的 MPPI 缓存；后台继续求解。')
        status=gui.add_markdown('')
        typ=gui.add_dropdown('Grasp type',options=['全部',*self.index.types()])
        quality=gui.add_dropdown('几何筛选',options=QUALITY)
        with gui.add_folder('姿态选择',expand_by_default=True):
            page=gui.add_number('页码（每页 100）',initial_value=1,min=1,step=1)
            choices=gui.add_dropdown('Pose',options=['加载中'])
            group=gui.add_button_group('翻页',options=('上一页','下一页','随机'))
            refresh=gui.add_button('刷新已完成结果')
            front=gui.add_button('正对手掌')
            back=gui.add_button('手背视角')
        with gui.add_folder('显示',expand_by_default=True):
            show_object=gui.add_checkbox('显示物体',initial_value=True)
            mapped=gui.add_checkbox('固定映射槽／物体投影',initial_value=True)
            near=gui.add_checkbox('2 mm 近表面顶点',initial_value=False)
            normals=gui.add_checkbox('接触法向',initial_value=False)
            labels=gui.add_checkbox('接触槽编号',initial_value=True)
            shadow=gui.add_checkbox('原始 Shadow 骨架',initial_value=False)
            hand_opacity=gui.add_slider('手透明度',min=.05,max=1,step=.05,initial_value=1.)
            obj_opacity=gui.add_slider('物体透明度',min=.05,max=1,step=.05,initial_value=.3)
            gui.add_markdown('绿色＝距离≤2 mm 且法向误差≤30°；橙色＝未贴合；蓝色＝物体投影；黄色＝近表面顶点。')
        details=gui.add_markdown('')
        with gui.add_folder('力闭合',expand_by_default=True):
            auto=gui.add_checkbox('选择后自动计算（CPU）',initial_value=auto_closure)
            compute=gui.add_button('计算／重算力闭合')
            closure_info=gui.add_markdown(closure_markdown(None))
            forces=gui.add_checkbox('显示投影闭合证书力',initial_value=False)
            scenario=gui.add_dropdown('外载荷场景',options=['0 无外载荷']+[f'{i+1} '+x for i,x in enumerate(('+Fx','+Fy','+Fz','+Tx','+Ty','+Tz','-Fx','-Fy','-Fz','-Tx','-Ty','-Tz'))])
            gui.add_markdown('证书力箭头仅表示当前摩擦模型下的数学平衡解，不是仿真实测接触力。')
        state=dict(rows=[],row=None,total=0,busy=False,handles=[],token=0,closed=False,future=None)
        def clear():
            for handle in state['handles']:handle.remove()
            state['handles']=[]
        def add(handle):state['handles'].append(handle);return handle
        def points(name,p,c,size=.004):
            if len(p):add(client.scene.add_point_cloud(name,np.asarray(p,dtype=np.float32),colors=c,point_size=size,point_shape='circle',precision='float32'))
        def lines(name,p,c,width=2):
            if len(p):add(client.scene.add_line_segments(name,np.asarray(p,dtype=np.float32),colors=c,line_width=width))
        def render():
            r=state['row']
            if r is None:return
            s=r['snapshot'];i=r['index'];p=s.poses;v,f=s.mano_mesh(i);ov,of=s.object_mesh(i)
            with client.atomic():
                clear()
                add(client.scene.add_mesh_simple('/pose/hand',v,f,color=(231,179,151),opacity=hand_opacity.value,side='double'))
                add(client.scene.add_mesh_simple('/pose/object',ov,of,color=(100,150,220),opacity=obj_opacity.value,side='double',visible=show_object.value))
                hp=p['mapped_hand_points'][i];op=p['mapped_object_points'][i]
                good=(np.abs(p['mapped_distances_m'][i])<=.002)&(np.sum(p['mapped_hand_normals'][i]*p['mapped_object_normals'][i],axis=-1)<=-np.cos(np.deg2rad(30)))
                colors=np.where(good[:,None],np.array([40,220,100]),np.array([255,135,35])).astype(np.uint8)
                if mapped.value:
                    points('/pose/slots',hp,colors,.005);points('/pose/projections',op,(60,145,255),.003)
                    lines('/pose/gaps',np.stack((hp,op),1),(240,180,60))
                if normals.value:
                    lines('/pose/hand_normals',np.stack((hp,hp+.012*p['mapped_hand_normals'][i]),1),(220,80,60))
                    lines('/pose/object_normals',np.stack((op,op+.012*p['mapped_object_normals'][i]),1),(70,140,255))
                if mapped.value and labels.value:
                    for k,x in enumerate(hp):add(client.scene.add_label('/pose/label_'+str(k),str(k+1),position=x,font_screen_scale=.7))
                if near.value:points('/pose/near',s.contacts(i)['hand_points_world'],(245,225,35),.0025)
                if shadow.value:
                    j=s.source_joints(i);pairs=[]
                    for base in (1,5,9,13,17):
                        pairs.append([j[0],j[base]])
                        pairs.extend([j[a],j[a+1]] for a in range(base,base+3))
                    lines('/pose/shadow',pairs,(170,130,230),3)
                if forces.value and self.closure.cached(r):
                    witness=self.closure.witness(r)
                    if witness is not None and 'closure_response_forces_n' in witness:
                        force=witness['closure_response_forces_n'][int(scenario.value.split()[0])]
                        scale=.035/max(float(np.linalg.norm(force,axis=-1).max()),1e-9)
                        lines('/pose/forces',np.stack((op,op+force*scale),1),(220,45,200),5)
        def camera(reverse=False):
            if state['row'] is not None:
                position,target,up=palm_camera(state['row'])
                if reverse:position=target-(position-target)
                client.camera.up_direction=up;client.camera.position=position;client.camera.look_at=target
        def evaluate(force=False):
            r=state['row']
            if r is None:return
            cached=None if force else self.closure.cached(r)
            if cached is not None:closure_info.content=closure_markdown(cached);render();return
            token=state['token'];closure_info.content='**力闭合：排队／计算中（CPU）**\n\n首次需要加载原始缩放物体，后续读取缓存。'
            previous=state['future']
            if previous is not None and not previous.done():previous.cancel()
            future=self.submit(r,force);state['future']=future
            def finished(f):
                if state['closed'] or state['token']!=token:return
                try:report=f.result();closure_info.content=closure_markdown(report);render()
                except Exception as error:
                    closure_info.content='**力闭合计算失败，未给出判定**\n\n'+str(error)
                    traceback.print_exc()
            future.add_done_callback(finished)
        def show():
            if state['busy'] or not state['rows']:return
            number=int(choices.value.split(' | ')[0]);ref=state['rows'][number]
            try:
                r=snapshot(ref['path']).row(ref['index']);state['row']=r;state['token']+=1
                details.content=pose_markdown(r)
                closure_info.content=closure_markdown(self.closure.cached(r));render();camera()
                if auto.value:evaluate()
            except Exception as error:
                details.content='**加载失败**\n\n'+str(error);traceback.print_exc()
        def load(reset=False):
            if state['busy']:return
            state['busy']=True
            try:
                self.index.refresh()
                options=['全部',*self.index.types()]
                if tuple(typ.options)!=tuple(options):typ.options=options
                offset=0 if reset else (max(1,int(page.value))-1)*100
                rows,total=self.index.page(typ.value,quality.value,offset,100)
                if not rows and total:
                    offset=max(0,(total-1)//100*100);rows,total=self.index.page(typ.value,quality.value,offset,100)
                state['rows']=rows;state['total']=total;page.value=offset//100+1
                choices.options=[f"{j} | {x['type']} / {x['part']:07d} / row {x['index']}" for j,x in enumerate(rows)] or ['无匹配结果']
                choices.value=choices.options[0]
                report=self.index.status()
                status.content=f"已完成 **{report['viewer_available_samples']:,}** 条；当前筛选 **{total:,}** 条；第 {int(page.value)} / {max(1,math.ceil(total/100))} 页。\n\n后台状态：{report.get('state','未知')}；本页显示完整最终姿态。"
            finally:state['busy']=False
            if rows:show()
            else:
                clear();state['row']=None;state['token']+=1;details.content='当前筛选没有已完成结果。';closure_info.content=closure_markdown(None)
        typ.on_update(lambda _:load(True));quality.on_update(lambda _:load(True))
        page.on_update(lambda _:load());choices.on_update(lambda _:show())
        refresh.on_click(lambda _:load());front.on_click(lambda _:camera());back.on_click(lambda _:camera(True))
        def navigate(event):
            if event.target.value=='随机':
                if state['total']:
                    n=random.randrange(state['total']);page.value=n//100+1;load()
                    if state['rows']:choices.value=choices.options[n%100];show()
            else:
                n=max(1,int(page.value)+(1 if event.target.value=='下一页' else -1))
                page.value=min(n,max(1,math.ceil(state['total']/100)));load()
        group.on_click(navigate)
        for control in (show_object,mapped,near,normals,labels,shadow,hand_opacity,obj_opacity,forces,scenario):control.on_update(lambda _:render())
        auto.on_update(lambda _:evaluate() if auto.value else None)
        compute.on_click(lambda _:evaluate(True))
        load()
        if start_timer:
            def update_status():
                while not state['closed']:
                    time.sleep(20)
                    try:
                        self.index.refresh();report=self.index.status()
                        status.content=f"已完成 **{report['viewer_available_samples']:,}** 条；当前筛选 {state['total']:,} 条（刷新可载入新增）。\n\n后台状态：{report.get('state','未知')}；当前姿态：{state['row']['key'] if state['row'] else '无'}"
                    except Exception:traceback.print_exc()
            threading.Thread(target=update_status,daemon=True).start()
        state.update(render=render,load=load,evaluate=evaluate,camera=camera,controls=dict(typ=typ,quality=quality,page=page,choices=choices,near=near,normals=normals,shadow=shadow,forces=forces,scenario=scenario),closure_info=closure_info)
        return state

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=DEFAULT_RESULTS);p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8101);a=p.parse_args()
    index=SolvedIndex(a.results).refresh()
    if not index.groups:raise RuntimeError('No verified solved MANO shards')
    import functools,websockets.asyncio.server
    # Allow the compressed Viser bundle to load on a slower LAN connection.
    websockets.asyncio.server.serve=functools.partial(websockets.asyncio.server.serve,open_timeout=60)
    server=viser.ViserServer(host=a.host,port=a.port,label='MANO 已求解抓取')
    viewer=Viewer(server,index,ClosureCache(a.cache))
    @server.on_client_connect
    def connected(client):
        state=viewer.connect(client)
        @server.on_client_disconnect
        def disconnected(other):
            if other.client_id==client.client_id:state['closed']=True
    write(a.cache/'server.json',dict(pid=os.getpid(),host=a.host,port=a.port,results=str(a.results),started=time.time()))
    print('READY',a.host,a.port,index.status(),flush=True)
    while True:time.sleep(1)
if __name__=='__main__':main()
