"""Browse every recorded glove pose and its unoptimized IK + MCP-aligned robot seed."""
import argparse,json,sys,time,threading
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def run(args):
    import torch,viser
    from scipy.spatial.transform import Rotation
    from moxian_web.cart_wrist import FloatingCartHand
    from moxian_web.cart_palm_alignment import alignment_from_topology,MCP,robot_palm_landmarks
    from moxian_web.qnbot_surface import QnbotSurface
    from retarget.baseline.geort.adapter import SourceAdapter
    torch.set_num_threads(1)
    folder=args.recording;topology=json.loads((folder/'topology.json').read_text())
    adapter=SourceAdapter(topology);source=QnbotSurface(topology['hand_side'],args.device,finger_twist=True)
    source.load_artifact(json.loads((folder/'profile.json').read_text()));source.fixed=True
    hand=FloatingCartHand(args.device);alignment=alignment_from_topology(hand,folder/'topology.json')
    with np.load(folder/'motion.npz') as f:motion=f['qpos']
    with np.load(folder/'cart_ik.npz') as f:ik=f['q'];valid=f['valid'];stamps=f['timestamps_ms']
    if len(ik)!=len(motion):raise ValueError('Recording and IK length mismatch')
    base=np.array(alignment['translation_m']+alignment['rotation_vector_rad'],np.float32)
    def compute(i,aligned=True):
        with torch.no_grad():
            v,j,_=source(torch.as_tensor(motion[i:i+1],device=args.device,dtype=torch.float32))
            j=j[0].cpu().numpy()[adapter.indices];v=v[0].cpu().numpy()
            v=(v-j[0])@adapter.transform;j=(j-j[0])@adapter.transform
            # Invalid IK is reported, never silently shown as a valid seed.
            q=ik[i] if valid[i] else np.zeros(20,np.float32)
            state=np.r_[q,base if aligned else np.zeros(6)]
            p,r,landmarks,*_=hand.frames(hand.tensor(state[None]))
            landmarks[:,list(MCP)]=robot_palm_landmarks(hand,hand.tensor(state[None]))[:,list(MCP)]
        return v,j,p[0].cpu().numpy(),r[0].cpu().numpy(),landmarks[0].cpu().numpy()
    if args.audit:
        before=[];after=[];tips_before=[];tips_after=[];frame_ids=[]
        start=time.perf_counter()
        for start_i in range(0,len(motion),256):
            ids=np.arange(start_i,min(start_i+256,len(motion)));ids=ids[valid[ids]]
            if not len(ids):continue
            with torch.no_grad():
                _,j,_=source(torch.as_tensor(motion[ids],device=args.device,dtype=torch.float32))
                j=j[:,adapter.indices];j=(j-j[:,:1])@hand.tensor(adapter.transform)
                state=hand.tensor(np.c_[ik[ids],np.tile(base,(len(ids),1))])
                a=hand.frames(state)[2];b=hand.frames(state[:,:20])[2]
                a[:,list(MCP)]=robot_palm_landmarks(hand,state)[:,list(MCP)]
                b[:,list(MCP)]=robot_palm_landmarks(hand,state[:,:20])[:,list(MCP)]
                for out,pts,ind in ((before,b,MCP),(after,a,MCP),(tips_before,b,(4,8,12,16,20)),(tips_after,a,(4,8,12,16,20))):
                    out.extend(((pts[:,list(ind)]-j[:,list(ind)]).norm(dim=-1)*1000).cpu().numpy())
                frame_ids.extend(ids.tolist())
        def stats(x):
            x=np.asarray(x);return dict(mean_mm=float(x.mean()),p95_mm=float(np.percentile(x,95)),max_mm=float(x.max()))
        result=dict(alignment=alignment,recording=folder.name,frames=len(motion),valid_frames=len(frame_ids),
            mcp_before=stats(before),mcp_after=stats(after),tips_before=stats(tips_before),tips_after=stats(tips_after),
            optimizer_updates=0,seconds=time.perf_counter()-start,
            meaning='all recorded valid IK seeds; zero learned wrist residual; no policy rollout or fitting to trajectory')
        args.audit.parent.mkdir(parents=True,exist_ok=True);args.audit.write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        if args.check:return
    server=viser.ViserServer(host=args.host,port=args.port,label='CART · MCP alignment audit')
    server.scene.set_up_direction('+z');lock=threading.RLock()
    server.gui.add_markdown(f'**完整训练录制：{folder.name} · {len(motion):,} 帧**\n\n橙色：glove；蓝色半透明：Wuji2。每帧为原 IK 手指姿态 + 初始配准，腕部残差为零，非 policy rollout。')
    frame=server.gui.add_slider('帧',0,len(motion)-1,1,0)
    mode=server.gui.add_dropdown('初始对齐',('掌骨刚体对齐','旧：原点重合'),initial_value='掌骨刚体对齐')
    opacity=server.gui.add_slider('Wuji2 透明度',.05,1.,.05,.35)
    glove_opacity=server.gui.add_slider('glove 透明度',.05,1.,.05,.8)
    previous=server.gui.add_button('上一帧');following=server.gui.add_button('下一帧');play=server.gui.add_checkbox('播放',False)
    fps=server.gui.add_slider('回放帧率',1,60,1,20)
    info=server.gui.add_markdown('')
    faces=source.faces.cpu().numpy()
    if np.linalg.det(adapter.transform)<0:faces=faces[:,::-1]
    glove=server.scene.add_mesh_simple('/glove',np.zeros((len(source.vertices0),3)),faces,color=(225,155,75),opacity=.8,side='double')
    robot=[]
    for l,mesh in enumerate(hand.surface.meshes):
        robot.append(server.scene.add_mesh_simple(f'/robot/link_{l}',np.asarray(mesh.vertices),np.asarray(mesh.faces),
            color=(65,160,245),opacity=.35,side='double'))
    axis=server.scene.add_frame('/glove_wrist',axes_length=.04,axes_radius=.0007)
    robot_axis=server.scene.add_frame('/robot_origin',axes_length=.025,axes_radius=.0005)
    mcp_lines=server.scene.add_line_segments('/mcp_errors',np.zeros((4,2,3)),colors=(255,65,120),line_width=3)
    def update(_=None):
        with lock,server.atomic():
            i=int(frame.value);v,j,p,r,lm=compute(i,mode.value.startswith('掌骨'))
            glove.vertices=v;glove.opacity=glove_opacity.value
            for link,position,rotation in zip(robot,p,r):
                link.position=position;link.wxyz=Rotation.from_matrix(rotation).as_quat(scalar_first=True);link.opacity=opacity.value
            robot_axis.position=p[0];robot_axis.wxyz=Rotation.from_matrix(r[0]).as_quat(scalar_first=True)
            mcp_lines.points=np.stack((j[list(MCP)],lm[list(MCP)]),1)
            errors=np.linalg.norm(j[list(MCP)]-lm[list(MCP)],axis=-1)*1000
            tips=np.linalg.norm(j[[4,8,12,16,20]]-lm[[4,8,12,16,20]],axis=-1)*1000
            info.content=f'帧 **{i} / {len(motion)-1}** · t={(stamps[i]-stamps[0])/1000:.3f}s · IK有效：{bool(valid[i])}\n\n掌骨误差 mm：'+', '.join(f'{x:.1f}' for x in errors)+'\n\n指尖误差 mm：'+', '.join(f'{x:.1f}' for x in tips)
    for control in (frame,mode,opacity,glove_opacity):control.on_update(update)
    @previous.on_click
    def _(_):frame.value=max(0,frame.value-1)
    @following.on_click
    def _(_):frame.value=min(len(motion)-1,frame.value+1)
    @server.on_client_connect
    def _(client):
        client.camera.position=(.35,-.3,.23);client.camera.look_at=(0,0,.09);client.camera.up=(0,0,1)
    update();print(f'VIEWER http://localhost:{args.port} frames={len(motion)}',flush=True)
    while True:
        time.sleep(1/fps.value)
        if play.value:frame.value=(frame.value+1)%len(motion)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,default=ROOT/'output/moxian/cart/recordings/20261118T134537_42b050e5')
    p.add_argument('--device',default='cuda:0');p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8772)
    p.add_argument('--audit',type=Path);p.add_argument('--check',action='store_true');run(p.parse_args())
