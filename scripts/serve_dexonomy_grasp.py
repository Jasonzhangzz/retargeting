#!/usr/bin/env python3
"""Paginated grasp-only viewer for every cached pose, with lazy rendering."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,functools,json,time,urllib.parse,io,mimetypes,re,threading,concurrent.futures,multiprocessing
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retarget.env.grasp.dexonomy_grasp_static import StaticModels

FULL=ROOT/'results/dexonomy_shadow_grasp_full'
PREVIEW=ROOT/'results/dexonomy_shadow_grasp_preview_20261103'
MODELS=None
RENDER_POOL=None
INDEX_LOCK=threading.RLock()

class Index:
    def __init__(self):self.position=0;self.full={};self.preview=None;self.inode=None
    def update(self):
        journal=FULL/'shards.jsonl'
        if journal.exists():
            stat=journal.stat()
            if stat.st_ino!=self.inode or stat.st_size<self.position:
                self.position=0;self.full={};self.inode=stat.st_ino
            with journal.open() as stream:
                stream.seek(self.position)
                while True:
                    start=stream.tell();line=stream.readline()
                    if not line or not line.endswith('\n'):break
                    try:item=json.loads(line)
                    except ValueError:break
                    self.position=stream.tell()
                    if item.get('samples',0)>0:self.full[(item['type'],int(item['part']))]=item
        if self.preview is None:
            paths=list(PREVIEW.glob('*/shards/*/manifest.json'))
            if paths:
                self.preview={}
                for path in paths:
                    m=json.loads(path.read_text());typ=m['cases'][0]['grasp_type']
                    self.preview[(typ,int(path.parent.name))]={'type':typ,'part':int(path.parent.name),'samples':len(m['cases']),'path':str(path.parent)}
    def groups(self,source,typ=''):
        self.update();data=self.full if source=='full' else (self.preview or {})
        result=[v for v in data.values() if not typ or v['type']==typ]
        return sorted(result,key=lambda v:(int(v['type'].split('_')[0]),v['part']))
    def types(self,source):
        result={}
        for group in self.groups(source):result[group['type']]=result.get(group['type'],0)+group['samples']
        return result
INDEX=Index()

@functools.lru_cache(maxsize=12)
def shard_data(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    with np.load(path/'poses.npz',allow_pickle=False) as arrays:
        poses={k:arrays[k] for k in ['qpos','grasp_qpos','object_scale','contact_counts']}
    return manifest,poses

def find_shard(source,typ,part):
    root=FULL if source=='full' else PREVIEW
    if not re.fullmatch(r'[0-9A-Za-z_]+',typ):raise ValueError('Invalid grasp type')
    path=(root/typ/'shards'/f'{int(part):07d}').resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Invalid path')
    return path

def locked_index(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        with INDEX_LOCK:return function(*args,**kwargs)
    return wrapped

@locked_index
def status():
    INDEX.update()
    p=FULL/'status.json';full=json.loads(p.read_text()) if p.exists() else {'state':'starting','samples_cached':0}
    full['all_reconstructed']=full.get('state')=='completed' and full.get('errors',0)==0
    return {'full':full,'full_types':INDEX.types('full'),'preview_types':INDEX.types('preview'),
        'preview_samples':sum(INDEX.types('preview').values()),'stage':'grasp','physical_success_evaluated':False}

@locked_index
def listing(source,typ,offset,limit):
    groups=INDEX.groups(source,typ);total=sum(g['samples'] for g in groups);rows=[];base=0
    for group in groups:
        end=base+group['samples']
        if end<=offset:base=end;continue
        if base>=offset+limit:break
        path=find_shard(source,group['type'],group['part']);manifest,poses=shard_data(str(path))
        start=max(0,offset-base);stop=min(group['samples'],offset+limit-base)
        for i in range(start,stop):
            entry=manifest['cases'][i];params=urllib.parse.urlencode({'source':source,'type':group['type'],'part':group['part'],'row':i})
            object_id=Path(entry['object']['source_record']).parts[-3]
            rows.append({'case':entry['name'],'grasp_type':group['type'],'object_id':object_id,
                'scale':poses['object_scale'][i].tolist(),'contacts':int(poses['contact_counts'][i]),'image':'/image?'+params})
        base=end
    return {'source':source,'type':typ,'offset':offset,'limit':limit,'total':total,'rows':rows,'stage':'grasp'}

def render(source,typ,part,row):
    global MODELS
    path=find_shard(source,typ,part);manifest,poses=shard_data(str(path));row=int(row)
    if not 0<=row<len(manifest['cases']):raise ValueError('Invalid pose index')
    cache=PREVIEW/'live_images';cache.mkdir(exist_ok=True)
    filename=cache/f'{source}_{typ}_{int(part):07d}_{row:05d}.jpg'
    if filename.exists():return filename.read_bytes()
    entry=manifest['cases'][row]
    # Reuse the already rendered original-grasp previews when available.
    if source=='preview':
        image=PREVIEW/'images'/f"{typ}_{entry['name']}_grasp.png"
        if image.exists():
            from PIL import Image
            buffer=io.BytesIO();Image.open(image).convert('RGB').save(buffer,format='JPEG',quality=85)
            filename.write_bytes(buffer.getvalue());return buffer.getvalue()
    import mujoco
    from PIL import Image,ImageDraw
    if MODELS is None:MODELS=StaticModels(ROOT/'data/dexonomy',2,visual=True)
    record={'scene_path':entry['scene_path'],'scene_scale':np.array([entry['scene_scale']]),'grasp_qpos':np.zeros((1,29))}
    env,_,_=MODELS.get(record,0);model,data=env._model,env._data
    data.qpos[:]=poses['qpos'][row]
    mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data);mujoco.mj_camlight(model,data)
    model.geom_group[(model.geom_contype!=0)|(model.geom_conaffinity!=0)]=3
    model.vis.headlight.active=1;model.vis.headlight.ambient[:]=[.55]*3;model.vis.headlight.diffuse[:]=[.8]*3
    for g in range(model.ngeom):
        if model.geom_group[g]==3:continue
        name=model.body(int(model.geom_bodyid[g])).name;model.geom_matid[g]=-1
        model.geom_rgba[g]=[.65,.74,.83,1] if name.startswith('hand-') else [.92,.58,.24,1]
    opt=mujoco.MjvOption();opt.geomgroup[3]=False;opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.azimuth=135;cam.elevation=-25
    ids=[b for b in range(1,model.nbody) if model.body_mocapid[b]<0];lower=data.xpos[ids].min(0);upper=data.xpos[ids].max(0)
    for g in range(model.ngeom):
        if model.geom_group[g]==3:continue
        lower=np.minimum(lower,data.geom_xpos[g]-model.geom_rbound[g]);upper=np.maximum(upper,data.geom_xpos[g]+model.geom_rbound[g])
    cam.lookat[:]=(lower+upper)*.5;cam.distance=max(.3,float(np.linalg.norm(upper-lower))*1.15)
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        renderer.update_scene(data,camera=cam,scene_option=opt);image=Image.fromarray(renderer.render().copy())
    draw=ImageDraw.Draw(image);draw.rectangle((0,0,640,40),fill=(22,27,34))
    draw.text((10,8),typ+' / original grasp',fill=(240,240,245))
    draw.text((10,24),'Geometric contacts: '+str(poses['contact_counts'][row]),fill=(160,220,240))
    buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=85);filename.write_bytes(buffer.getvalue());return buffer.getvalue()

PAGE="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow Hand · 全量 grasp 抓取目标</title><style>body{font:15px system-ui;background:#111820;color:#e8eef5;margin:0;padding:24px}h1{font-size:25px}p{color:#afc1d2;line-height:1.6}select,button{padding:9px;background:#243747;border:1px solid #536c81;border-radius:6px;color:white;margin:4px}button:disabled{opacity:.4}#status{background:#1b3040;padding:14px;border-radius:8px;line-height:1.8}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:15px;margin:18px 0}article{border:1px solid #35485a;border-radius:8px;overflow:hidden;background:#1b2733}article img{width:100%;display:block;min-height:100px}article header,article footer{padding:10px;word-break:break-all}small{color:#adbdcc}</style>
<h1>Shadow Hand · 仅 grasp 抓取目标</h1>
<div id="status">读取全量进度…</div>
<p>每条是一个独立的原始 grasp 姿态。全量缓存按抓型保存，下方每页显示 12 条，并按需渲染；图片数量不等于 pose 总数。<br>已还原表示姿态、物体位姿及几何接触缓存完成；未进行动态稳定性或力闭合验证。原始 grasp 目标可能留有间隙，几何接触为 0 时也会保留。</p>
<label>数据来源 <select id="source"><option value="full">全量已缓存 pose（分页浏览）</option><option value="preview">跨 31 抓型的 100 条抽样预览</option></select></label>
<label>抓型 <select id="type"><option value="">全部已缓存抓型</option></select></label>
<button id="refresh">刷新列表</button><button id="random">随机一页</button><input id="goto" type="number" min="1" placeholder="跳转到第几条" style="width:125px;padding:9px"><button id="jump">跳转</button><div><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div><div id="grid"></div>
<script>let offset=0,total=0;const limit=12;let latest=null;const $=s=>document.querySelector(s);
async function readStatus(){const r=await fetch('/api/status');latest=await r.json();const f=latest.full;$('#status').textContent='全量'+(f.all_reconstructed?'静态还原已完成':'仍在处理中')+'：已缓存 '+Number(f.samples_cached||0).toLocaleString()+' 条 grasp · 错误 '+(f.errors||0)+' · 当前处理/已有抓型 '+Object.keys(latest.full_types).join(', ')+'。抽样预览 '+latest.preview_samples+' 条、'+Object.keys(latest.preview_types).length+' 个抓型。';}
function updateTypes(){const before=$('#type').value;$('#type').replaceChildren();const all=document.createElement('option');all.value='';all.textContent='全部已缓存抓型';$('#type').append(all);const types=$('#source').value==='full'?Object.fromEntries(Object.keys(latest.preview_types).map(t=>[t,latest.full_types[t]||0])):latest.preview_types;Object.entries(types).sort((a,b)=>parseInt(a[0])-parseInt(b[0])).forEach(([type,n])=>{const o=document.createElement('option');o.value=type;o.textContent=type+' · '+(n?n.toLocaleString()+' 条':'等待全量处理');o.disabled=!n;$('#type').append(o)});if(types[before])$('#type').value=before;}
async function load(){const q=new URLSearchParams({source:$('#source').value,type:$('#type').value,offset,limit});$('#page').textContent='读取…';const r=await fetch('/api/list?'+q);const data=await r.json();if(!r.ok){$('#page').textContent=data.error;return}total=data.total;$('#grid').replaceChildren();for(const row of data.rows){const a=document.createElement('article'),h=document.createElement('header'),img=document.createElement('img'),f=document.createElement('footer');h.textContent=row.grasp_type+' · '+row.case;img.loading='lazy';img.src=row.image;img.alt=row.grasp_type+' original grasp';f.textContent='物体 '+row.object_id+' · 几何接触 '+row.contacts+' · 缩放 '+row.scale.map(x=>x.toPrecision(4)).join(', ');a.append(h,img,f);$('#grid').append(a)}$('#page').textContent=(total?offset+1:0)+'–'+Math.min(offset+limit,total)+' / '+total.toLocaleString()+' 条已缓存 grasp';$('#prev').disabled=offset===0;$('#next').disabled=offset+limit>=total;}
$('#random').onclick=()=>{offset=Math.floor(Math.random()*Math.max(1,total-limit+1));load()};$('#jump').onclick=()=>{offset=Math.min(Math.max(0,total-1),Math.max(0,Number($('#goto').value||1)-1));load()};$('#source').onchange=()=>{offset=0;updateTypes();load()};$('#type').onchange=()=>{offset=0;load()};$('#refresh').onclick=async()=>{await readStatus();updateTypes();load()};$('#prev').onclick=()=>{offset=Math.max(0,offset-limit);load()};$('#next').onclick=()=>{offset+=limit;load()};(async()=>{await readStatus();updateTypes();load()})();setInterval(()=>readStatus().catch(console.error),15000);</script></html>"""

class Handler(BaseHTTPRequestHandler):
    def send(self,data,content='application/json; charset=utf-8',code=200):
        if isinstance(data,dict):data=json.dumps(data).encode()
        if isinstance(data,str):data=data.encode()
        self.send_response(code);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','max-age=3600' if content=='image/jpeg' else 'no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        try:
            parsed=urllib.parse.urlparse(self.path);q=urllib.parse.parse_qs(parsed.query);value=lambda k,default='':q.get(k,[default])[0]
            source=value('source','full')
            if source not in {'full','preview'}:raise ValueError('Invalid source')
            if parsed.path=='/':self.send(PAGE,'text/html; charset=utf-8')
            elif parsed.path=='/api/status':self.send(status())
            elif parsed.path=='/api/list':self.send(listing(source,value('type'),max(0,int(value('offset','0'))),min(48,max(1,int(value('limit','12'))))))
            elif parsed.path=='/image':self.send(RENDER_POOL.submit(render,source,value('type'),int(value('part','0')),int(value('row','0'))).result(timeout=120),'image/jpeg')
            else:self.send({'error':'Not found'},code=404)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as error:self.send({'error':str(error)},code=400)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bind',default='192.168.0.32');p.add_argument('--port',type=int,default=8098);a=p.parse_args()
    print('GRASP_VIEWER',a.bind,a.port,flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        RENDER_POOL=pool
        ThreadingHTTPServer((a.bind,a.port),Handler).serve_forever()
