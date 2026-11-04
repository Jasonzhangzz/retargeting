"""MANO result snapshots and independently cached repository closure certificates."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from dataclasses import asdict
import collections,errno,gzip,hashlib,json,threading,time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RESULTS=ROOT/'results/dexonomy_mano_current'
DEFAULT_CACHE=ROOT/'results/dexonomy_mano_results_viewer_20261103'

def retry(function,*args,**kwargs):
    for attempt in range(7):
        try:return function(*args,**kwargs)
        except OSError as error:
            if error.errno not in (errno.ESTALE,errno.EAGAIN,errno.EIO) or attempt==6:raise
            time.sleep(.05*2**attempt)

def read(path):return json.loads(retry(Path(path).read_text))
def sha(path):return hashlib.sha256(retry(Path(path).read_bytes)).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    retry(temporary.write_text,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    retry(temporary.replace,path)

class ManoSnapshot:
    def __init__(self,path):
        self.path=Path(path);self.manifest=read(self.path/'manifest.json')
        if self.manifest.get('hand')!='mano':raise ValueError('Expected solved MANO snapshot')
        if not read(self.path/'verification.json').get('success'):raise ValueError('Unverified MANO shard')
        with retry(np.load,self.path/'poses.npz',allow_pickle=False) as a:self.poses={k:a[k].copy() for k in a.files}
        with retry(np.load,self.path/'geometry.npz',allow_pickle=False) as a:self.geometry={k:a[k].copy() for k in a.files}
        self.metrics=read_metrics(str(self.path))
        self.templates=json.loads(gzip.decompress(retry((self.path/'templates.json.gz').read_bytes)))
        self._contacts=None;self._mesh_cache={};self._source_poses=None
        self.verification=read(self.path/'verification.json')
    def summary(self,index):return self.metrics[index]
    def template(self,index):return self.templates[self.manifest['cases'][index]['template_id']]
    def diagnostics(self,index):
        result=dict(self.metrics[index]['geometry_diagnostics'])
        result['projected_points']=self.poses['mapped_object_points'][index].tolist()
        return result
    def object_mesh(self,index,world=True):
        key='object_'+str(int(self.poses['object_geometry_ids'][index]))
        vertices=self.geometry[key+'_vertices'].astype(np.float64)*float(self.poses['scene_scale'][index])
        faces=self.geometry[key+'_faces'].copy()
        if world:
            pose=self.poses['object_pose_world'][index];rotation=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
            vertices=vertices@rotation.T+pose[:3]
        return vertices,faces
    def mano_mesh(self,index):
        if 'mano_vertices' in self.poses:return self.poses['mano_vertices'][index].copy(),self.geometry['mano_faces'].copy()
        if index not in self._mesh_cache:
            import torch
            info=read(self.path/'mano_model.json');model=mesh_decoder(info['model_path'],info['model_sha256'],info['pca_components'])
            variables=np.concatenate((self.poses['qpos'][index,:6],self.poses['mano_pca'][index]))
            with torch.no_grad():vertices=model(torch.as_tensor(variables[None],dtype=torch.float32),full=True)[0][0].numpy()
            self._mesh_cache[index]=vertices
        return self._mesh_cache[index].copy(),self.geometry['mano_faces'].copy()
    def source_pose(self,index):
        if 'source_shadow_grasp_qpos' in self.poses:return self.poses['source_shadow_grasp_qpos'][index]
        if self._source_poses is None:
            source=Path(self.manifest['cases'][index]['source_manifest']).parent/'poses.npz'
            with retry(np.load,source,allow_pickle=False) as a:self._source_poses=a['grasp_qpos'].copy()
        return self._source_poses[self.manifest['cases'][index]['source_cached_case']]
    def source_joints(self,index):
        if 'source_shadow_joints_world' in self.poses:return self.poses['source_shadow_joints_world'][index]
        from retarget.env.grasp.mano_solver import ShadowSkeleton
        return ShadowSkeleton(ROOT/'data/dexonomy/shadow_assets/right.xml')(self.source_pose(index)[None])[0]

    def contacts(self,index):
        if self._contacts is None:
            with retry(np.load,self.path/'contacts.npz',allow_pickle=False) as a:self._contacts={k:a[k].copy() for k in a.files}
        start,stop=map(int,self._contacts['offsets'][index:index+2])
        return {k:v[start:stop] for k,v in self._contacts.items() if k!='offsets' and not k.startswith('mapped_')}
    def row(self,index):
        return dict(snapshot=self,index=int(index),entry=self.manifest['cases'][index],metric=self.metrics[index],
            key=self.path.parents[1].name+'/'+self.path.name+'/'+str(index))

@lru_cache(maxsize=2)
def mesh_decoder(path,checksum,components):
    from retarget.env.grasp.mano_solver import ManoDecoder
    if sha(path)!=checksum:raise ValueError('MANO model checksum mismatch')
    return ManoDecoder(path,'cpu',components)

@lru_cache(maxsize=6)
def snapshot(path):return ManoSnapshot(path)
@lru_cache(maxsize=256)
def read_metrics(path):
    return json.loads(gzip.decompress(retry((Path(path)/'metrics.json.gz').read_bytes)))

class SolvedIndex:
    """Read only completed shards; track unfinished directories without loading all poses."""
    def __init__(self,root):
        self.root=Path(root);self.groups={};self.pending=set();self.mtimes={};self.lock=threading.RLock()
    def refresh(self):
        with self.lock:
            references=self.root/'reused_shards.json'
            if references.exists() and self.mtimes.get(references)!=retry(references.stat).st_mtime_ns:
                for item in read(references):
                    path=Path(item['path'])
                    self.groups[str(path)]=dict(path=str(path),type=path.parents[1].name,part=int(path.name),
                        samples=int(item['samples']),accepted=int(item.get('geometry_accepted',0)))
                self.mtimes[references]=retry(references.stat).st_mtime_ns
            for folder in sorted(self.root.glob('*/shards')):
                modified=retry(folder.stat).st_mtime_ns
                if self.mtimes.get(folder)!=modified:
                    self.mtimes[folder]=modified
                    for path in folder.iterdir():
                        if path.is_dir() and path.name.isdigit() and str(path) not in self.groups:self.pending.add(path)
            for path in list(self.pending):
                verification=path/'verification.json'
                if not verification.exists():continue
                try:v=read(verification)
                except (OSError,ValueError):continue
                if v.get('success'):
                    self.groups[str(path)]=dict(path=str(path),type=path.parents[1].name,part=int(path.name),
                        samples=int(v['samples']),accepted=int(v.get('geometry_accepted',0)))
                    self.pending.remove(path)
            return self
    def types(self):
        result=collections.Counter()
        with self.lock:
            for g in self.groups.values():result[g['type']]+=g['samples']
        return dict(sorted(result.items(),key=lambda x:int(x[0].split('_')[0])))
    def ordered(self,typ='全部'):
        with self.lock:groups=[g.copy() for g in self.groups.values() if typ=='全部' or g['type']==typ]
        return sorted(groups,key=lambda g:(int(g['type'].split('_')[0]),g['part']))
    def indices(self,group,quality):
        if quality=='全部':return range(group['samples'])
        if quality=='几何达标' and not group['accepted']:return []
        if quality=='几何未达标' and group['accepted']==group['samples']:return []
        desired=quality=='几何达标'
        return [i for i,m in enumerate(read_metrics(group['path'])) if bool(m.get('geometry_accepted',False))==desired]
    def page(self,typ='全部',quality='全部',offset=0,limit=100):
        rows=[];total=0
        for group in self.ordered(typ):
            ids=self.indices(group,quality);count=len(ids)
            start=max(0,offset-total);stop=min(count,offset+limit-total)
            if stop>start:
                rows.extend(dict(path=group['path'],index=int(ids[i]),type=group['type'],part=group['part']) for i in range(start,stop))
            total+=count
        return rows,total
    def status(self):
        try:s=read(self.root/'status.json')
        except (OSError,ValueError):s={}
        return dict(**s,viewer_available_samples=sum(g['samples'] for g in self.groups.values()),viewer_types=self.types())

class ClosureCache:
    """CPU-only on-demand certificates, with source pose and solver fingerprints."""
    def __init__(self,cache=DEFAULT_CACHE):
        self.root=Path(cache);self.lock=threading.RLock();self.models=None
        self.code_sha={name:sha(ROOT/'retarget/env/grasp'/name) for name in ('projected_qp.py','closure_types.py','dexonomy_replay.py')}
    def cache_path(self,row):
        s=row['snapshot'];e=row['entry']
        return self.root/'closure'/e['grasp_type']/s.path.name/(str(row['index']).zfill(5)+'.json')
    def fingerprint(self,row):
        s=row['snapshot'];i=row['index']
        digest=hashlib.sha256()
        for k in ('qpos','mapped_hand_points','mapped_object_points','mapped_object_normals','object_pose_world'):
            digest.update(np.asarray(s.poses[k][i]).tobytes())
        digest.update(json.dumps(self.code_sha,sort_keys=True).encode())
        return digest.hexdigest()
    def cached(self,row):
        path=self.cache_path(row)
        if path.exists():
            value=read(path)
            if value.get('fingerprint')==self.fingerprint(row):return value
        return None
    def evaluate(self,row,force=False):
        with self.lock:
            if not force:
                old=self.cached(row)
                if old is not None:return old
            import torch,mujoco
            from retarget.env.grasp.dexonomy_grasp_static import StaticModels
            from retarget.env.grasp.dexonomy_replay import closure
            torch.set_num_threads(1)
            s=row['snapshot'];i=row['index'];e=row['entry']
            if self.models is None:self.models=StaticModels(ROOT/'data/dexonomy',2,visual=False)
            record=dict(scene_path=e['scene_path'],scene_scale=np.array([e['scene_scale']]),grasp_qpos=np.zeros((1,29)))
            env,_,_=self.models.get(record,0);model,data=env._model,env._data
            data.qpos[:29]=s.source_pose(i);data.qpos[29:]=s.poses['object_pose_world'][i]
            mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
            projected=s.poses['mapped_object_points'][i];normal=s.poses['mapped_object_normals'][i]
            report,arrays=closure(model,data,env._obj_active_bodyid,dict(points=projected,object_points=projected,normals=normal))
            # Independently audit the saved certificate wrench witness.
            if 'closure_model' in report:
                m=report['closure_model'];f=arrays['closure_response_forces_n'];spin=arrays['closure_response_intrinsic_torques_nm']
                wrench=np.concatenate((f.sum(1),(np.cross(projected[None]-np.array(m['center']),f)+spin).sum(1)/m['radius_m']),-1)
                loads=np.concatenate((np.zeros((1,6)),np.eye(6),-np.eye(6)))*m['test_force_n']
                residual=np.linalg.norm((wrench-loads)/m['test_force_n'],axis=-1)
                report['audited_max_wrench_residual']=float(residual[1:].max())
                report['witness_audit_passed']=bool(not report['certified'] or residual[1:].max()<=m['certificate_tolerance']+1e-6)
                if not report['witness_audit_passed']:raise ValueError('Projection force witness failed independent audit')
            actual=s.contacts(i);points=actual['object_points_world'];normals=actual['object_normals_world']
            actual_report,actual_arrays=closure(model,data,env._obj_active_bodyid,dict(points=points,object_points=points,normals=normals))
            result=dict(fingerprint=self.fingerprint(row),pose_key=row['key'],projected=report,near_surface=actual_report,
                geometry_accepted=bool(row['metric']['geometry_accepted']),
                combined_geometry_and_projected=bool(row['metric']['geometry_accepted'] and report['certified']),
                contact_tolerance_m=.002,near_surface_semantics='MANO vertex proximity at 2 mm tolerance, not measured forces',
                object_physics_source='same released scaled coacd object as Shadow replay; aggregate mass/COM/inertia',
                dynamic_evaluated=False,source_pose_modified=False)
            path=self.cache_path(row);path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path.with_suffix('.npz'),**arrays,**{'near_'+k:v for k,v in actual_arrays.items()})
            write(path,result);return result
    def witness(self,row):
        path=self.cache_path(row).with_suffix('.npz')
        if not path.exists():return None
        with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
