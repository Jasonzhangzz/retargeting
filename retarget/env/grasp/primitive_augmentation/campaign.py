"""Restartable balanced generation; only independently screened rows commit."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import h5py
import numpy as np
from scipy.stats import qmc
import torch
from . import SCHEMA
from .geometry import carve,volume_points
from .templates import Templates,SOURCE,sha256,matrix,world_pose
from .solver import Solver,ObjectQuery
from .storage import write_json,save_geometry,load_geometry,save_template,write_shard,Dataset
from .coverage import Pool,metrics

MODES=('shape','noise','slide')


def quota(count):
    a=int(.4*count);b=int(.3*count)
    return dict(shape=a,noise=b,slide=count-a-b)


def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()


class Campaign:
    def __init__(self, root,source=SOURCE,per_type=50000,points=1024,seed=20261117,
                 device='cuda:0',batch=64,iterations=60,patch_samples=128,min_free_gib=5,baseline=200):
        if min(per_type,points,batch,iterations,patch_samples)<=0 or per_type<10:
            raise ValueError('Positive settings and at least 10 samples/type required')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'campaign.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.templates=Templates(source,patch_samples)
        self.solver=Solver(self.templates,device,iterations)
        self.batch=batch;self.points=points;self.target=per_type;self.quota=quota(per_type)
        self.reserve=min_free_gib;self.baseline=baseline
        code=sorted(Path(__file__).parent.glob('*.py'))
        from retarget.baseline.geort.adapter import ROOT
        code += [ROOT/p for p in ('retarget/env/grasp/pose_gpu.py','retarget/baseline/card/collision.py',
                  'moxian_web/wuji_surface.py','retarget/task/grasp/dexonomy.py','retarget/env/grasp/inputs.py',
                  'retarget/baseline/geort/pose_domain.py','retarget/env/grasp/hand.py')]
        self.config=dict(schema=SCHEMA,per_type=per_type,points=points,seed=seed,batch=batch,iterations=iterations,
                         source_root=str(Path(source).resolve()),
                         baseline_samples_per_type=baseline,quota=self.quota,identity=self.templates.identity(),
                         code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in code},
                         units=dict(position='m',joints='rad'),pose='qpos native GEORT 20 joints; base/object XYZ+WXYZ',
                         cloud_frame='object_local',max_object_penetration_m=.005,max_self_penetration_m=.005,
                         patch_acceptance='ceil(0.8*N) patches with a witness -0.005<d<=0',
                         slide_acceptance='-0.0001<=d<=0 and normal dot<=-cos(30deg)',
                         normal_target_distance_m=0.,numerical_inward_target_m=.00002,
                         sampling='Sobol global + continuous walks + shared sparse-pool proposals',
                         dynamic_stability_verified=False,continuous_collision_proof=False)
        path=self.root/'config.json'
        if path.exists():
            if json.loads(path.read_text())!=self.config:raise ValueError('Resume configuration/assets/code mismatch; use a new output directory')
        else:write_json(path,self.config)
        self.config_hash=fingerprint(self.config)
        self.rng=np.random.default_rng(seed);self.sobol=qmc.Sobol(20,scramble=True,seed=seed)
        self.pool=Pool(self.templates.lower,self.templates.upper)
        self.state=dict(phase='created',sequence=0,sobol_count=0,job=0,proposed=0,accepted=0,
                        counts={n:{m:0 for m in MODES} for n in self.templates.names},failures={},
                        row_counts={n:[0]*self.templates.entries[n]['samples'] for n in self.templates.names},
                        stalled={n:0 for n in self.templates.names},elapsed_seconds=0.)
        self.seen={n:set() for n in self.templates.names}
        self.started=time.monotonic();self.restore()

    def check_disk(self):
        if shutil.disk_usage(self.root).free<self.reserve*2**30:
            self.checkpoint('paused_disk');raise RuntimeError('Less than reserved free disk space; safely paused')

    def restore(self):
        path=self.root/'checkpoint.json'
        if path.exists():
            stored=json.loads(path.read_text())
            if stored['config_hash']!=self.config_hash:raise ValueError('Checkpoint identity mismatch')
            self.state=stored['state'];self.rng.bit_generator.state=stored['rng']
            self.pool.data=stored['pool']
        shards=sorted((self.root/'shards').glob('*/*.h5'))
        latest=None
        counts={n:{m:0 for m in MODES} for n in self.templates.names}
        rows={n:[0]*self.templates.entries[n]['samples'] for n in self.templates.names}
        rebuilt=Pool(self.templates.lower,self.templates.upper)
        for p in sorted(shards,key=lambda p:int(p.stem)):
            name=p.parent.name
            with h5py.File(p,'r') as f:
                if f.attrs['config_hash']!=self.config_hash:raise ValueError('Foreign shard')
                commit=json.loads(f.attrs['commit'])
                if latest is None or commit['sequence']>latest['sequence']:latest=commit
                q=f['qpos'][:];full=f['pose_object_local'][:];row=f['source_row'][:];gids=f['geometry_id'][:].astype(str)
                for value,mode,r in zip(q,f['augmentation'][:].astype(str),row):
                    key=self.pose_key(value)
                    if key in self.seen[name]:raise ValueError('Duplicate committed joint pose')
                    self.seen[name].add(key);counts[name][mode]+=1;rows[name][int(r)]+=1
                rebuilt.add(name,full,row,gids)
        if latest and latest['sequence']>=self.state['sequence']:
            self.state=latest['state'];self.state['sequence']=latest['sequence']+1
            self.rng.bit_generator.state=latest['rng'];self.pool=rebuilt
        # Counts always derive from committed shards, including the case where
        # an audit removed the last shard or a write failed before its rename.
        self.state['counts']=counts;self.state['row_counts']=rows
        self.state['accepted']=sum(sum(v.values()) for v in counts.values())
        self.pool=rebuilt
        if self.state['sobol_count']:self.sobol.fast_forward(self.state['sobol_count'])
        self.offset=self.state['elapsed_seconds']

    def checkpoint(self,phase=None):
        if phase:self.state['phase']=phase
        self.state['elapsed_seconds']=self.offset+time.monotonic()-self.started
        write_json(self.root/'checkpoint.json',dict(config_hash=self.config_hash,state=self.state,
                   rng=self.rng.bit_generator.state,pool=self.pool.data))
        write_json(self.root/'status.json',self.state)

    @staticmethod
    def pose_key(q):return np.rint(np.asarray(q,dtype=float)/1e-4).astype('<i4').tobytes()

    def failure(self,reason,count=1):
        d=self.state['failures'];d[reason]=d.get(reason,0)+int(count)

    def calibrate(self):
        self.checkpoint('calibrating')
        for name in self.templates.names:
            self.check_disk();t=self.templates.get(name);save_template(self.root,t,self.templates.surface)
            path=self.root/'calibration'/f'{name}.npz'
            if path.exists():continue
            path.parent.mkdir(parents=True,exist_ok=True)
            z=t['grid'];count=len(t['seeds'])
            poses=np.full((count,26),np.nan,np.float32);accepted=np.zeros(count,bool)
            hit=np.zeros(count,np.int32);od=np.full(count,np.nan);sd=od.copy()
            progress_path=path.with_suffix('.progress.npz');first_row=0
            if progress_path.exists():
                with np.load(progress_path,allow_pickle=False) as progress:
                    poses=progress['poses'];accepted=progress['accepted'];hit=progress['patch_hits']
                    od=progress['object_penetration_m'];sd=progress['self_penetration_m'];first_row=int(progress['next_row'])
                    self.rng.bit_generator.state=json.loads(str(progress['rng']))
            for start in range(first_row,count,max(1,self.batch//4)):
                self.check_disk()
                end=min(start+max(1,self.batch//4),count)
                seed_blocks=[];meshes=[]
                for row in range(start,end):
                    shape=carve(t['kind'],z['radius_m'][row],z['height_m'][row],self.rng,0)
                    seeds=np.repeat(t['seeds'][row:row+1],4,axis=0)
                    seeds[1:,6:]+=self.rng.normal(0,np.deg2rad(8),(3,20))
                    seeds[1:,:3]+=self.rng.normal(0,.002,(3,3))
                    seed_blocks.append(seeds);meshes.extend([shape.mesh]*4)
                obj=ObjectQuery(meshes,self.solver.device)
                solved=self.solver.solve(np.concatenate(seed_blocks),t,obj);audit=self.solver.audit(solved,t,obj)
                score=audit['patch_hit'].sum(1)
                for row in range(start,end):
                    offset=(row-start)*4;ids=np.flatnonzero(audit['accepted'][offset:offset+4])+offset
                    best=int(ids[np.argmax(score[ids])]) if len(ids) else int(score[offset:offset+4].argmax()+offset)
                    if len(ids):poses[row]=solved[best];accepted[row]=True
                    hit[row]=score[best];od[row]=audit['object_penetration_m'][best];sd[row]=audit['self_penetration_m'][best]
                tmp=progress_path.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,poses=poses,accepted=accepted,patch_hits=hit,
                    object_penetration_m=od,self_penetration_m=sd,next_row=end,rng=json.dumps(self.rng.bit_generator.state))
                os.replace(tmp,progress_path)
                print(f'CALIBRATE {name} {end}/{count} passed={accepted.sum()}',flush=True)
                self.state['calibration_current']=dict(type=name,row=end,total=count,accepted=int(accepted.sum()))
                self.checkpoint()
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,poses=poses,accepted=accepted,patch_hits=hit,
                                                       object_penetration_m=od,self_penetration_m=sd)
            os.replace(tmp,path)
            progress_path.unlink(missing_ok=True)
            print(f'CALIBRATED {name} {accepted.sum()}/{count}',flush=True)
        report={}
        for p in sorted((self.root/'calibration').glob('*.npz')):
            if p.name.endswith('.progress.npz'):continue
            with np.load(p,allow_pickle=False) as z:
                report[p.stem]=dict(sizes=len(z['accepted']),accepted_sizes=int(z['accepted'].sum()),
                                    failed_rows=np.flatnonzero(~z['accepted']).tolist())
        write_json(self.root/'calibration_report.json',report)
        self.checkpoint('calibrated')

    def calibration(self,name):
        with np.load(self.root/'calibration'/f'{name}.npz',allow_pickle=False) as z:
            return z['poses'].copy(),z['accepted'].copy()

    def choose(self,name,mode):
        t=self.templates.get(name);corrected,valid=self.calibration(name)
        entries=self.pool.data.get(name,[])
        # Unsuccessful calibration sizes still get explicit attempts; they are
        # not silently discarded from the proposed size distribution.
        if mode!='shape' and entries:
            parent=entries[int(self.rng.choice(len(entries),p=self.pool.weights(name)))]
            row=parent['row'];shape=load_geometry(self.root,parent['geometry'])
            initial=np.asarray(parent['pose']);gid=parent['geometry']
        else:
            rc=np.asarray(self.state['row_counts'][name]);weights=1/np.sqrt(1+rc)
            row=int(self.rng.choice(len(rc),p=weights/weights.sum()))
            initial=corrected[row] if valid[row] else t['seeds'][row]
            fraction=0. if self.rng.random()<.05 else self.rng.uniform(0,.5)
            shape=carve(t['kind'],t['grid']['radius_m'][row],t['grid']['height_m'][row],self.rng,fraction)
            gid=None
        return t,row,shape,initial,gid

    def propose(self,t,row,initial,mode):
        count=self.batch;seeds=np.repeat(initial[None],count,0)
        parents=seeds.copy();scheme=np.full(count,'local',dtype='<U16')
        entries=self.pool.data.get(t['name'],[])
        if mode=='noise':
            seeds[:,6:]+=self.rng.normal(0,np.deg2rad(3),(count,20))
            seeds[:,:3]+=self.rng.normal(0,.001,(count,3))
            seeds[:,3:6]+=self.rng.normal(0,np.deg2rad(1),(count,3))
        elif mode=='shape':
            # Stable proportions: 1/3 global, 1/3 continuous, 1/3 shared sparse.
            a=count//3;b=2*count//3
            points=self.sobol.random(a);self.state['sobol_count']+=a
            scheme[:a]='sobol';scheme[a:b]='walk';scheme[b:]='sparse'
            seeds[:a,6:]=self.templates.lower+points*(self.templates.upper-self.templates.lower)
            if entries:
                picks=self.rng.choice(len(entries),count-a,p=self.pool.weights(t['name']))
                for i,j in enumerate(picks,a):
                    parent=entries[int(j)]
                    seeds[i]=parent['pose'];parent_row=parent['row']
                    # Base is already in object-local coordinates; objects of
                    # different sizes share axes but never inherit a world offset.
                    if parent_row!=row:seeds[i,:3]*=t['grid']['radius_m'][row]/t['grid']['radius_m'][parent_row]
                    parents[i]=seeds[i]
            seeds[a:b,6:]+=self.rng.normal(0,np.deg2rad(10),(b-a,20))
            seeds[b:,6:]+=self.rng.normal(0,np.deg2rad(20),(count-b,20))
            seeds[a:,:3]+=self.rng.normal(0,.002,(count-a,3))
            # Cross-type joints are proposals only; target template contacts and
            # full geometry are solved and screened anew before assigning a label.
            shared=[p for n,ps in self.pool.data.items() if n!=t['name'] for p in ps]
            if shared:
                for i in range(b,count,2):
                    seeds[i,6:]=shared[int(self.rng.integers(len(shared)))]['pose'][6:];scheme[i]='cross_type'
        return seeds,parents,scheme

    def batch_job(self,name,mode):
        self.check_disk();self.state['job']+=1
        try:t,row,shape,initial,gid=self.choose(name,mode)
        except ValueError:
            self.failure('invalid_carved_geometry');self.state['stalled'][name]+=self.batch
            self.checkpoint();return 0
        obj=ObjectQuery(shape.mesh,self.solver.device)
        seeds,parents,scheme=self.propose(t,row,initial,mode);slide=None;slide_delta=np.zeros((len(seeds),3),np.float32)
        slide_parent=np.zeros((len(seeds),len(t['patches']),3),np.float32)
        if mode=='slide':
            w=self.solver.witnesses(self.solver.tensor(seeds),t,obj)
            anchors=w['hit'].cpu().numpy();normal=w['object_normal'].cpu().numpy()
            slide_parent=anchors.copy()
            direction=self.rng.normal(size=(len(seeds),3))
            direction/=np.linalg.norm(direction,axis=1,keepdims=True)
            length=self.rng.uniform(.001,.01,(len(seeds),1))
            tangent=direction[:,None]-np.sum(direction[:,None]*normal,axis=-1,keepdims=True)*normal
            tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-12)
            target=anchors+tangent*length[:,None]
            slide=obj.query(self.solver.tensor(target))[0].cpu().numpy()
            slide_delta=(slide-anchors).mean(1).astype(np.float32)
            seeds[:,:3]+=slide_delta
        solved=self.solver.solve(seeds,t,obj,slide=slide)
        audit=self.solver.audit(solved,t,obj,sliding=mode=='slide')
        if mode=='slide':
            displacement=np.linalg.norm(audit['contact_object_local']-slide_parent,axis=-1)
            progress=(displacement>=.0005)&(displacement<=.015)&audit['patch_hit']
            moved=progress.sum(1)>=math.ceil(.8*len(t['patches']))
            self.failure('slide_without_surface_progress',int((~moved).sum()))
            audit['accepted'] &= moved
        self.state['proposed']+=len(solved)
        for reason,mask in [('joint_domain',~audit['domain_valid']),('invalid_query',~audit['query_valid']),
                            ('patch_contacts',audit['patch_hit'].sum(1)<math.ceil(.8*len(t['patches']))),
                            ('object_penetration',audit['object_penetration_m']>.005),
                            ('self_penetration',audit['self_penetration_m']>.005)]:self.failure(reason,int(mask.sum()))
        keep=[];keys=set()
        for i in np.flatnonzero(audit['accepted']):
            key=self.pose_key(solved[i,6:])
            if key in self.seen[name] or key in keys:self.failure('duplicate_pose');continue
            keys.add(key);keep.append(i)
        remaining=self.quota[mode]-self.state['counts'][name][mode];keep=np.array(keep[:remaining],int)
        if not len(keep):
            self.state['stalled'][name]+=len(solved);self.checkpoint();return 0
        self.state['stalled'][name]=0
        gid=gid or save_geometry(self.root,shape)
        solved=solved[keep];base=np.stack([world_pose(q[:6],t['grid']['object_pose_world'][row]) for q in solved]).astype(np.float32)
        object_pose=t['grid']['object_pose_world'][row];rotation=matrix(object_pose)
        a={key:value[keep] for key,value in audit.items() if key!='accepted'}
        # The canonical geometry written to disk must be used for point labels.
        # Manifold output is float32 already, so this does not change the solid.
        cloud_rng_seed=self.rng.integers(0,2**63,size=len(keep),dtype=np.int64)
        clouds=np.stack([volume_points(shape,self.points,np.random.default_rng(int(s)),obj.contains) for s in cloud_rng_seed])
        local_ids=a.pop('witness_ids');face=np.empty_like(local_ids,dtype=np.int32);bary=np.empty((*local_ids.shape,3),np.float32)
        for j,patch in enumerate(t['patches']):
            face[:,j]=patch['face_ids'][local_ids[:,j]];bary[:,j]=patch['barycentric'][local_ids[:,j]]
        contact_world=a['contact_object_local']@rotation.T+object_pose[:3]
        normal_world=a['contact_normal_object_local']@rotation.T
        n=len(keep)
        arrays=dict(qpos=solved[:,6:].astype(np.float32),pose_object_local=solved.astype(np.float32),
                    target_base_pose=base,object_pose_world=np.repeat(object_pose[None],n,0),
                    grasp_type=np.full(n,t['type_id'],np.int16),source_row=np.full(n,row,np.int32),
                    geometry_id=np.full(n,gid),augmentation=np.full(n,mode),
                    requested_mask_fraction=np.full(n,shape.requested_fraction,np.float32),
                    mask_fraction=np.full(n,shape.removed_fraction,np.float32),volume_points_m=clouds,
                    cloud_seed=cloud_rng_seed,contact_owner=np.repeat(t['owners'][None],n,0),
                    contact_face_id=face,contact_barycentric=bary,contact_position_world=contact_world.astype(np.float32),
                    contact_normal_world=normal_world.astype(np.float32),slide_displacement_m=slide_delta[keep],
                    proposal_pose_object_local=seeds[keep].astype(np.float32),
                    parent_pose_object_local=parents[keep].astype(np.float32),proposal_scheme=scheme[keep],**a)
        arrays['slide_parent_surface_m']=slide_parent[keep]
        arrays['slide_target_surface_m']=slide[keep].astype(np.float32) if slide is not None else np.zeros_like(slide_parent[keep])
        sequence=self.state['sequence']
        # State saved inside each atomic shard allows roll-forward after a crash
        # between the shard rename and the separate checkpoint update.
        self.state['counts'][name][mode]+=n;self.state['accepted']+=n
        self.state['row_counts'][name][row]+=n
        self.pool.add(name,solved,np.full(n,row),[gid]*n)
        commit=dict(sequence=sequence,state=self.state.copy(),rng=self.rng.bit_generator.state)
        write_shard(self.root,name,sequence,arrays,commit,self.config_hash)
        self.state['sequence']+=1
        for value in solved[:,6:]:self.seen[name].add(self.pose_key(value))
        self.checkpoint()
        print(f'ACCEPT {name} {mode} +{n} total={sum(self.state["counts"][name].values())}/{self.target} global={self.state["accepted"]} proposals={self.state["proposed"]}',flush=True)
        return n

    def generate(self,max_jobs=None):
        self.calibrate()
        if self.baseline:self.generate_baseline()
        self.checkpoint('generating');jobs=0
        while True:
            pending=[n for n,c in self.state['counts'].items() if sum(c.values())<self.target]
            if not pending:break
            for name in pending:
                if self.state['stalled'][name]>=100000:
                    self.checkpoint('incomplete_stalled')
                    self.report();raise RuntimeError(f'{name}: 100,000 consecutive proposals without a new valid pose; constraints retained')
                counts=self.state['counts'][name]
                available=[m for m in MODES if counts[m]<self.quota[m]]
                # Bootstrap shapes before noise/slide; then keep quotas balanced.
                mode=min(available,key=lambda m:counts[m]/max(self.quota[m],1))
                self.batch_job(name,mode);jobs+=1
                if max_jobs and jobs>=max_jobs:
                    self.checkpoint('paused_job_budget');self.report();return
        self.checkpoint('auditing');self.audit(repair=True)
        if any(sum(v.values())<self.target for v in self.state['counts'].values()):
            return self.generate(max_jobs)
        self.checkpoint('complete');self.report()

    def generate_baseline(self):
        """Separate valid near-template Gaussian baseline; never counts to quota."""
        folder=self.root/'baseline';folder.mkdir(exist_ok=True)
        for name in self.templates.names:
            path=folder/f'{name}.npz'
            if path.exists():continue
            t=self.templates.get(name);poses,valid=self.calibration(name);ids=np.flatnonzero(valid)
            collected=[];proposed=0
            # Diagnostic baseline uses an independent deterministic RNG stream.
            rng=np.random.default_rng(self.config['seed']+t['type_id']*1009)
            while sum(len(q) for q in collected)<self.baseline and proposed<self.baseline*40 and len(ids):
                row=int(rng.choice(ids));shape=carve(t['kind'],t['grid']['radius_m'][row],t['grid']['height_m'][row],rng,0)
                obj=ObjectQuery(shape.mesh,self.solver.device);seed=np.repeat(poses[row:row+1],self.batch,0)
                seed[:,6:]+=rng.normal(0,np.deg2rad(3),(self.batch,20))
                q=self.solver.solve(seed,t,obj);a=self.solver.audit(q,t,obj)
                collected.append(q[a['accepted'],6:]);proposed+=len(q)
            values=np.concatenate(collected)[:self.baseline] if collected else np.empty((0,20),np.float32)
            tmp=path.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,qpos=values,proposed=proposed,grasp_type=t['type_id'])
            os.replace(tmp,path)
            print(f'BASELINE {name} {len(values)}/{self.baseline}',flush=True)

    def audit(self,repair=False):
        report=dict(checked=0,invalid=0,files={},sampling_method='native vertices+face-centres, winding signs',
                    continuous_intersection_proof=False)
        for path in sorted((self.root/'shards').glob('*/*.h5')):
            self.check_disk();name=path.parent.name;t=self.templates.get(name)
            with h5py.File(path,'r') as f:
                arrays={k:f[k][:] for k in f};attrs=dict(f.attrs)
            keep=np.ones(len(arrays['qpos']),bool)
            gids=arrays['geometry_id'].astype(str)
            for gid in np.unique(gids):
                ids=np.flatnonzero(gids==gid);shape=load_geometry(self.root,gid);obj=ObjectQuery(shape.mesh,self.solver.device)
                for mode in np.unique(arrays['augmentation'][ids]):
                    mode_ids=ids[arrays['augmentation'][ids]==mode]
                    for start in range(0,len(mode_ids),self.batch):
                        ix=mode_ids[start:start+self.batch]
                        a=self.solver.audit(arrays['pose_object_local'][ix],t,obj,sliding=mode==b'slide')
                        inside=np.array([obj.contains(cloud).all() for cloud in arrays['volume_points_m'][ix]])
                        keep[ix]=a['accepted']&inside
                        # Stored hand-face correspondence is checked against the
                        # frozen hand triangles, independently of witness search.
                        for j,owner in enumerate(t['owners']):
                            mesh=self.templates.surface.meshes[owner]
                            bary=arrays['contact_barycentric'][ix,j]
                            face=arrays['contact_face_id'][ix,j]
                            local=(mesh.triangles[face]*bary[...,None]).sum(1)
                            keep[ix] &= np.linalg.norm(local-arrays['contact_local'][ix,j],axis=1)<1e-6
                        q=self.solver.tensor(arrays['pose_object_local'][ix])
                        pp,rr,_,_=self.solver.frames(q)
                        owners=arrays['contact_owner'][ix[0]]
                        bodies=np.array(self.solver.body_ids)[owners]
                        local=self.solver.tensor(arrays['contact_local'][ix])
                        points=pp[:,bodies]+torch.einsum('bkij,bkj->bki',rr[:,bodies],local)
                        _,normal,gap=obj.query(points)
                        gap=gap.cpu().numpy()
                        stored_hit=(gap>-.005)&(gap<=0)
                        if mode==b'slide':
                            # Stored normals are transformed back from world to
                            # the object's query frame before opposing-normal checks.
                            normals=np.stack([v@matrix(pose) for v,pose in zip(
                                arrays['contact_normal_world'][ix],arrays['object_pose_world'][ix])])
                            dot=(normals*normal.cpu().numpy()).sum(-1)
                            stored_hit &= (gap>=-.0001)&(dot<=-math.cos(math.pi/6))
                        keep[ix] &= stored_hit.sum(1)>=math.ceil(.8*len(t['patches']))
                        if mode==b'slide':
                            displacement=np.linalg.norm(points.cpu().numpy()-arrays['slide_parent_surface_m'][ix],axis=-1)
                            moved=(displacement>=.0005)&(displacement<=.015)&stored_hit
                            keep[ix] &= moved.sum(1)>=math.ceil(.8*len(t['patches']))
                        keep[ix] &= np.max(np.abs(gap-arrays['contact_gap_m'][ix]),axis=1)<1e-6
                        for k,id in enumerate(ix):
                            objpose=arrays['object_pose_world'][id]
                            expected=world_pose(arrays['pose_object_local'][id,:6],objpose)
                            actual=arrays['target_base_pose'][id]
                            contact=points[k].cpu().numpy()@matrix(objpose).T+objpose[:3]
                            keep[id] &= np.linalg.norm(expected[:3]-actual[:3])<1e-6
                            keep[id] &= np.max(abs(matrix(expected)-matrix(actual)))<1e-5
                            keep[id] &= np.max(abs(arrays['qpos'][id]-arrays['pose_object_local'][id,6:]))<1e-7
                            keep[id] &= np.max(abs(contact-arrays['contact_position_world'][id]))<1e-6
            report['checked']+=len(keep);report['invalid']+=int((~keep).sum())
            report['files'][str(path.relative_to(self.root))]=dict(samples=len(keep),invalid=int((~keep).sum()))
            if repair and not keep.all():
                arrays={k:v[keep] for k,v in arrays.items()}
                if len(arrays['qpos']):
                    temp=path.with_suffix('.audit.tmp')
                    with h5py.File(temp,'w') as f:
                        for k,v in attrs.items():f.attrs[k]=v
                        for k,v in arrays.items():f.create_dataset(k,data=v,compression='gzip',compression_opts=1,shuffle=True)
                    os.replace(temp,path)
                else:path.unlink()
        write_json(self.root/'audit_report.json',report)
        if repair and report['invalid']:
            # Rebuild acceptance counts/pools/dedup from the retained rows.
            self.pool=Pool(self.templates.lower,self.templates.upper)
            self.seen={n:set() for n in self.templates.names}
            self.restore();self.checkpoint('refilling_after_audit')
        return report

    def report(self):
        rng=np.random.default_rng(81);values=[];labels=[];baseline=[];base_labels=[];distribution={}
        for name in self.templates.names:
            blocks=[];masks=[];rows=[]
            for path in sorted((self.root/'shards'/name).glob('*.h5')):
                with h5py.File(path,'r') as f:
                    blocks.append(f['qpos'][:]);masks.extend(f['mask_fraction'][:].tolist());rows.extend(f['source_row'][:].tolist())
            if blocks:
                q=np.concatenate(blocks);n=min(200,len(q));ids=rng.choice(len(q),n,replace=False)
                values.append(q[ids]);labels.extend([int(name.split('_')[0])]*n)
            distribution[name]=dict(accepted=sum(self.state['counts'][name].values()),by_augmentation=self.state['counts'][name],
                size_counts=self.state['row_counts'][name],mask_histogram=np.histogram(masks,bins=10,range=(0,.5))[0].tolist(),
                unique_joint_poses=len(self.seen[name]))
            path=self.root/'baseline'/f'{name}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    baseline.append(z['qpos']);base_labels.extend([int(name.split('_')[0])]*len(z['qpos']))
        q=np.concatenate(values) if values else np.empty((0,20));bq=np.concatenate(baseline) if baseline else np.empty((0,20))
        report=dict(state=self.state['phase'],target=31*self.target,accepted=self.state['accepted'],types=distribution,
            coverage=metrics(q,labels,self.templates.lower,self.templates.upper),
            gaussian_baseline=metrics(bq,base_labels,self.templates.lower,self.templates.upper),
            failures=self.state['failures'],elapsed_seconds=self.state['elapsed_seconds'],
            completion_requires_full_audit=True)
        write_json(self.root/'report.json',report)
        ds=Dataset(self.root)
        write_json(self.root/'manifest.json',dict(schema=SCHEMA,complete=self.state['phase']=='complete',
                   config_hash=self.config_hash,accepted=self.state['accepted'],target=31*self.target,files=ds.manifest()))
        return report
