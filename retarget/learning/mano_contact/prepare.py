"""Read-only source adapters; immutable snapshot and resumable contact corpus."""
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import io
import json
import pickle
import tarfile
import zipfile
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .data import CorpusWriter, digest, write_json, audit_corpus, finalize_corpus, split_for
from .geometry import ManoGeometry, MeshDistance, mirror_pose


def snapshot_generated(root, output):
    root=Path(root); target=Path(output)
    if target.exists():
        return json.loads(target.read_text())
    sources=[]; seen=set(); model_hashes={}
    for name in ['mano_personalized_cem_20261105','mano_personalized_cem_20261105_b2048','mano50_wuji_success_20261108']:
        folder=root/'results'/name
        if not folder.exists():continue
        for receipt in sorted(folder.rglob('verification.json')):
            v=json.loads(receipt.read_text())
            if not v.get('success'):continue
            for relative, sha in v.get('files',{}).items():
                if not relative.endswith('final.npz'):continue
                path=receipt.parent/relative
                if sha in seen:continue
                seen.add(sha)
                variant='hand_id' in v.get('job',{})
                hand=v['job']['hand_id'] if variant else -1
                model=root/'output/mano50_20261108/models'/f'mano{hand}'/'MANO_RIGHT.pkl' if variant else root/'data/grasp/assets/mano/MANO_RIGHT.pkl'
                if str(model) not in model_hashes:model_hashes[str(model)]=digest(model)
                sources.append(dict(path=str(path),sha256=sha,receipt=str(receipt),variant=variant,
                                    hand_id=hand,model=str(model),model_sha256=model_hashes[str(model)],input_rows=v['samples']))
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),root=str(root),sources=sources,
               policy='Only successful committed verification receipts; derived aligned H5 and seeds are not counted again')
    write_json(target,value)
    return value


def load_labels(path):
    labels={}
    if not Path(path).exists():return labels
    with tarfile.open(path,'r|gz') as t:
        for m in t:
            if m.isfile() and '/contact_data/' in '/'+m.name and m.name.endswith('.npy'):
                a=np.load(io.BytesIO(t.extractfile(m).read()),allow_pickle=False)
                if a.shape!=(778,) or not np.isin(a,[0,1]).all():raise ValueError('Invalid HACO labels')
                labels[Path(m.name).stem]=a.astype(np.uint8)
    return labels


def _physical(model, q, beta, convention='mano'):
    vertices=[]; poses=[]
    for start in range(0,len(q),256):
        v,p=model(q[start:start+256],None if beta is None else beta[start:start+256],convention)
        vertices.append(v);poses.append(p)
    return np.concatenate(vertices),np.concatenate(poses)


def _world(pose, camera):
    out=pose.copy();out[:,:3]=np.einsum('bij,bj->bi',camera[:,:3,:3],pose[:,:3])+camera[:,:3,3]
    out[:,3:6]=Rotation.from_matrix(camera[:,:3,:3]@Rotation.from_rotvec(pose[:,3:6]).as_matrix()).as_rotvec()
    return out


def _hoi4d_pose(row):
    values=[np.asarray(row[k],np.float32).reshape(-1) for k in ['trans','poseCoeff','beta']]
    if [len(v) for v in values]!=[3,48,10] or not all(np.isfinite(v).all() for v in values):
        return None
    return np.concatenate(values[:2]),values[2]


def prepare_hoi4d(dataset, assets, writer, max_sequences=0):
    path=Path(dataset)/'HOI4D/HOI4D_Hand_pose.zip'
    if not path.exists():return
    labels=load_labels(Path(dataset)/'MANO_contact/haco_labels/HOI4D.tar.gz')
    models={s:ManoGeometry(Path(assets)/f'MANO_{s.upper()}.pkl') for s in ['right','left']}
    with zipfile.ZipFile(path) as z:
        groups=defaultdict(list)
        for name in z.namelist():
            if name.endswith('.pickle'):
                parts=Path(name).parts;side='left' if 'left_hand' in parts[1] else 'right'
                groups[(side,'/'.join(parts[2:-1]))].append(name)
        ordered=sorted(groups.items(),key=lambda item:(item[0][0]=='left',item[0][1]))
        for number,((side,sequence),members) in enumerate(ordered):
            if max_sequences and number>=max_sequences:break
            identity='HOI4D:'+side+':'+sequence
            if writer.contains(identity):continue
            members.sort(key=lambda n:int(Path(n).stem))
            accepted=[];rejected=[];raw_rows=len(members)
            for member in members:
                parsed=_hoi4d_pose(pickle.loads(z.read(member)))
                if parsed is None:rejected.append(int(Path(member).stem))
                else:accepted.append((member,parsed))
            if not accepted:
                print('HOI4D empty invalid sequence',sequence,flush=True);continue
            members=[v[0] for v in accepted]
            q=np.stack([v[1][0] for v in accepted]);beta=np.stack([v[1][1] for v in accepted])
            _,physical=_physical(models[side],q,beta)
            if side=='left':physical=mirror_pose(physical)
            ids=[f"{sequence.replace('/','-')}-{int(Path(n).stem):05d}" for n in members]
            # HACO's HOI4D contact release is right-hand only.
            valid=np.array([side=='right' and k in labels for k in ids])
            contact=np.stack([labels[k] if v else np.zeros(778,np.uint8) for k,v in zip(ids,valid)])
            writer.add(identity,physical,contact,valid,frames=[int(Path(n).stem) for n in members],
                       metadata=dict(dataset='HOI4D',kind='temporal',group_id='HOI4D:'+sequence,
                                     source=str(path),side=side,coordinate_frame='camera',fps=30.,label_source='HACO:10mm',
                                     source_record_rows=raw_rows,rejected_pose_rows=len(rejected),rejected_frame_ids=json.dumps(rejected)),
                       beta=beta)
            if number%100==0:print('HOI4D',number,len(groups),flush=True)


def prepare_h2o(dataset, assets, writer, max_sequences=0):
    """Use official cam4 pose packages, not the duplicate manolabel bundle."""
    dataset=Path(dataset); labels=load_labels(dataset/'MANO_contact/haco_labels/H2O.tar.gz')
    models={s:ManoGeometry(Path(assets)/f'MANO_{s.upper()}.pkl') for s in ['right','left']}
    objects={}; object_archive=dataset/'H2O/object.zip'
    if object_archive.exists():
        import trimesh
        with zipfile.ZipFile(object_archive) as z:
            for i,name in enumerate(['book','espresso','lotion','spray','milk','cocoa','chips','cappuccino'],1):
                candidates=[n for n in z.namelist() if Path(n).parent.name==name and n.endswith('.obj')]
                if len(candidates)!=1:raise ValueError(f'Ambiguous H2O object mesh {name}: {candidates}')
                mesh=trimesh.load(io.BytesIO(z.read(candidates[0])),file_type='obj',process=False,force='mesh')
                objects[i]=MeshDistance(mesh.vertices,mesh.faces)
    count=0
    for path in sorted((dataset/'H2O').glob('subject*_pose_v1_1.tar.gz')):
        groups=defaultdict(lambda:defaultdict(dict))
        with tarfile.open(path,'r|gz') as t:
            for m in t:
                if not m.isfile() or not m.name.endswith('.txt'):continue
                parts=Path(m.name).parts
                if len(parts)<4 or parts[-3]!='cam4' or parts[-2].lower() not in ('hand_pose_mano','cam_pose','obj_pose_rt'):continue
                groups['/'.join(parts[:-2])][parts[-2].lower()][int(Path(m.name).stem)]=np.fromstring(t.extractfile(m).read().decode(),sep=' ')
        for sequence,pack in sorted(groups.items()):
            if max_sequences and count>=max_sequences:return
            count+=1
            for hand,side in [(0,'left'),(1,'right')]:
                identity='H2O:'+side+':'+sequence
                if writer.contains(identity):continue
                frames=sorted(k for k,v in pack['hand_pose_mano'].items() if len(v)==124 and v[hand*62]>.5)
                if not frames:continue
                raw=np.stack([pack['hand_pose_mano'][i][hand*62:(hand+1)*62] for i in frames])
                q=raw[:,1:52].astype(np.float32);beta=raw[:,52:62].astype(np.float32)
                vertices,physical=_physical(models[side],q,beta)
                contact=np.zeros((len(q),778),np.uint8);valid=np.zeros(len(q),bool)
                penetration=np.full(len(q),np.nan,np.float32)
                for row,frame in enumerate(frames):
                    fields=sequence.split('/');fields[0]+='_ego'
                    sample=f"{'-'.join(fields)}-{frame:06d}"
                    if side=='right' and sample in labels:
                        contact[row]=labels[sample];valid[row]=True
                    else:
                        obj=pack['obj_pose_rt'].get(frame)
                        if obj is not None and len(obj)==17 and int(obj[0]) in objects:
                            transform=obj[1:].reshape(4,4)
                            local=(vertices[row]-transform[:3,3])@transform[:3,:3]
                            distance=objects[int(obj[0])].within(local,.01)
                            contact[row]=distance<.01;valid[row]=True
                cameras=np.stack([pack['cam_pose'].get(i,np.eye(4).ravel()).reshape(4,4) for i in frames])
                # If extrinsics are missing, do not fabricate world motion.
                all_cameras=all(i in pack['cam_pose'] for i in frames)
                if all_cameras:physical=_world(physical,cameras)
                if side=='left':physical=mirror_pose(physical)
                writer.add(identity,physical,contact,valid,frames=frames,
                           metadata=dict(dataset='H2O',kind='temporal',group_id='H2O:'+sequence.rsplit('/',1)[0],
                                         source=str(path),side=side,fps=30.,coordinate_frame='world' if all_cameras else 'camera',label_source='HACO_or_geometry:10mm'),beta=beta,penetration_depth_m=penetration)
                print('H2O',sequence,side,len(q),'known',int(valid.sum()),'negative',int((valid&~contact.any(-1)).sum()),flush=True)


def _generated_shard(job):
    snapshot,path=job
    torch.set_num_threads(1)
    with CorpusWriter(path) as writer:
        prepare_generated(snapshot,writer,workers=1)
    return str(path)


def merge_generated_shard(path,writer):
    """Copy only fully processed sources; preserve interrupted source shards."""
    path=Path(path)
    with h5py.File(path,'r') as source:
        for key,g in source['records'].items():
            prefix=':'.join(g.attrs['source_id'].split(':')[:3])
            if '__source_'+CorpusWriter.key(prefix) not in source.attrs:continue
            if key in writer.file['records']:continue
            source.copy(g,writer.file,name='__pending')
            writer.file.move('__pending','records/'+key)
        for name,value in source.attrs.items():
            if name.startswith('__source_'):writer.file.attrs[name]=value
        writer.file.flush()
    print('merged generated shard',path.name,flush=True)
    path.unlink()


def prepare_generated(snapshot, writer, max_sources=0, workers=1):
    if workers>1:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        import multiprocessing
        import hashlib
        cache=writer.path.parent/'generated_shards';cache.mkdir(exist_ok=True)
        # Import cleanly closed partial work before redistributing sources.
        for path in sorted(cache.glob('*.h5')):merge_generated_shard(path,writer)
        pending=[]
        for s in snapshot['sources'][:max_sources or None]:
            prefix='generated:'+s['sha256']+':'+s['model_sha256']
            if '__source_'+CorpusWriter.key(prefix) not in writer.file.attrs:pending.append(s)
        jobs=[]
        for i in range(min(workers,len(pending))):
            sources=pending[i::workers]
            identity=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:20]
            jobs.append((dict(snapshot,sources=sources),cache/(identity+'.h5')))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(_generated_shard,j) for j in jobs]):
                path=Path(future.result())
                merge_generated_shard(path,writer)
        return
    root=Path(snapshot['root']); model_cache={}; mesh_cache=OrderedDict();model_identities={}
    from scripts.mano_campaign_inputs import GeometryLoader
    from .geometry_cache import GeometryCache
    cache_root=writer.path.parent.parent if writer.path.parent.name=='generated_shards' else writer.path.parent
    loader=GeometryCache(GeometryLoader(root/'data/dexonomy'),cache_root/'geometry_cache')
    for number,source in enumerate(snapshot['sources']):
        if max_sources and number>=max_sources:break
        prefix='generated:'+source['sha256']+':'+source['model_sha256']
        # A batch receipt is recorded only after all its object groups commit.
        done='__source_'+CorpusWriter.key(prefix)
        if done in writer.file.attrs:continue
        path=Path(source['path'])
        stat=Path(source['model']).stat();identity=(stat.st_size,stat.st_mtime_ns)
        if source['model'] not in model_identities:
            if digest(source['model'])!=source['model_sha256']:raise ValueError('Frozen model changed')
            model_identities[source['model']]=identity
        if digest(path)!=source['sha256'] or model_identities[source['model']]!=identity:
            raise ValueError('Frozen generated source changed: '+str(path))
        if source['model'] not in model_cache:model_cache[source['model']]=ManoGeometry(source['model'])
        model=model_cache[source['model']]
        with np.load(path,allow_pickle=False) as f:
            a={k:f[k] for k in f.files if k in ['qpos','valid','rollout_valid','rollout_stable','object_scene','object_pose_world','scene_scale','penetration_depth_m','mano_joints','base_pair_id']}
        q=a['qpos'];n=len(q)
        keep=np.isfinite(q).all(-1)
        for key in ['valid','rollout_valid','rollout_stable']:
            if key in a:keep &= a[key]
        rows=np.flatnonzero(keep)
        if not len(rows):
            writer.file.attrs[done]=True;writer.file.flush();continue
        if source['variant']:
            scenes=[v.decode() for v in a['object_scene']]
        else:
            manifest=json.loads((Path(source['receipt']).parent/'MANIFEST.json').read_text())
            cases=manifest['source']['cases'];offset=manifest['job']['offset']
            if len(cases)!=n:cases=cases[offset:offset+n]
            if len(cases)!=n:raise ValueError('Source cases do not align')
            scenes=[c['scene_path'] for c in cases]
        groups=defaultdict(list)
        for i in rows:groups[scenes[i]].append(i)
        all_vertices,all_physical=_physical(model,q[rows],None,'origin')
        if 'mano_joints' in a:
            error=np.max(np.abs(all_physical[:,:3]-a['mano_joints'][rows,0]))
            if error>1e-4:raise ValueError(f'Wrist FK mismatch {error}: {path}')
        lookup=np.zeros(n,np.int64);lookup[rows]=np.arange(len(rows))
        contacts=np.zeros((len(rows),778),np.uint8);minimum=np.zeros(len(rows),np.float32)
        splits=np.empty(len(rows),dtype='U5');leakage_ids=np.empty(len(rows),dtype='S64')
        for scene,ids in groups.items():
            ids=np.asarray(ids);positions=lookup[ids];vertices=all_vertices[positions]
            key,geo=loader.get(dict(scene_path=scene))
            if key not in mesh_cache:
                mesh_cache[key]=MeshDistance(geo['vertices'],geo['faces'])
                if len(mesh_cache)>128:mesh_cache.popitem(last=False)
            mesh=mesh_cache[key]
            mesh_cache.move_to_end(key)
            obj=a['object_pose_world'][ids]
            rotation=Rotation.from_quat(obj[:,[4,5,6,3]]).as_matrix()
            scales=a.get('scene_scale',np.ones(n))[ids]
            local=np.einsum('bvi,bij->bvj',vertices-obj[:,None,:3],rotation)/scales[:,None,None]
            distance=mesh.within(local,.01/scales[:,None])*scales[:,None]
            family=scene.split('/scene_cfg/')[-1].split('/')[0]
            group_id='Dexonomy-object:'+family
            import hashlib
            contacts[positions]=distance<.01
            closest=distance.min(-1);minimum[positions]=np.where(closest<.01,closest,np.nan)
            splits[positions]=split_for(group_id);leakage_ids[positions]=hashlib.sha256(group_id.encode()).hexdigest().encode()
        for split in ['train','val','test']:
            selected=np.flatnonzero(splits==split)
            if not len(selected):continue
            ids=rows[selected]
            writer.add(prefix+':'+split,all_physical[selected],contacts[selected],
                       metadata=dict(dataset='generated',kind='static',group_id=prefix+':'+split,split=split,
                                     source=str(path),model=source['model'],model_sha256=source['model_sha256'],
                                     label_source='exact_triangle_geometry:10mm',hand_id=source['hand_id'],min_distance_censored_at_m=.01),
                       source_row=ids,penetration_depth_m=a.get('penetration_depth_m',np.full((n,2),np.nan))[ids],
                       min_surface_distance_m=minimum[selected],object_pose_world=a['object_pose_world'][ids],leakage_id=leakage_ids[selected])
        writer.file.attrs[done]=True;writer.file.flush()
        if number%10==0:print('generated',number,len(snapshot['sources']),len(rows),flush=True)


def prepare_primitives(source_root,writer,max_sources=0):
    """Use saved personalized MANO vertices; aligned paired H5 is a derivative."""
    root=Path(source_root)/'output/hf_publish_20261106/primitives'
    if not (root/'manifest.json').exists():return
    from retarget.env.grasp.primitives import primitive_geometry
    manifest=json.loads((root/'manifest.json').read_text())
    geometry_sha=manifest['mano']['geometry_sha256']
    if digest(root/'mano_geometry.npz')!=geometry_sha:raise ValueError('Primitive hand geometry changed')
    for number,entry in enumerate(manifest['entries']):
        if max_sources and number>=max_sources:break
        identity='primitive:'+entry['sha256'];path=root/entry['file']
        if writer.contains(identity):continue
        if digest(path)!=entry['sha256']:raise ValueError('Primitive pose source changed')
        with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
        q=np.concatenate((a['mano_variables'][:,:6],a['mano_angles'].reshape(-1,45)),axis=1).astype(np.float32)
        keep=np.isfinite(q).all(-1)&np.isfinite(a['mano_vertices']).all((1,2))
        for flag in ['valid','rollout_valid','rollout_stable']:keep &= a[flag]
        rows=np.flatnonzero(keep)
        if not len(rows):continue
        q=q[rows];q[:,:3]=a['mano_joints'][rows,0]
        vertices=a['mano_vertices'][rows]
        tip_error=float(np.max(np.abs(vertices[:,[744,320,443,554,671]]-a['mano_joints'][rows][:,[4,8,12,16,20]])))
        if tip_error>1e-4:raise ValueError('Saved primitive MANO vertex/joint mismatch')
        labels=[]
        for index,row in enumerate(rows):
            v,f=primitive_geometry(entry['kind'],float(a['radius_m'][row]),float(a['height_m'][row]))
            mesh=MeshDistance(v,f);obj=a['object_pose_world'][row]
            rotation=Rotation.from_quat(obj[[4,5,6,3]]).as_matrix()
            local=(vertices[index]-obj[:3])@rotation
            labels.append(mesh.within(local,.01)<.01)
        writer.add(identity,q,np.asarray(labels),metadata=dict(dataset='generated_primitives',kind='static',
                   group_id='primitive-object-family:'+entry['kind'],source=str(path),model_sha256=geometry_sha,
                   label_source='saved_MANO_vertices_exact_primitive_triangles:10mm',saved_tip_error_m=tip_error),
                   source_row=rows,penetration_depth_m=a['penetration_depth_m'][rows],
                   object_pose_world=a['object_pose_world'][rows],radius_m=a['radius_m'][rows],height_m=a['height_m'][rows])
        print('primitives',entry['type'],len(rows),flush=True)


def _dexycb_frame(path):
    # Independent small-file reads overlap CPFS latency; the writer stays single.
    with np.load(path,allow_pickle=False) as f:
        pose=f['pose_m'][0];wrist=f['joint_3d'][0,0]
    if pose.shape!=(51,) or not np.isfinite(pose).all() or not np.any(pose):return None
    if not np.isfinite(wrist).all() or np.all(wrist==-1) or not np.any(wrist):return None
    return int(path.stem.split('_')[-1]),np.r_[pose[48:51],pose[:48]],wrist


def prepare_dexycb(dataset, assets, writer, max_sequences=0):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as pool:
        _prepare_dexycb(dataset,assets,writer,max_sequences,pool)


def _prepare_dexycb(dataset, assets, writer, max_sequences, pool):
    """Deduplicate camera views using camera-free leakage IDs; labels stay local."""
    root=Path(dataset)/'DexYCB/extracted'
    if not root.exists():return
    import yaml
    labels=load_labels(Path(dataset)/'MANO_contact/haco_labels/DexYCB.tar.gz')
    models={s:ManoGeometry(Path(assets)/f'MANO_{s.upper()}.pkl') for s in ['right','left']}
    calibration={}
    for base in [root/'calibration',root.parent/'calibration',Path(__file__).resolve().parents[3]/'data/dexycb/raw/official/calibration']:
        if base.exists():
            for p in base.glob('mano_*/mano.yml'):calibration[p.parent.name]=p
    count=0
    for meta_path in sorted(root.glob('*/*/meta.yml')):
        if max_sequences and count>=max_sequences:break
        meta=yaml.safe_load(meta_path.read_text());side=meta.get('mano_sides',['right'])[0]
        sequence=meta_path.parent; group='DexYCB:'+str(sequence.relative_to(root))
        beta_path=calibration.get('mano_'+meta['mano_calib'][0])
        beta=None if beta_path is None else np.asarray(yaml.safe_load(beta_path.read_text())['betas'],np.float32)
        for camera in sorted(p for p in sequence.iterdir() if p.is_dir()):
            identity=group+':'+camera.name
            if writer.contains(identity):continue
            paths=sorted(camera.glob('labels_*.npz'))
            poses=[];frames=[];contacts=[];known=[];wrists=[]
            for result in pool.map(_dexycb_frame,paths):
                if result is None:continue
                frame,pose,wrist=result;subject=int(sequence.parent.name.rsplit('-',1)[-1])
                sample=f'{subject}_{sequence.name}_{camera.name}_{frame}'
                poses.append(pose);frames.append(frame)
                wrists.append(wrist)
                known.append(side=='right' and sample in labels);contacts.append(labels.get(sample,np.zeros(778,np.uint8)))
            if not poses:continue
            # DexYCB pose_m joint values are PCA coefficients; official toolkit
            # ManoLayer defaults to 45 PCA components with hands_mean added.
            q=np.asarray(poses,np.float32);q[:,6:]=q[:,6:]@models[side].hands_components.numpy()+models[side].hands_mean.numpy()
            physical=q.copy();physical[:,:3]=np.asarray(wrists)
            if side=='left':physical=mirror_pose(physical)
            extra={} if beta is None else dict(beta=np.broadcast_to(beta,(len(q),len(beta))))
            writer.add(identity,physical,np.asarray(contacts),known,frames=frames,
                       metadata=dict(dataset='DexYCB',kind='temporal',group_id=group,source=str(sequence),side=side,
                                     coordinate_frame='fixed_camera',fps=30.,label_source='HACO:10mm',
                                     wrist_source='annotated_joint_3d_0',shape_available=beta is not None),**extra)
        count+=1
        if count%25==0:print('DexYCB',count,flush=True)


def prepare_epic(dataset, assets, writer, max_sequences=0):
    path=Path(dataset)/'EPIC_Contact/epic_contact_full.zip'
    if not path.exists():return
    models={s:ManoGeometry(Path(assets)/f'MANO_{s.upper()}.pkl') for s in ['right','left']}
    with zipfile.ZipFile(path) as z:
        names=z.namelist(); central=sorted(n for n in names if '/central_frames/' in n and n.endswith('/hand_pose.npz') and '__MACOSX' not in n)
        for number,name in enumerate(central):
            if max_sequences and number>=max_sequences:break
            folder=name.rsplit('/',1)[0];clip=folder.rsplit('/',1)[-1];identity='EPIC-central:'+clip
            if writer.contains(identity):continue
            side='left' if '_left_' in clip else 'right'
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as d:
                q=np.r_[d['transl'].reshape(3),d['global_orient'].reshape(3),d['hand_pose'].reshape(45)][None].astype(np.float32)
                beta=d['betas'].reshape(1,10)
            vertices,physical=_physical(models[side],q,beta)
            reference=np.asarray([[float(v) for v in line.split()[1:4]] for line in z.read(folder+'/hand_mesh.obj').decode().splitlines() if line.startswith('v ')])
            if reference.shape!=(778,3):raise ValueError('EPIC mesh has incompatible topology')
            correction=np.median(reference-vertices[0],axis=0)
            error=float(np.linalg.norm(vertices[0]+correction-reference,axis=-1).max())
            physical[:,:3]+=correction
            if side=='left':physical=mirror_pose(physical)
            manual=json.loads(z.read(folder+'/contact_vertices_sparse.json'))
            y=np.zeros((1,778),np.uint8)
            for values in manual.values():y[0,np.asarray(values,dtype=int)]=1
            video='_'.join(clip.split('_')[:2])
            writer.add(identity,physical,y,valid=[error<.002],metadata=dict(dataset='EPIC-Contact',kind='static',group_id='EPIC-video:'+video,
                       source=str(path)+':'+folder,side=side,label_source='manual_central_GT',coordinate_frame='camera',
                       fk_max_error_after_translation_m=error,source_translation_correction_m=correction),beta=beta)
        import csv
        quality={}
        for name in names:
            if '/quality_csvs/' in name and name.endswith('_frame_quality.csv') and '__MACOSX' not in name:
                for row in csv.DictReader(io.StringIO(z.read(name).decode())):
                    quality[(row['clip'],int(row['frame_num']))]=row
        propagated=sorted(n for n in names if '/per_frame_projected_npz/' in n and n.endswith('.npz') and '__MACOSX' not in n)
        for number,name in enumerate(propagated):
            if max_sequences and number>=max_sequences:break
            clip=Path(name).stem;identity='EPIC-propagated:'+clip
            if writer.contains(identity):continue
            side='left' if '_left_' in clip else 'right';suffix='l' if side=='left' else 'r'
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
                if not bool(a['clip_verified']):continue
                pose=a['mano.pose.'+suffix];beta=a['mano.beta.'+suffix]
                wrist=a['mano.j3d.cam.'+suffix][:,0];frames=a['_frame_num']
                usable=(a['left_valid' if side=='left' else 'right_valid']>.5)&(a['is_valid']>.5)
            score=np.asarray([float(quality.get((clip,int(f)),{}).get('confidence',1)) for f in frames],np.float32)
            duplicate=np.asarray([str(quality.get((clip,int(f)),{}).get('is_keyframe','0')).lower() in ('1','1.0','true') for f in frames])
            usable &= np.isfinite(pose).all(-1)&np.isfinite(wrist).all(-1)&np.isfinite(beta).all(-1)&(score>0)&~duplicate
            if not usable.any():continue
            physical=np.concatenate((wrist[usable],pose[usable]),-1).astype(np.float32)
            if side=='left':physical=mirror_pose(physical)
            video='_'.join(clip.split('_')[:2])
            # Sampled/propagated camera placements are independent reconstruction
            # observations, never claimed to be measured motion/contact GT.
            writer.add(identity,physical,frames=frames[usable],metadata=dict(dataset='EPIC-Contact',kind='static',
                       group_id='EPIC-video:'+video,source=str(path)+':'+name,side=side,label_source='propagated_pose_only',
                       coordinate_frame='camera',reconstruction_weight=.25),beta=beta[usable],
                       reconstruction_weight=.25*score[usable],source_frame_id=frames[usable])
            if number%200==0:print('EPIC propagated',number,len(propagated),flush=True)


def run_prepare(args):
    torch.set_num_threads(args.threads)
    root=Path(args.dataset); output=root/'MANO_contact'
    snapshot=snapshot_generated(args.source_root,output/'generated_snapshot.json') if 'generated' in args.sources.split(',') else None
    funcs={'hoi4d':prepare_hoi4d,'h2o':prepare_h2o,'dexycb':prepare_dexycb,'epic':prepare_epic}
    with CorpusWriter(args.output) as writer:
        writer.file.attrs.update(requested_sources=args.sources,source_limit=args.limit)
        for name in args.sources.split(','):
            print('START',name,flush=True)
            if name=='generated':prepare_generated(snapshot,writer,args.limit,args.workers)
            elif name=='primitives':prepare_primitives(args.source_root,writer,args.limit)
            else:funcs[name](root,args.assets,writer,args.limit)
            print('DONE',name,flush=True)
    report=finalize_corpus(args.output,Path(args.output).with_suffix('.inventory.json'))
    print(json.dumps(report),flush=True)
