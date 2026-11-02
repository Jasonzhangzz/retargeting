"""Export dense size and pose arrays from every final solved grasp grid."""
import argparse,hashlib,json,os,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from retarget.common.results import write_json
from retarget.viewer.dense_pose import interpolate_pose
from retarget.viewer.snapshot import get_snapshot


def export(batch,output,subdivisions=64,activate=False):
    batch=Path(batch).resolve();output=Path(output).resolve()
    if subdivisions<2:raise ValueError('Need at least two subdivisions')
    manifest=batch/'manifest.json';entries=json.loads(manifest.read_text())['cases'];families=defaultdict(list)
    for e in entries:
        if e.get('task')=='grasp' and e.get('primitive'):families[(e['template'],e['primitive']['kind'])].append(e)
    output.mkdir(parents=True,exist_ok=False);groups=[]
    for (name,kind),rows in families.items():
        rows=sorted(rows,key=lambda e:e['primitive']['size_scale']);scales=np.array([e['primitive']['size_scale'] for e in rows])
        if len(scales)<2:continue
        if len(np.unique(scales))!=len(scales):raise ValueError(f'Duplicate source sizes: {name}')
        poses=[];projections=[]
        for e in rows:
            if 'cached_case' in e:
                snapshot=get_snapshot(batch);index=int(e['cached_case'])
                poses.append(snapshot.poses['qpos'][index].copy())
                projections.append(np.asarray(snapshot.diagnostics(index)['projected_points']))
            else:
                case=(batch/Path(e['trajectory'])).parent
                poses.append(np.load(case/'grasp.npz',allow_pickle=False)['qpos'])
                projections.append(np.asarray(json.loads((case/'projection_force_closure.json').read_text())['projected_points']))
        poses=np.asarray(poses)
        query=np.r_[np.concatenate([np.linspace(a,b,subdivisions,endpoint=False) for a,b in zip(scales,scales[1:])]),scales[-1]]
        q=interpolate_pose(scales,poses,query)
        values=dict(scales=query,qpos=q,source_scales=scales,source_qpos=poses,
            source_indices=np.arange(len(scales))*subdivisions)
        for key in ('radius_m','height_m'):
            values[key]=np.interp(query,scales,[e['primitive'][key] for e in rows])
        dimensions=np.array([e['primitive']['dimensions_m'] for e in rows])
        values['dimensions_m']=np.stack([np.interp(query,scales,dimensions[:,i]) for i in range(3)],axis=1)
        if len({p.shape for p in projections})==1:
            points=np.asarray(projections);flat=points.reshape(len(scales),-1)
            values['projected_points']=np.stack([np.interp(query,scales,flat[:,i]) for i in range(flat.shape[1])],axis=1).reshape(len(query),*points.shape[1:])
        # Original solved knots and dimensions must survive byte-for-byte.
        assert np.array_equal(q[values['source_indices']],poses)
        for key in ('radius_m','height_m'):
            assert np.array_equal(values[key][values['source_indices']],[e['primitive'][key] for e in rows])
        assert np.array_equal(values['dimensions_m'][values['source_indices']],dimensions)
        file=output/f'{name}_{kind}.npz';np.savez_compressed(file,**values)
        groups.append(dict(template=name,kind=kind,file=str(file),source_cases=len(rows),dense_poses=len(query),
            min_scale=float(scales[0]),max_scale=float(scales[-1]),source_rows=rows))
    index=dict(version=1,policy='linear_dimensions_and_joints_shortest_arc_root_rotation',
        source_manifest=str(manifest),source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        subdivisions=subdivisions,groups=groups,source_cases=sum(g['source_cases'] for g in groups),
        dense_poses=sum(g['dense_poses'] for g in groups),interpolated_poses_are_solver_verified=False)
    write_json(output/'index.json',index)
    if activate:
        target=batch/'dense_pose_index.json';tmp=target.with_suffix('.tmp');write_json(tmp,index);os.replace(tmp,target)
    print(json.dumps({k:v for k,v in index.items() if k!='groups'},indent=2))
    return index


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--subdivisions',type=int,default=64);parser.add_argument('--activate',action='store_true');args=parser.parse_args()
    export(args.batch,args.output,args.subdivisions,args.activate)
