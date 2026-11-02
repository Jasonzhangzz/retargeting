"""Dense size/pose interpolation, preserving solved knots and SO(3) rotations."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def interpolate_pose(scales,qpos,query):
    scales=np.asarray(scales,float);qpos=np.asarray(qpos,float);query=np.atleast_1d(query).astype(float)
    if len(scales)<2 or np.any(np.diff(scales)<=0) or qpos.shape!=(len(scales),26):
        raise ValueError('Need increasing scales and finite [N,26] poses')
    if not np.isfinite(qpos).all() or query.min()<scales[0]-1e-12 or query.max()>scales[-1]+1e-12:
        raise ValueError('Invalid interpolation input')
    result=np.stack([np.interp(query,scales,qpos[:,i]) for i in range(26)],axis=1)
    rotation=Slerp(scales,Rotation.from_euler('XYZ',qpos[:,3:6]))(query)
    result[:,3:6]=rotation.as_euler('XYZ')
    # Exact knots preserve the saved generalized coordinates, including their
    # Euler branch. Rotation interpolation itself follows the shortest SO(3) arc.
    ids=np.searchsorted(scales,query).clip(0,len(scales)-1)
    exact=np.abs(scales[ids]-query)<1e-12
    result[exact]=qpos[ids[exact]]
    return result


class DensePoseGrid:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as values:
            self.values={k:values[k].copy() for k in values.files}
        self.scales=self.values['scales'];self.qpos=self.values['qpos']
        self.rotation=Slerp(self.scales,Rotation.from_euler('XYZ',self.qpos[:,3:6]))
        self.source_scales=self.values['source_scales']
        self.last=None

    def sample(self,scale):
        scale=float(np.clip(scale,self.scales[0],self.scales[-1]))
        right=int(np.searchsorted(self.scales,scale,side='right').clip(1,len(self.scales)-1));left=right-1
        u=(scale-self.scales[left])/(self.scales[right]-self.scales[left])
        def blend(name):return self.values[name][left]*(1-u)+self.values[name][right]*u
        q=blend('qpos');q[3:6]=self.rotation([scale]).as_euler('XYZ')[0]
        source=int(np.argmin(np.abs(self.source_scales-scale)))
        solved=bool(abs(self.source_scales[source]-scale)<1e-10)
        if solved:q=self.values['source_qpos'][source].copy()
        projections = blend('projected_points') if 'projected_points' in self.values else None
        if 'source_projected_points' in self.values:
            original=self.values['source_projected_points'];flat=original.reshape(len(self.source_scales),-1)
            projections=np.array([np.interp(scale,self.source_scales,flat[:,i]) for i in range(flat.shape[1])]).reshape(original.shape[1:])
        return dict(scale=scale,qpos=q,radius_m=float(blend('radius_m')),height_m=float(blend('height_m')),
            dimensions_m=blend('dimensions_m'),projected_points=projections,
            solved=solved,source_index=source)


@lru_cache(maxsize=128)
def _load_grid(path,modified):
    return DensePoseGrid(path)


def load_dense_grid(batch,template,kind):
    index=Path(batch)/'dense_pose_index.json'
    if not index.is_file():return None
    description=json.loads(index.read_text())
    manifest=Path(batch)/'manifest.json'
    if manifest.is_file() and hashlib.sha256(manifest.read_bytes()).hexdigest()!=description['source_manifest_sha256']:
        return None
    group=next((g for g in description['groups'] if g['template']==template and g['kind']==kind),None)
    if group is None:return None
    path=Path(group['file']);path=path if path.is_absolute() else index.parent/path
    grid=_load_grid(str(path),path.stat().st_mtime_ns)
    grid.source_rows=group.get('source_rows',[])
    if 'source_case_indices' in group:
        rows=json.loads(manifest.read_text())['cases']
        grid.source_rows=[rows[i] for i in group['source_case_indices']]
    return grid


def dense_revision(batch):
    try:
        return ((Path(batch)/'manifest.json').stat().st_mtime_ns,
                (Path(batch)/'dense_pose_index.json').stat().st_mtime_ns)
    except FileNotFoundError:
        return None
