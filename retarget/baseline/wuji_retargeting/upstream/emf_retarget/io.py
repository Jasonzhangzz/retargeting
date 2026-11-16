"""Dataset adapters; all numerical state lives in EmfRetargetPipeline."""
from pathlib import Path
import numpy as np
from .frames import EmfFrame


def _text(value):
    return value.decode('utf8') if isinstance(value,bytes) else str(value)


def iter_emf_frames(path, side, start=0, limit=None):
    """Read native glove H5, or a simple single-hand NPZ; no time resampling.

    NPZ requires positions[N,5,3], orientations_xyzw[N,5,4], confidence[N,5].
    Optional seq[N], timestamp_us[N], frame_id[N]; absent headers use row
    indices and an empty frame_id. H5 uses the project's left/right schema.
    """
    path=Path(path)
    if side not in ('left','right'):raise ValueError('side must be left or right')
    if start<0 or (limit is not None and limit<=0):raise ValueError('start >= 0 and limit > 0 required')
    if path.suffix.lower() in ('.h5','.hdf5'):
        try:import h5py
        except ImportError as exc:raise RuntimeError('HDF5 input requires h5py; install the hdf5 extra') from exc
        with h5py.File(path,'r',locking=False) as h:
            if side not in h:raise ValueError(f'input has no {side} stream')
            g=h[side]
            arrays=[g['emf_poses/position'],g['emf_poses/orientation_xyzw'],g['emf_poses/confidence']]
            n=len(g['source_seq'])
            if [a.shape for a in arrays]!=[(n,5,3),(n,5,4),(n,5)]:raise ValueError('invalid HDF5 EMF array shapes')
            if any(len(g[k])!=n for k in ('source_timestamp_ns','source_frame_id')):raise ValueError('inconsistent HDF5 headers')
            stop=n if limit is None else min(n,start+limit)
            for i in range(start,stop):
                yield EmfFrame(*(a[i] for a in arrays),seq=int(g['source_seq'][i]),
                    timestamp_us=int(g['source_timestamp_ns'][i])//1000,frame_id=_text(g['source_frame_id'][i]))
    elif path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as z:
            arrays=[z[name] for name in ('positions','orientations_xyzw','confidence')]
            n=len(arrays[0])
            if [a.shape for a in arrays]!=[(n,5,3),(n,5,4),(n,5)]:raise ValueError('invalid NPZ EMF array shapes')
            for key in ('seq','timestamp_us','frame_id'):
                if key in z and z[key].shape!=(n,):raise ValueError(f'{key} must have shape (N,)')
            stop=n if limit is None else min(n,start+limit)
            for i in range(start,stop):
                yield EmfFrame(*(a[i] for a in arrays),seq=int(z['seq'][i]) if 'seq' in z else i,
                    timestamp_us=int(z['timestamp_us'][i]) if 'timestamp_us' in z else i,
                    frame_id=_text(z['frame_id'][i]) if 'frame_id' in z else '')
    else:raise ValueError('input must be .h5, .hdf5 or .npz')


def output_arrays(results, pipeline):
    if not results:raise ValueError('selected input contains no frames')
    arrays={name:np.asarray([getattr(r,name) for r in results]) for name in
        ('seq','timestamp_us','frame_id','human_q','human_confidence','skeleton_position',
         'skeleton_orientation_xyzw','skeleton_confidence','tip_position','tip_orientation_xyzw','human_position_error_m')}
    arrays['human_joint_names']=np.array(pipeline.human.kinematics.joint_names)
    arrays['skeleton_names']=np.array(pipeline.human.kinematics.skeleton_names)
    for model in pipeline.models:
        arrays[model+'_q']=np.stack([r.robot_q[model] for r in results])
        arrays[model+'_joint_names']=np.array(pipeline.robots[model].kinematics.joint_names)
    return {pipeline.side+'_'+key:value for key,value in arrays.items()}

