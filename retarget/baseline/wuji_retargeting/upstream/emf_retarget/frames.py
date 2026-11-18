"""Typed data at the single public EMF-to-robot boundary."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EmfFrame:
    """Five TX-frame RX poses: thumb, index, middle, ring, pinky.

    Position units are metres; quaternions are XYZW. This is decoded pose
    data, not raw electromagnetic field measurements. Header values take
    part in the last-frame cache identity.
    """
    positions: np.ndarray
    orientations_xyzw: np.ndarray
    confidence: np.ndarray
    seq: int = 0
    timestamp_us: int = 0
    frame_id: str = ''

    def __post_init__(self):
        for name, shape in [('positions',(5,3)),('orientations_xyzw',(5,4)),('confidence',(5,))]:
            value=np.array(getattr(self,name),dtype=np.float32,order='C',copy=True)
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError(f'{name} must be finite with shape {shape}')
            value.flags.writeable=False
            object.__setattr__(self,name,value)
        if np.any(np.linalg.norm(self.orientations_xyzw.astype(float),axis=1)==0):
            raise ValueError('EMF quaternion cannot be zero')
        for name in ('seq','timestamp_us'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or not 0<=int(value)<2**64:
                raise ValueError(f'{name} must be an unsigned 64-bit integer')
            object.__setattr__(self,name,int(value))
        if not isinstance(self.frame_id,str):raise TypeError('frame_id must be a string')

    def cache_key(self):
        return (self.seq,self.timestamp_us,self.frame_id,self.positions.tobytes(),
                self.orientations_xyzw.tobytes(),self.confidence.tobytes())


@dataclass
class RetargetResult:
    seq: int
    timestamp_us: int
    frame_id: str
    side: str
    human_q: np.ndarray
    human_confidence: np.ndarray
    skeleton_position: np.ndarray
    skeleton_orientation_xyzw: np.ndarray
    skeleton_confidence: np.ndarray
    tip_position: np.ndarray
    tip_orientation_xyzw: np.ndarray
    robot_q: dict
    human_position_error_m: np.ndarray

