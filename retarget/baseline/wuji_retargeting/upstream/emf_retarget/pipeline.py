"""One stateful frame path: EMF -> human IK/FK -> robot retarget sessions."""
import copy
import math
import threading
import numpy as np
from human_ik_reference import HumanIKSession
from retarget_reference import RetargetSession
from .frames import EmfFrame, RetargetResult

_MODELS={'wuji1':'WujiHand','wuji2':'WujiHand2','WujiHand':'WujiHand','WujiHand2':'WujiHand2'}


def matrix_to_xyzw(rotation):
    """Stable unit quaternion; q and -q denote the same pose."""
    r=np.asarray(rotation,dtype=float);trace=float(np.trace(r))
    if trace>0:
        s=2*math.sqrt(1+trace)
        q=np.array([(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s,s/4])
    else:
        i=int(np.argmax(np.diag(r)));j=(i+1)%3;k=(i+2)%3
        s=2*math.sqrt(max(0.,1+r[i,i]-r[j,j]-r[k,k]));q=np.empty(4)
        q[i]=s/4;q[j]=(r[j,i]+r[i,j])/s;q[k]=(r[k,i]+r[i,k])/s;q[3]=(r[k,j]-r[j,k])/s
    return q/np.linalg.norm(q)


class EmfRetargetPipeline:
    """Use one instance per hand stream; models can contain wuji1, wuji2 or both.

    Human IK state is shared by both robots. Each robot retains its own warm
    start and low-pass filter. Identical consecutive frames are computed once;
    a changed header is a new frame. Outputs are copies of the cached value.
    Runtime code never imports wuji_sdk. URDF defaults are the bundled CQQ
    models, selected explicitly by side; serial-number lookup is not emulated.
    """
    def __init__(self, side='right', models=('wuji1','wuji2'), human_urdf=None):
        self.side=str(side).lower()
        if self.side not in ('left','right'):raise ValueError('side must be left or right')
        if isinstance(models,str):models=(models,)
        names=[]
        for model in models:
            if model not in _MODELS:raise ValueError('models must contain wuji1 or wuji2')
            key='wuji1' if _MODELS[model]=='WujiHand' else 'wuji2'
            if key not in names:names.append(key)
        if not names:raise ValueError('at least one robot model is required')
        self.models=tuple(names)
        self.human=HumanIKSession(self.side,human_urdf)
        self.robots={name:RetargetSession.for_hand(_MODELS[name],self.side) for name in self.models}
        self._lock=threading.Lock();self._key=None;self._result=None
        self.frames_computed=0

    def reset(self):
        """Restore human seeds, robot warm/filter states and the frame cache."""
        with self._lock:
            self.human.reset()
            for robot in self.robots.values():robot.reset()
            self._key=self._result=None;self.frames_computed=0

    def process(self, frame: EmfFrame) -> RetargetResult:
        if not isinstance(frame,EmfFrame):raise TypeError('process expects an EmfFrame')
        with self._lock:
            key=frame.cache_key()
            if key==self._key:return copy.deepcopy(self._result)
            # Failed frames must not leave just one component advanced. This
            # is our unified API contract, not a claim about all SDK failures.
            human_state=copy.deepcopy((self.human._history,self.human._history2,self.human.last_trace))
            robot_states={name:copy.deepcopy((robot._warm,robot._filtered,robot.last_trace)) for name,robot in self.robots.items()}
            try:
                h=self.human.step(frame.positions,frame.orientations_xyzw,frame.confidence)
                # SDK poses cross an f32 public boundary before RetargetSession.
                skeleton_position=h['skeleton'][:,:3,3].astype(np.float32)
                skeleton_orientation=np.array([matrix_to_xyzw(t[:3,:3]) for t in h['skeleton']],dtype=np.float32)
                robot_q={name:robot.step(skeleton_position) for name,robot in self.robots.items()}
                result=RetargetResult(frame.seq,frame.timestamp_us,frame.frame_id,self.side,
                    h['q'],h['confidence'],skeleton_position,skeleton_orientation,h['skeleton_confidence'].astype(np.float32),
                    h['tips'][:,:3,3].astype(np.float32),
                    np.array([matrix_to_xyzw(t[:3,:3]) for t in h['tips']],dtype=np.float32),
                    robot_q,h['position_error_m'])
            except Exception:
                self.human._history,self.human._history2,self.human.last_trace=human_state
                for name,robot in self.robots.items():
                    robot._warm,robot._filtered,robot.last_trace=robot_states[name]
                raise
            self._key=key;self._result=result;self.frames_computed+=1
            return copy.deepcopy(result)
