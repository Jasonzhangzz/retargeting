"""NumPy FK/Jacobian for the four URDFs extracted from SDK 2026.8.31.

No SDK import or private ABI is used here. Positions are metres; q is radians.
The SDK oracle is confined to the separate probe-kinematics.py verification.
"""
from pathlib import Path
import ctypes
import math
import xml.etree.ElementTree as ET
import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets'

# The Linux SDK uses the system libm sincos symbol. Separate sin/cos calls
# differ by one ULP for some inputs, enough to change an SLSQP branch later.
# This is a platform math function, not a call into wuji_sdk or its private ABI.
try:
    _libm = ctypes.CDLL(None)
    _native_sincos = _libm.sincos
    _native_sincos.argtypes = (ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(ctypes.c_double))
    _native_sincos.restype = None
except (AttributeError, OSError):
    _native_sincos = None


def _sincos(angle):
    if _native_sincos is None:
        return math.sin(angle), math.cos(angle)
    sine, cosine = ctypes.c_double(), ctypes.c_double()
    _native_sincos(float(angle), ctypes.byref(sine), ctypes.byref(cosine))
    return sine.value, cosine.value


def _rpy_quaternion(values):
    r, p, y = np.asarray(values, dtype=float) * .5
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    return np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                     cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])


def _quaternion_product(a, b):
    # Preserve the scalar operation order in native helper 0x10aa1bd.
    x, y, z, w = a
    X, Y, Z, W = b
    return np.array([(y*Z + (W*x + w*X)) - Y*z,
                     X*z + (W*y + (w*Y - x*Z)),
                     W*z + ((x*Y + w*Z) - X*y),
                     ((w*W - x*X) - y*Y) - z*Z])


def _cross(a, b):
    x, y, z = a
    X, Y, Z = b
    return np.array([y*Z-z*Y, z*X-x*Z, x*Y-y*X])


def _quaternion_rotate(q, vector):
    # Native helper 0x10aa12e; do not expand to a rotation matrix.
    twice_cross = _cross(q[:3], vector) * 2.
    return (twice_cross*q[3] + _cross(q[:3], twice_cross)) + vector


def _rpy(values):
    r,p,y = values
    cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def _axis_angle(axis, angle):
    x,y,z = axis
    k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+math.sin(angle)*k+(1.-math.cos(angle))*(k@k)


def _numbers(element, attribute, default):
    return np.fromstring(element.get(attribute,default) if element is not None else default,sep=' ')


class HandKinematics:
    """20-DOF, 21-landmark rigid tree with analytic position Jacobians.

    forward(q) -> (P[21,3], J[21,3,20]); no clipping is applied to q.
    joint_names/bounds use proximal-to-distal chain order, thumb to pinky.
    """
    def __init__(self, model='WujiHand', side='right'):
        model = {'hand1':'WujiHand','hand2':'WujiHand2','wuji_hand':'WujiHand',
                 'wuji_hand_2':'WujiHand2'}.get(str(model).lower(),str(model))
        side = str(side).lower()
        if model not in ('WujiHand','WujiHand2') or side not in ('left','right'):
            raise ValueError('Expected WujiHand/WujiHand2 and left/right')
        self.model,self.side=model,side
        self.urdf_path=ASSETS/(model+'_'+side+'.urdf')
        tree=ET.parse(self.urdf_path).getroot()
        self.link_names=[x.get('name') for x in tree.findall('link')]
        if model=='WujiHand':
            prefix=side+'_'
            fingers=['finger'+str(i) for i in range(1,6)]
            self.landmark_names=[prefix+'palm_link']+[prefix+f+s for f in fingers for s in ('_link2','_link3','_link4','_tip_link')]
            tips=[prefix+f+'_tip_link' for f in fingers]
        else:
            prefix=side[0]+'_'
            fingers=['thumb','index_finger','middle_finger','ring_finger','pinky']
            self.landmark_names=[prefix+'wrist']+[prefix+f+s for f in fingers for s in ('_proximal','_middle','_distal','_tip')]
            tips=[prefix+f+'_tip' for f in fingers]
        self.joints=[]
        by_child={}
        for e in tree.findall('joint'):
            kind=e.get('type')
            if kind not in ('fixed','revolute') or e.find('mimic') is not None:
                raise ValueError('Unsupported joint kind/mimic')
            axis=_numbers(e.find('axis'),'xyz','1 0 0')
            axis=axis/np.linalg.norm(axis)
            lim=e.find('limit')
            j={'name':e.get('name'),'kind':kind,'parent':e.find('parent').get('link'),
               'child':e.find('child').get('link'),'xyz':_numbers(e.find('origin'),'xyz','0 0 0'),
               'rotation':_rpy(_numbers(e.find('origin'),'rpy','0 0 0')),
               'quaternion':_rpy_quaternion(_numbers(e.find('origin'),'rpy','0 0 0')),'axis':axis,
               'lower':float(lim.get('lower','-inf')) if lim is not None else -np.inf,
               'upper':float(lim.get('upper','inf')) if lim is not None else np.inf}
            self.joints.append(j)
            by_child[j['child']]=j
        roots=set(self.link_names)-set(by_child)
        if len(roots)!=1:raise ValueError('URDF is not one rooted tree')
        self.root=roots.pop()
        self.joint_names=[]
        for tip in tips:
            chain=[]; node=tip
            while node in by_child:
                j=by_child[node]
                if j['kind']!='fixed':chain.append(j['name'])
                node=j['parent']
            self.joint_names.extend(reversed(chain))
        if len(self.joint_names)!=20 or len(set(self.joint_names))!=20:
            raise ValueError('Expected five disjoint chains covering 20 actuated joints')
        index={n:i for i,n in enumerate(self.joint_names)}
        self.bounds=np.empty((20,2))
        for j in self.joints:
            j['q_index']=index.get(j['name'])
            if j['q_index'] is not None:self.bounds[j['q_index']]=j['lower'],j['upper']
        # Observed session+0x170 vector: only the two distal joints of each chain.
        self.antirev_indices=np.array([i for i in range(20) if i%4 in (2,3)],dtype=np.int64)
        self.landmark_ids=np.array([self.link_names.index(n) for n in self.landmark_names],dtype=np.int64)
        self.lower=self.bounds[:,0].copy(); self.upper=self.bounds[:,1].copy()
        self.nq=20
        remaining=list(self.joints); ordered=[]; found={self.root}
        while remaining:
            available=[j for j in remaining if j['parent'] in found]
            if not available:raise ValueError('Disconnected/cyclic URDF')
            for j in available:ordered.append(j);found.add(j['child']);remaining.remove(j)
        self._ordered=ordered

    def forward(self,q,backoff=0.0):
        q=np.asarray(q,dtype=np.float64)
        if q.shape!=(20,) or not np.isfinite(q).all():raise ValueError('q must be 20 finite radians')
        if backoff!=0.0:raise NotImplementedError('Only SDK builtin backoff=0 is currently verified')
        rotations={self.root:np.array([0.,0.,0.,1.])}; positions={self.root:np.zeros(3)}
        ancestors={self.root:[]}; axes=np.zeros((20,3)); origins=np.zeros((20,3))
        for j in self._ordered:
            rp=rotations[j['parent']]
            pos=_quaternion_rotate(rp,j['xyz'])+positions[j['parent']]
            rot=_quaternion_product(rp,j['quaternion'])
            path=ancestors[j['parent']].copy()
            k=j['q_index']
            if k is not None:
                axes[k]=_quaternion_rotate(rot,j['axis']);origins[k]=pos
                sine,cosine=_sincos(q[k]*.5)
                rot=_quaternion_product(rot,np.r_[j['axis']*sine,cosine])
                path.append(k)
            rotations[j['child']]=rot;positions[j['child']]=pos;ancestors[j['child']]=path
        p=np.stack([positions[n] for n in self.landmark_names])
        jac=np.zeros((21,3,20))
        for i,name in enumerate(self.landmark_names):
            for k in ancestors[name]:
                jac[i,:,k]=_cross(axes[k],p[i]-origins[k])
        return p,jac

    fk = forward
    fk_jacobian = forward


def load_model(model='WujiHand',side='right'):
    return HandKinematics(model,side)
