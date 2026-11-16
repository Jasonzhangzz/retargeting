"""Independent human-hand URDF FK in metres, with analytic spatial Jacobians.

No SDK import. Assets are the project's explicit cqq URDF override, not an
assertion about the SDK's serial-number-dependent default model selection.
"""
from pathlib import Path
import ctypes
import math
import xml.etree.ElementTree as ET
import numpy as np

FINGERS = ('thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky')
RX_NAMES = ('thumb_rx_coil', 'index_rx_coil', 'middle_rx_coil', 'ring_rx_coil', 'pinky_rx_coil')

# Native joint-axis quaternions call the platform sincos function. This is
# system libm only; the independent implementation never calls into wuji_sdk.
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
    sr, cr = _sincos(float(values[0]) * .5)
    sp, cp = _sincos(float(values[1]) * .5)
    sy, cy = _sincos(float(values[2]) * .5)
    q = np.array([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                  cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy])
    x, y, z, w = q
    # Human model construction normalizes fixed-origin quaternions. The
    # grouping is the native norm helper at ELF 0xdc840b.
    return q / math.sqrt(0. + ((w*w+y*y)+(z*z+x*x)))


def _quaternion_product(a, b):
    """ELF 0xdac738 scalar multiplication/addition order."""
    x, y, z, w = a
    X, Y, Z, W = b
    return np.array([(y*Z + (W*x + w*X)) - Y*z,
                     X*z + (W*y + (w*Y - x*Z)),
                     W*z + ((x*Y + w*Z) - X*y),
                     ((w*W - x*X) - y*Y) - z*Z])


def _cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]])


def _quaternion_rotate(q, vector):
    twice_cross = _cross(q[:3], vector) * 2.
    return (twice_cross*q[3] + _cross(q[:3], twice_cross)) + vector


def _quaternion_matrix(q):
    """ELF 0xdac03f: do not renormalize an accumulated quaternion."""
    x,y,z,w = q
    xx,yy,zz,ww = x*x,y*y,z*z,w*w
    xy,xz,yz,xw,yw,zw = x*y,x*z,y*z,x*w,y*w,z*w
    xy,xz,yz,xw,yw,zw = xy+xy,xz+xz,yz+yz,xw+xw,yw+yw,zw+zw
    return np.array([[((xx+ww)-yy)-zz,xy-zw,xz+yw],
                     [zw+xy,(yy+(ww-xx))-zz,yz-xw],
                     [xz-yw,xw+yz,((ww-xx)-yy)+zz]])


def skew(v):
    x,y,z = v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def so3_exp(v):
    v=np.asarray(v,dtype=float); theta=float(np.linalg.norm(v)); k=skew(v)
    if theta < 1e-7:
        t2=theta*theta; a=1-t2/6+t2*t2/120; b=.5-t2/24+t2*t2/720
    else:
        a=math.sin(theta)/theta; b=(1-math.cos(theta))/(theta*theta)
    return np.eye(3)+a*k+b*(k@k)


def so3_log(r):
    """Principal rotation vector; the derivative is undefined at the pi cut."""
    r=np.asarray(r,dtype=float)
    v=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])*.5
    sine=float(np.linalg.norm(v)); cosine=float(np.clip((np.trace(r)-1)*.5,-1.,1.))
    theta=math.atan2(sine,cosine)
    if theta < 1e-7:return v*(1+theta*theta/6+7*theta**4/360)
    if math.pi-theta < 1e-6:
        # Symmetric axis recovery stays stable when sin(theta) vanishes.
        vals,vecs=np.linalg.eigh((r+r.T)*.5)
        axis=vecs[:,np.argmax(vals)]
        if np.dot(axis,v)<0:axis=-axis
        return theta*axis
    return (theta/sine)*v


def so3_left_jacobian_inverse(v):
    v=np.asarray(v,dtype=float); theta=float(np.linalg.norm(v)); k=skew(v)
    if theta < 1e-5:
        coefficient=1/12+theta*theta/720+theta**4/30240
    else:
        coefficient=(1-.5*theta/math.tan(.5*theta))/(theta*theta)
    return np.eye(3)-.5*k+coefficient*(k@k)


def pose_residual_jacobian(transform, target, j_position, j_angular):
    """Return r=[p_current-p_target, log(R_current.T@R_target)] and dr/dq.

    Inputs use one common frame (normally TX), metres and radians. j_angular
    is a spatial angular-velocity Jacobian in that frame. This is the exact
    derivative of the stated residual, before weights or SDK length scaling.
    """
    rotation=transform[:3,:3]
    rot_error=so3_log(rotation.T@target[:3,:3])
    residual=np.r_[transform[:3,3]-target[:3,3],rot_error]
    jac=np.vstack((j_position,-so3_left_jacobian_inverse(rot_error)@rotation.T@j_angular))
    return residual,jac


def _rpy(v):
    r,p,y=v
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def _numbers(element,key,default):
    return np.fromstring(element.get(key,default) if element is not None else default,sep=' ')


class HumanHandKinematics:
    """21 DOF: thumb (rot, flex, abd, mcp, ip), then four (flex, abd, pip, dip).

    forward returns wrist-frame link/skeleton/tip transforms and TX-frame RX
    poses/Jacobians. finger_pose_jacobian returns just one RX and local columns.
    q is never clipped; bounds are the URDF limits, not necessarily IK overrides.
    """
    def __init__(self, side='right', urdf_path=None, sdk_tx_rounding=False):
        side=str(side).lower()
        if side not in ('left','right'):raise ValueError('side must be left or right')
        self.side=side
        self.urdf_path=Path(urdf_path) if urdf_path else Path(__file__).parent/'assets'/(side+'_hand.urdf')
        self.finger_names=FINGERS
        self.joint_names=tuple('thumb_'+s for s in ('cmc_rot','cmc_flex','cmc_abd','mcp','ip'))+tuple(f+'_'+s for f in FINGERS[1:] for s in ('mcp_flex','mcp_abd','pip','dip'))
        self.finger_slices=(slice(0,5),slice(5,9),slice(9,13),slice(13,17),slice(17,21))
        self.nq=21
        self.skeleton_names=('wrist',)+tuple(f+'_'+s for f in FINGERS for s in (('cmc','mcp','ip','tip') if f=='thumb' else ('mcp','pip','dip','tip')))
        self.skeleton_links=('wrist',)+tuple(f+'_'+s for f in FINGERS for s in (('metacarpal','middle','distal','tip') if f=='thumb' else ('proximal','middle','distal','tip')))
        tree=ET.parse(self.urdf_path).getroot(); index={n:i for i,n in enumerate(self.joint_names)}
        self.bounds=np.empty((21,2)); joints=[]; by_child={}; found_indices=set()
        for e in tree.findall('joint'):
            name=e.get('name'); kind=e.get('type')
            if kind not in ('fixed','revolute') or e.find('mimic') is not None:raise ValueError('Unsupported joint type or mimic')
            k=index.get(name)
            if kind!='fixed' and k is None:raise ValueError('Unmapped actuated joint '+name)
            axis=_numbers(e.find('axis'),'xyz','1 0 0'); axis=axis/np.linalg.norm(axis)
            rpy=_numbers(e.find('origin'),'rpy','0 0 0')
            origin=np.eye(4); origin[:3,3]=_numbers(e.find('origin'),'xyz','0 0 0'); origin[:3,:3]=_rpy(rpy)
            joint={'name':name,'parent':e.find('parent').get('link'),'child':e.find('child').get('link'),'axis':axis,'origin':origin,'quaternion':_rpy_quaternion(rpy),'index':k}
            joints.append(joint);by_child[joint['child']]=joint
            if k is not None:
                lim=e.find('limit'); self.bounds[k]=float(lim.get('lower')),float(lim.get('upper'));found_indices.add(k)
        if found_indices!=set(range(21)):raise ValueError('Missing expected 21 joints')
        remaining=joints.copy();ordered=[];found={'wrist'}
        while remaining:
            ready=[j for j in remaining if j['parent'] in found]
            if not ready:raise ValueError('URDF is disconnected from wrist')
            for j in ready:ordered.append(j);found.add(j['child']);remaining.remove(j)
        self._ordered=ordered; self._chains=[]
        for rx in RX_NAMES:
            chain=[];node=rx
            while node!='wrist':
                j=by_child[node];chain.append(j);node=j['parent']
            self._chains.append(tuple(reversed(chain)))
        tx=np.eye(4);chain=[];node='emf_tx_base'
        while node!='wrist':
            j=by_child[node]
            if j['index'] is not None:raise ValueError('TX must be fixed to wrist')
            chain.append(j);node=j['parent']
        tx_quaternion=np.array([0.,0.,0.,1.])
        for j in reversed(chain):
            tx=tx@j['origin']
            tx_quaternion=_quaternion_product(tx_quaternion,j['quaternion'])
        self.tx_transform=tx;self._tx_inverse=np.eye(4)
        self._tx_inverse[:3,:3]=tx[:3,:3].T;self._tx_inverse[:3,3]=-tx[:3,:3].T@tx[:3,3]
        self._tx_inverse_quaternion=tx_quaternion*np.array([-1.,-1.,-1.,1.])
        self.sdk_tx_rounding=bool(sdk_tx_rounding)
        if sdk_tx_rounding:
            # Native rerooting rebuilds the inverse TX joint as decimal URDF.
            # dc847a -> dc8340 (inverse pose), dac03f (matrix), da71a5 (RPY).
            # The generated inverse joint's xyz/rpy are serialized to 10 places.
            ri=self._tx_inverse[:3,:3]
            pitch=math.asin(float(np.clip(-ri[2,0],-1.,1.)))
            if abs(math.cos(pitch))<1e-8:
                raise ValueError('SDK TX rounding at Euler gimbal lock is not yet verified')
            rpy=np.array([math.atan2(ri[2,1],ri[2,2]),pitch,math.atan2(ri[1,0],ri[0,0])])
            rounded_rpy=np.array([float(format(x,'.10f')) for x in rpy])
            self._tx_inverse[:3,:3]=_rpy(rounded_rpy)
            self._tx_inverse_quaternion=_rpy_quaternion(rounded_rpy)
            self._tx_inverse[:3,3]=[float(format(x,'.10f')) for x in self._tx_inverse[:3,3]]
        self.lower=self.bounds[:,0].copy();self.upper=self.bounds[:,1].copy()

    def _finger_index(self,finger):
        if isinstance(finger,str):return FINGERS.index({'index':'index_finger','middle':'middle_finger','ring':'ring_finger','little':'pinky'}.get(finger,finger))
        finger=int(finger)
        if not 0<=finger<5:raise ValueError('finger index must be 0..4')
        return finger

    def finger_pose_jacobian(self,finger,q_finger):
        f=self._finger_index(finger);sl=self.finger_slices[f];n=sl.stop-sl.start
        q=np.asarray(q_finger,dtype=float)
        if q.shape!=(n,) or not np.isfinite(q).all():raise ValueError('Incorrect or nonfinite finger q')
        rotation=self._tx_inverse_quaternion.copy()
        position=self._tx_inverse[:3,3].copy()
        axes=np.zeros((n,3));origins=np.zeros((n,3))
        for j in self._chains[f]:
            local=j['quaternion'];k=j['index']
            if k is not None:
                k-=sl.start;sine,cosine=_sincos(q[k]*.5)
                local=_quaternion_product(local,np.r_[j['axis']*sine,cosine])
            position=position+_quaternion_rotate(rotation,j['origin'][:3,3])
            rotation=_quaternion_product(rotation,local)
            if k is not None:
                # Native dac139 stores axes after composing the joint's own
                # rotation; moving this before composition changes rounding.
                axes[k]=_quaternion_rotate(rotation,j['axis']);origins[k]=position
        t=np.eye(4);t[:3,:3]=_quaternion_matrix(rotation);t[:3,3]=position
        jp=np.stack([_cross(axis,position-origin) for axis,origin in zip(axes,origins)]).T
        return t,jp,axes.T

    def forward(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(21,) or not np.isfinite(q).all():raise ValueError('q must contain 21 finite radians')
        # The offline model is rooted at TX. Public skeletons are obtained by
        # composing its link poses with the inverse wrist pose afterwards.
        # Direct wrist-root FK loses small but meaningful rounding residuals.
        rotations={'wrist':self._tx_inverse_quaternion.copy()}
        positions={'wrist':self._tx_inverse[:3,3].copy()}
        paths={'wrist':[]};axes=np.zeros((21,3));origins=np.zeros((21,3))
        for j in self._ordered:
            parent_rotation=rotations[j['parent']]
            position=positions[j['parent']]+_quaternion_rotate(parent_rotation,j['origin'][:3,3])
            local=j['quaternion'];path=paths[j['parent']].copy();k=j['index']
            if k is not None:
                sine,cosine=_sincos(q[k]*.5)
                local=_quaternion_product(local,np.r_[j['axis']*sine,cosine])
                path.append(k)
            rotation=_quaternion_product(parent_rotation,local)
            if k is not None:
                axes[k]=_quaternion_rotate(rotation,j['axis']);origins[k]=position
            rotations[j['child']]=rotation;positions[j['child']]=position;paths[j['child']]=path
        inverse_wrist=rotations['wrist']*np.array([-1.,-1.,-1.,1.])
        inverse_position=-_quaternion_rotate(inverse_wrist,positions['wrist'])
        links={}
        for name,rotation in rotations.items():
            t=np.eye(4)
            t[:3,:3]=_quaternion_matrix(_quaternion_product(inverse_wrist,rotation))
            # Preserve compose order; rotating (position-wrist) is not the
            # same float64 calculation at nominally zero MCP coordinates.
            t[:3,3]=_quaternion_rotate(inverse_wrist,positions[name])+inverse_position
            links[name]=t
        rx=np.tile(np.eye(4),(5,1,1));jp=np.zeros((5,3,21));jw=np.zeros_like(jp)
        for f,name in enumerate(RX_NAMES):
            rx[f,:3,:3]=_quaternion_matrix(rotations[name]);rx[f,:3,3]=positions[name]
            for k in paths[name]:
                jp[f,:,k]=_cross(axes[k],positions[name]-origins[k])
                jw[f,:,k]=axes[k]
        return {'link_transforms':links,'skeleton':np.stack([links[n] for n in self.skeleton_links]),
                'tips':np.stack([links[f+'_tip'] for f in FINGERS]),'tx_rx':rx,'j_position':jp,'j_angular':jw}

    fk=forward
