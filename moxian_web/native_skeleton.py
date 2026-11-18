"""Read-only ctypes binding to the supplied QnBot hand C ABI 2.1.

Consumes observations only; no serial ports or hardware commands are exposed.
Struct definitions follow the installed qnbot_hand/c/api.h, not SDK internals.
"""
from __future__ import annotations
import ctypes as C
import json
import math
import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

def joint_limit_diagnostics(names,positions,limits,tolerance_rad=.01):
    """Report saturation, without modifying official joint commands."""
    result=[]
    for name,value in zip(names,positions):
        if name not in limits:continue
        lo,hi=limits[name]
        if value-lo<=tolerance_rad:result.append({'joint':name,'bound':'lower','value_rad':value,'limit_rad':lo})
        elif hi-value<=tolerance_rad:result.append({'joint':name,'bound':'upper','value_rad':value,'limit_rad':hi})
    return result

class View(C.Structure):
    _fields_=[('data',C.c_char_p),('size',C.c_size_t)]
    @classmethod
    def text(cls,text):
        data=str(text).encode();return cls(data,len(data))
class Pose(C.Structure):
    _fields_=[('position',(C.c_double*3)),('rotation',(C.c_double*9))]
class GloveFrame(C.Structure):
    _fields_=[('size',C.c_uint32),('abi',C.c_uint32),('glove',View),('timestamp',C.c_double),('side',C.c_int),
              ('local_mask',C.c_uint32),('world_mask',C.c_uint32),('root',Pose),('local',Pose*5),('world',Pose*5)]
class EngineOptions(C.Structure):
    _fields_=[('size',C.c_uint32),('abi',C.c_uint32),('config',View),('bone_config',View),('algo_config',View)]
class SdkOptions(C.Structure):
    _fields_=[('size',C.c_uint32),('abi',C.c_uint32),('resource_root',View)]
class JointBuffer(C.Structure):
    _fields_=[('size',C.c_uint32),('abi',C.c_uint32),('timestamp',C.c_double),('side',C.c_int),('count',C.c_size_t),('capacity',C.c_size_t),('positions',C.POINTER(C.c_double))]
class SkeletonPose(C.Structure):
    _fields_=[('position',C.c_double*3),('quaternion',C.c_double*4)]
class SkeletonBuffer(C.Structure):
    _fields_=[('size',C.c_uint32),('abi',C.c_uint32),('timestamp',C.c_double),('side',C.c_int),('count',C.c_size_t),('capacity',C.c_size_t),('poses',C.POINTER(SkeletonPose))]
class Target(C.Structure):
    _fields_=[('size',C.c_uint32),('abi',C.c_uint32),('glove',View),('timestamp',C.c_double),('side',C.c_int),('mask',C.c_uint32),('pinch_mask',C.c_uint32),
              ('wrist',Pose),('tips',Pose*5),('tips_local',Pose*5),('human',((C.c_double*3)*5)),('shape_weights',C.c_double*5),('pinch_targets',(C.c_double*3)*5),('pinch_weights',C.c_double*5),('orientation_weights',C.c_double*5)]

def quaternion_matrix(q):
    if len(q)!=4 or not all(math.isfinite(v) for v in q):raise ValueError('Invalid quaternion')
    x,y,z,w=q;n=sum(v*v for v in q)**.5
    if n<1e-8:raise ValueError('Zero quaternion')
    x,y,z,w=[v/n for v in q]
    return [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]

def matrix_quaternion(m):
    trace=m[0]+m[4]+m[8]
    if trace>0:
        s=math.sqrt(trace+1)*2;return [(m[7]-m[5])/s,(m[2]-m[6])/s,(m[3]-m[1])/s,s/4]
    i=max(range(3),key=lambda k:m[4*k]);j=(i+1)%3;k=(i+2)%3
    s=math.sqrt(max(1+m[4*i]-m[4*j]-m[4*k],1e-15))*2;q=[0.]*4
    q[i]=s/4;q[j]=(m[3*i+j]+m[3*j+i])/s;q[k]=(m[3*i+k]+m[3*k+i])/s;q[3]=(m[3*k+j]-m[3*j+k])/s
    return q

def registered_observation(node,transform=None):
    """Transform a private native input, leaving raw observations untouched."""
    p=list(node['position']);r=quaternion_matrix(node['quaternion_xyzw'])
    if transform:
        a=transform['rotation'];b=transform['translation_m']
        p=[sum(a[i][j]*p[j] for j in range(3))+b[i] for i in range(3)]
        r=[sum(a[i][k]*r[3*k+j] for k in range(3)) for i in range(3) for j in range(3)]
    return p,r

class NativeSkeleton:
    def __init__(self,package_root,config_path,side="left"):
        from .hand_side import check_side
        self.side=check_side(side);self.side_id=1 if side=="left" else 0
        # The native library names its generated URDF cache after the config
        # basename, ignoring the parent directory. Independent calibrations
        # previously raced on evaluation_left/right.urdf. Give every process
        # and configuration a distinct cache key; preserve source files.
        contents=Path(config_path).read_text()
        self.config_directory=tempfile.TemporaryDirectory(prefix='qnbot-native-')
        config_path=Path(self.config_directory.name)/f'native-{os.getpid()}-{hashlib.sha256(contents.encode()).hexdigest()[:24]}.yaml'
        config_path.write_text(contents);self.runtime_config_path=config_path
        self.previous_joints=None;self.previous_timestamp=None;self.frames=0
        self.lib=C.CDLL(str(Path(package_root)/'lib/libqnbot_hand.so'));lib=self.lib;self.handle=C.c_void_p();self.sdk_context=C.c_void_p()
        def sig(name,args,result):
            f=getattr(lib,name);f.argtypes=args;f.restype=result;return f
        ptr=C.c_void_p
        sig('qh_abi_version_major',[],C.c_uint32)
        if lib.qh_abi_version_major()!=2:raise ValueError('Unsupported QnBot hand ABI')
        sig('qh_engine_create',[C.POINTER(EngineOptions),C.POINTER(ptr)],C.c_int)
        sig('qh_sdk_create',[C.POINTER(SdkOptions),C.POINTER(ptr)],C.c_int)
        sig('qh_sdk_create_engine',[ptr,View,View,View,C.POINTER(ptr)],C.c_int)
        sig('qh_sdk_destroy',[ptr],None);sig('qh_sdk_last_error',[ptr],C.c_char_p)
        sig('qh_engine_destroy',[ptr],None);sig('qh_engine_last_error',[ptr],C.c_char_p)
        sig('qh_last_error',[],C.c_char_p)
        for name in ['joint_count','skeleton_node_count','physics_step_count']:
            sig('qh_engine_'+name,[ptr],C.c_size_t)
        for name in ['physics_enabled','physics_contact_count','physics_actuator_count','physics_collision_geom_count']:
            sig('qh_engine_'+name,[ptr],C.c_int)
        sig('qh_engine_skeleton_node_name',[ptr,C.c_size_t],C.c_char_p)
        sig('qh_engine_joint_name',[ptr,C.c_size_t],C.c_char_p)
        sig('qh_engine_step',[ptr,C.POINTER(GloveFrame),C.POINTER(JointBuffer)],C.c_int)
        sig('qh_engine_get_last_target',[ptr,C.POINTER(Target)],C.c_int)
        sig('qh_engine_get_last_skeleton_pose',[ptr,C.POINTER(SkeletonBuffer)],C.c_int)
        resource_root=None;self.input_registration={}
        for line in Path(config_path).read_text().splitlines():
            if line.startswith('__resource_root__:'):
                value=line.split(':',1)[1].strip()
                resource_root=json.loads(value) if value.startswith('"') else value
            if line.startswith('__moxian_input_registration__:'):
                from .kinematic_policy import validate_input_transform
                self.input_registration=json.loads(line.split(':',1)[1])
                for name,transform in self.input_registration.items():
                    if name not in ('thumb','index','middle','ring','pinky'):raise ValueError('Unknown registered channel')
                    validate_input_transform(transform)
        if resource_root:
            # A prior SDK session can retain its default resource context. Use
            # an explicit C ABI context so this engine owns the derived limits.
            options=SdkOptions(C.sizeof(SdkOptions),2,View.text(resource_root))
            status=lib.qh_sdk_create(C.byref(options),C.byref(self.sdk_context))
            if status:raise RuntimeError((lib.qh_last_error() or b'Native SDK context creation failed').decode())
            config=View.text(Path(config_path).resolve())
            status=lib.qh_sdk_create_engine(self.sdk_context,View.text('dynamic_openxr_hand'),config,config,C.byref(self.handle))
        else:
            options=EngineOptions(C.sizeof(EngineOptions),2,View.text(config_path),View(),View())
            status=lib.qh_engine_create(C.byref(options),C.byref(self.handle))
        if status:
            error=lib.qh_sdk_last_error(self.sdk_context) if self.sdk_context else lib.qh_last_error()
            self.close();raise RuntimeError((error or b'Native engine creation failed').decode())
        self.joint_count=lib.qh_engine_joint_count(self.handle);self.node_count=lib.qh_engine_skeleton_node_count(self.handle)
        self.positions=(C.c_double*self.joint_count)();self.nodes=(SkeletonPose*self.node_count)()
        self.names=[lib.qh_engine_skeleton_node_name(self.handle,i).decode() for i in range(self.node_count)]
        self.joint_names=[lib.qh_engine_joint_name(self.handle,i).decode() for i in range(self.joint_count)]
        self.joint_side_initialized=False
        resource=Path(resource_root) if resource_root else Path(package_root)/'share/qnbot_hand'
        urdf=resource/f'hands/dynamic_openxr_hand/resources/pinocchio/{self.side}_skeleton.urdf'
        self.joint_limits={}
        for joint in ET.parse(urdf).getroot().findall('joint'):
            limit=joint.find('limit')
            if limit is not None:self.joint_limits[joint.get('name')]=(float(limit.get('lower')),float(limit.get('upper')))

    def check(self,status):
        if status:raise RuntimeError((self.lib.qh_engine_last_error(self.handle) or b'Native hand error').decode())

    def step(self,frame):
        if frame.get('hand_side',self.side)!=self.side:raise ValueError('输入帧与原生引擎手侧不匹配')
        if not math.isfinite(frame['timestamp_ms']):raise ValueError('Invalid millisecond timestamp')
        for name in ('thumb','index','middle','ring','pinky'):
            p=frame['fingers'][name]['position']
            if len(p)!=3 or not all(math.isfinite(v) for v in p):raise ValueError('Invalid metre position')
        f=GloveFrame();f.size=C.sizeof(f);f.abi=2;f.glove=View.text('moxian');f.timestamp=frame['timestamp_ms']/1000.;f.side=self.side_id;f.local_mask=31
        f.root.rotation[:]=[1,0,0,0,1,0,0,0,1]
        for i,name in enumerate(('thumb','index','middle','ring','pinky')):
            p,r=registered_observation(frame['fingers'][name],self.input_registration.get(name))
            f.local[i].position[:]=p;f.local[i].rotation[:]=r
        output=JointBuffer(C.sizeof(JointBuffer),2,0,self.side_id,0,self.joint_count,self.positions)
        self.check(self.lib.qh_engine_step(self.handle,C.byref(f),C.byref(output)))
        if not self.joint_side_initialized:
            # The package defaults to right; first left frame selects the actual
            # hand. Query names after that switch so diagnostics name left DOFs.
            self.joint_names=[self.lib.qh_engine_joint_name(self.handle,i).decode() for i in range(self.joint_count)]
            if not all(n.startswith(self.side+'_') for n in self.joint_names):raise ValueError('原生引擎未切换到 '+self.side)
            self.joint_side_initialized=True
        target=Target();target.size=C.sizeof(target);target.abi=2
        self.check(self.lib.qh_engine_get_last_target(self.handle,C.byref(target)))
        bones=SkeletonBuffer(C.sizeof(SkeletonBuffer),2,0,self.side_id,0,self.node_count,self.nodes)
        self.check(self.lib.qh_engine_get_last_skeleton_pose(self.handle,C.byref(bones)))
        current=list(self.positions);steps=[]
        if self.previous_joints is not None:
            steps=[{'joint':name,'delta_deg':math.degrees(now-old)} for name,old,now in zip(self.joint_names,self.previous_joints,current) if name.startswith(self.side+'_thumb_')]
        self.frames+=1
        continuity={'engine_frames':self.frames,'reused_engine':self.frames>1,'interval_ms':None if self.previous_timestamp is None else frame['timestamp_ms']-self.previous_timestamp,
                    'thumb_joint_steps':steps,'max_thumb_step_deg':max((abs(s['delta_deg']) for s in steps),default=0.)}
        self.previous_joints=current;self.previous_timestamp=frame['timestamp_ms']
        return {'continuity':continuity,'joint_limit_diagnostics':joint_limit_diagnostics(self.joint_names,current,self.joint_limits),
                'pose':{'timestamp_ms':frame['timestamp_ms'],'hand_side':self.side,'coordinate_frame':'wrist_local','names':self.names,
                        'joint_names':self.joint_names,'joint_positions_rad':list(self.positions),'positions_local':[list(n.position) for n in self.nodes],'orientations_local_xyzw':[list(n.quaternion) for n in self.nodes]},
                'targets':{'timestamp_ms':frame['timestamp_ms'],'hand_side':self.side,'coordinate_frame':'wrist_local',
                           'names':['thumb','index','middle','ring','pinky'],
                           'positions_local':[list(n.position) for n in target.tips_local],
                           'rotations_local':[list(n.rotation) for n in target.tips_local],
                           'quaternions_local_xyzw':[matrix_quaternion(n.rotation) for n in target.tips_local],
                           'human_positions_local':[list(n) for n in target.human],'pinch_mask':target.pinch_mask,
                           'wrist_position':list(target.wrist.position),'wrist_rotation':list(target.wrist.rotation),
                           'pinch_weights':list(target.pinch_weights),'shape_weights':list(target.shape_weights)},
                'physics':{'enabled':bool(self.lib.qh_engine_physics_enabled(self.handle)),
                           'steps':self.lib.qh_engine_physics_step_count(self.handle),'contacts':self.lib.qh_engine_physics_contact_count(self.handle),
                           'actuators':self.lib.qh_engine_physics_actuator_count(self.handle)}}

    def close(self):
        if self.handle:self.lib.qh_engine_destroy(self.handle);self.handle=C.c_void_p()
        if self.sdk_context:self.lib.qh_sdk_destroy(self.sdk_context);self.sdk_context=C.c_void_p()
        if self.config_directory:self.config_directory.cleanup();self.config_directory=None
