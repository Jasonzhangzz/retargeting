"""Pinned Wuji 2026.7.1 calibration ABI; full fit runs only in a child process.

The functions and layouts below are checked against this exact ELF's caller,
collector and return-value assembly. No SDK manager/device session is created.
"""
from __future__ import annotations
import ctypes as C
import struct
import numpy as np
from .wuji_calibration_solver import NativeLibrary,PYINIT_RVA,ELF_SHA256

SDK_POSES=('pinch_index','pinch_middle','pinch_ring','pinch_pinky','four_finger_bend_90','flat_open')
FINGERS=('thumb','index','middle','ring','pinky')
FULL_FIT_RVA=0xbc6d8e
HASH_RVA=0xfadaca
LABELS=dict(pinch_index='拇指与食指指腹对指',pinch_middle='拇指与中指指腹对指',
    pinch_ring='拇指与无名指指腹对指',pinch_pinky='拇指与小指指腹对指',
    four_finger_bend_90='四指根关节弯曲90°，其余指节伸直',flat_open='五指伸直并拢，手掌展平')


class NativeCalibrationABI:
    def __init__(self):
        self.api=NativeLibrary();self.base=C.cast(self.api.library.PyInit__wuji_sdk,C.c_void_p).value-PYINIT_RVA
        self.hash=C.CFUNCTYPE(C.c_uint64,C.c_uint64,C.c_uint64,C.c_void_p,C.c_size_t)(self.base+HASH_RVA)
        self.fit=C.CFUNCTYPE(None,C.c_void_p,C.c_void_p,C.c_uint8,C.c_uint8)(self.base+FULL_FIT_RVA)
        self.quaternion=C.CFUNCTYPE(None,C.c_void_p,C.c_void_p)(self.base+0xed1215)
        self.euler=C.CFUNCTYPE(None,C.c_void_p,C.c_void_p)(self.base+0xecff63)
        self.poses=[]
        # Same six-name filter, applied to the SDK's eight-entry original order.
        for index in range(8):
            values=struct.unpack('<6Q',C.string_at(self.base+0x13b7b20+48*index,48))
            name=C.string_at(values[0],values[1]).decode()
            if name not in SDK_POSES:continue
            constraints=[]
            for j in range(values[5]):
                raw=C.string_at(values[4]+j*56,56);kind=struct.unpack_from('<I',raw)[0]
                if kind==0:
                    distance=struct.unpack_from('<f',raw,4)[0];a,b=struct.unpack_from('<QQ',raw,8)
                    constraints.append(dict(kind='distance',fingers=[a,b],distance_m=distance))
                elif kind==1:
                    ranges=[]
                    for off in (4,16,28):
                        enabled,lo,hi=struct.unpack_from('<Iff',raw,off);ranges.append([lo,hi] if enabled else None)
                    pointer,count=struct.unpack_from('<QQ',raw,40)
                    ids=list(struct.unpack('<'+'Q'*count,C.string_at(pointer,count*8)))
                    constraints.append(dict(kind='orientation_rpy',fingers=ids,ranges_deg=ranges))
                else:raise ValueError('固定版本的默认动作约束发生变化')
            self.poses.append(dict(id=name,label=LABELS[name],constraints=constraints))
        if tuple(p['id'] for p in self.poses)!=SDK_POSES:raise ValueError('Wuji默认六动作表不匹配')

    def rpy_degrees(self,xyzw):
        raw=(C.c_float*4)(*xyzw);unit=(C.c_double*4)();result=(C.c_double*3)()
        self.quaternion(unit,raw);self.euler(result,unit)
        return (np.asarray(result)*57.29577951308232).astype(np.float32)

    def check_pose(self,stage,fingers,side):
        definition=next(p for p in self.poses if p['id']==stage);checks=[]
        for constraint in definition['constraints']:
            if constraint['kind']=='distance':
                a,b=constraint['fingers'];d=np.asarray(fingers[FINGERS[a]]['position'],np.float32)-np.asarray(fingers[FINGERS[b]]['position'],np.float32)
                value=float(np.sqrt(np.float32(np.float32(d[0]*d[0]+d[1]*d[1])+d[2]*d[2])))
                maximum=constraint['distance_m'];checks.append(dict(label='接收器距离mm',value=value*1000,maximum=maximum*1000,passed=value<=maximum))
            else:
                for f in constraint['fingers']:
                    rpy=self.rpy_degrees(fingers[FINGERS[f]]['quaternion_xyzw'])
                    for axis,limits in enumerate(constraint['ranges_deg']):
                        if limits is None:continue
                        lo,hi=limits
                        if side=='left' and axis in (0,2):lo,hi=-hi,-lo
                        value=float(rpy[axis]);checks.append(dict(label=FINGERS[f]+(' roll°',' pitch°',' yaw°')[axis],value=value,minimum=lo,maximum=hi,passed=lo<=value<=hi))
        failures=[c['label']+'：'+f"{c['value']:.1f}"+'，要求'+(f"[{c['minimum']:.1f}, {c['maximum']:.1f}]" if 'minimum' in c else f"≤{c['maximum']:.1f}") for c in checks if not c['passed']]
        return dict(accepted=not failures,checks=checks,failures=failures,source='wuji_native_pose_table_and_quaternion_math',sdk_version='2026.7.1')

    def fit_profile(self,poses,side,profile):
        """Consumes native-owned input; call only in an isolated child process."""
        if tuple(poses)!=SDK_POSES or side not in ('left','right') or profile not in (0,1):raise ValueError('Wuji原生标定输入无效')
        libc=C.CDLL(None);libc.malloc.argtypes=[C.c_size_t];libc.malloc.restype=C.c_void_p
        def malloc(n):
            p=libc.malloc(n)
            if not p:raise MemoryError('原生标定输入分配失败')
            C.memset(p,0,n);return p
        def write(p,b):C.memmove(p,b,len(b))
        def words(p,*v):write(p,struct.pack('<'+'Q'*len(v),*v))
        slots=16;table=malloc(slots*48+32);ctrl=table+slots*48;C.memset(ctrl,255,32);used=set()
        for name,frames in poses.items():
            if not frames or len(frames)>2000:raise ValueError('原生静态动作采样数量无效')
            key=name.encode();keyp=malloc(len(key));write(keyp,key);framep=malloc(64*len(frames))
            for i,nodes in enumerate(frames):
                if len(nodes)!=5:raise ValueError('原生标定需要五个接收器')
                pp=malloc(5*32)
                for f,node in enumerate(nodes):
                    q=np.asarray(node['quaternion_xyzw'],float);p=np.asarray(node['position'],float)
                    if q.shape!=(4,) or p.shape!=(3,) or not np.isfinite(np.r_[q,p]).all() or np.linalg.norm(q)<1e-8:raise ValueError('无效EMF位姿')
                    write(pp+32*f,struct.pack('<8f',*q,*p,1.))
                words(framep+64*i+8,1);words(framep+64*i+40,5,pp,5)
            h=self.hash(0,0,keyp,len(key));slot=h&15
            while slot in used:slot=(slot+1)&15
            used.add(slot);tag=h>>57
            write(ctrl+slot,bytes([tag]));write(ctrl+slot+16,bytes([tag]))
            words(ctrl-(slot+1)*48,len(key),keyp,len(key),len(frames),framep,len(frames))
        mapping=(C.c_uint64*6)(ctrl,15,14-len(poses),len(poses),0,0);out=C.create_string_buffer(4096)
        self.fit(out,C.byref(mapping),int(side=='left'),profile)
        # Result's Ok contains wrist-root and TX-root URDF Strings. Result Err
        # uses a high-bit niche; never interpret it as a pointer/length tuple.
        result={}
        for offset,label in ((0,'wrist_urdf'),(24,'tx_urdf')):
            cap,p,n=struct.unpack_from('<QQQ',out.raw,offset)
            if cap>>63 or not p or not 0<n<=cap<10000000:raise RuntimeError('Wuji原生标定未返回成功结果')
            value=C.string_at(p,n).decode('utf-8')
            if '<robot ' not in value or '</robot>' not in value:raise RuntimeError('Wuji原生URDF结果无效')
            result[label]=value
        return result
