"""Call the *actual* L-BFGS engine vendored in Wuji calibration, via its C ABI.

The stripped SDK does not export NLopt symbols. These entry points were
resolved from calibration's relocation slots, and are enabled ONLY for the
byte-identical, SHA256-pinned 2026.7.1 ELF. No Rust/PyO3 object is constructed,
no Python extension initializer is called, and no device/SDK session is opened.
The objective callback remains responsible for the caller's hand topology.
"""
from __future__ import annotations
import ctypes as C
import hashlib
import os
import platform
from pathlib import Path
import numpy as np

ELF_SHA256='8041b5ee059de6e8aa44641f5e7f323d0997604ff908d2dc108d18976631300c'
SDK_VERSION='2026.7.1'
NLOPT_VERSION='2.9.1'
PYINIT_RVA=0x6244f9
# Verified C functions, not guessed Rust ABIs. See native/fit_notes.md and
# sdk/optimizer_api_got_slots.json in the reverse-engineering evidence.
ENTRY_POINTS=dict(create=0x125dede,destroy=0x125d3e4,set_min_objective=0x125dca6,
    set_lower_bounds=0x125dcb0,set_upper_bounds=0x125ddbe,
    set_ftol_rel=0x125e2af,set_maxeval=0x125e444,
    set_force_stop=0x125e4a1,optimize=0x125eeef)
DOUBLE=C.POINTER(C.c_double)
CALLBACK=C.CFUNCTYPE(C.c_double,C.c_uint,DOUBLE,DOUBLE,C.c_void_p)


def library_path():
    explicit=os.environ.get('MOXIAN_WUJI_CALIBRATION_ELF')
    root=Path(__file__).resolve().parents[1]
    paths=[Path(explicit)] if explicit else [
        root/'moxian_web/vendor/wuji-calibration-native/wuji_sdk/_wuji_sdk.cpython-310-x86_64-linux-gnu.so']
    for path in paths:
        if path.is_file():return path.resolve()
    raise FileNotFoundError('缺少Wuji原生标定solver；运行 scripts/setup_wuji_calibration_solver.py，或设置 MOXIAN_WUJI_CALIBRATION_ELF')


class NativeLibrary:
    def __init__(self,path=None):
        if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
            raise RuntimeError('此固定版本的Wuji原生solver需要Linux x86_64')
        self.path=Path(path).resolve() if path else library_path()
        with self.path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        if checksum!=ELF_SHA256:raise ValueError('Wuji标定ELF指纹不匹配；拒绝调用不同版本的内部入口')
        self.library=C.CDLL(str(self.path),mode=os.RTLD_LOCAL|os.RTLD_NOW)
        base=C.cast(self.library.PyInit__wuji_sdk,C.c_void_p).value-PYINIT_RVA
        def function(name,result,*args):return C.CFUNCTYPE(result,*args)(base+ENTRY_POINTS[name])
        self.create=function('create',C.c_void_p,C.c_int,C.c_uint)
        self.destroy=function('destroy',None,C.c_void_p)
        self.objective=function('set_min_objective',C.c_int,C.c_void_p,CALLBACK,C.c_void_p)
        self.lower=function('set_lower_bounds',C.c_int,C.c_void_p,DOUBLE)
        self.upper=function('set_upper_bounds',C.c_int,C.c_void_p,DOUBLE)
        self.ftol=function('set_ftol_rel',C.c_int,C.c_void_p,C.c_double)
        self.maxeval=function('set_maxeval',C.c_int,C.c_void_p,C.c_int)
        self.force_stop=function('set_force_stop',C.c_int,C.c_void_p,C.c_int)
        self.optimize=function('optimize',C.c_int,C.c_void_p,DOUBLE,DOUBLE)
        self.provenance=dict(backend='wuji_calibration_native',sdk_version=SDK_VERSION,
            nlopt_version=NLOPT_VERSION,elf_sha256=checksum,algorithm='LD_LBFGS',algorithm_enum=10,
            source=str(self.path),entry_points={key:hex(value) for key,value in ENTRY_POINTS.items()},
            native_solver_reused=True,native_caliber_objective_reused=False)


class WujiCalibrationLbfgs:
    """Small Python adapter; all optimization steps execute in Wuji's ELF."""
    def __init__(self,dimension,path=None):
        if type(dimension) is not int or dimension<1:raise ValueError('无效优化维数')
        self.dimension=dimension;self.native=NativeLibrary(path);self.provenance=self.native.provenance
        self.lower=np.full(dimension,-np.inf);self.upper=np.full(dimension,np.inf)
        self.ftol=1e-10;self.maxeval=20000;self.callback=None;self.status=None;self.value=None

    def set_lower_bounds(self,value):self.lower=self._bounds(value)
    def set_upper_bounds(self,value):self.upper=self._bounds(value)
    def _bounds(self,value):
        array=np.ascontiguousarray(value,dtype=np.float64)
        if array.shape!=(self.dimension,) or np.isnan(array).any():raise ValueError('优化边界无效')
        return array.copy()
    def set_ftol_rel(self,value):
        if not np.isfinite(value) or value<=0:raise ValueError('无效ftol_rel')
        self.ftol=float(value)
    def set_maxeval(self,value):
        if type(value) is not int or value<1:raise ValueError('无效maxeval')
        self.maxeval=value
    def set_min_objective(self,callback):self.callback=callback
    def last_optimize_result(self):return self.status

    def optimize(self,initial):
        x=np.array(initial,dtype=np.float64,order='C',copy=True)
        if x.shape!=(self.dimension,) or not np.isfinite(x).all() or np.any(self.lower>self.upper) or np.any(x<self.lower) or np.any(x>self.upper):raise ValueError('初始向量或边界无效')
        if self.callback is None:raise ValueError('缺少objective')
        api=self.native;handle=api.create(10,self.dimension)
        if not handle:raise MemoryError('Wuji原生L-BFGS创建失败')
        failures=[]
        @CALLBACK
        def objective(n,values,gradient,_):
            try:
                if n!=self.dimension:raise RuntimeError('原生objective维数不匹配')
                xx=np.ctypeslib.as_array(values,shape=(n,)).copy()
                gg=np.ctypeslib.as_array(gradient,shape=(n,)) if gradient else np.empty(0)
                if failures:gg.fill(0);return 1e300
                result=float(self.callback(xx,gg))
                if not np.isfinite(result) or not np.isfinite(gg).all():raise ValueError('objective或梯度不是有限数')
                return result
            except BaseException as exc:
                # ctypes cannot unwind a Python exception through C. Request
                # NLopt's native forced stop, then re-raise after it returns.
                failures.append(exc)
                if gradient:np.ctypeslib.as_array(gradient,shape=(n,)).fill(0)
                api.force_stop(handle,1)
                return 1e300
        def checked(result):
            if result<0:raise ValueError('Wuji原生L-BFGS设置失败：'+str(result))
        try:
            checked(api.lower(handle,self.lower.ctypes.data_as(DOUBLE)))
            checked(api.upper(handle,self.upper.ctypes.data_as(DOUBLE)))
            checked(api.ftol(handle,self.ftol));checked(api.maxeval(handle,self.maxeval))
            checked(api.objective(handle,objective,None))
            minimum=C.c_double()
            self.status=int(api.optimize(handle,x.ctypes.data_as(DOUBLE),C.byref(minimum)))
            self.value=minimum.value
            if failures:raise failures[0]
            return x
        finally:api.destroy(handle)
