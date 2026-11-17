"""Official runtime configuration and fixed input-transform validation."""
from pathlib import Path
import hashlib,json,math,re

def validate_input_transform(transform):
    """A channel registration is SE(3), never a reflection or pose-dependent warp."""
    r=transform['rotation'];p=transform['translation_m']
    if len(r)!=3 or any(len(row)!=3 for row in r) or len(p)!=3:
        raise ValueError('通道固定变换维数错误')
    if not all(math.isfinite(v) for v in p+[v for row in r for v in row]) or sum(v*v for v in p)>.5**2:
        raise ValueError('通道固定变换超出范围')
    if any(abs(sum(r[k][i]*r[k][j] for k in range(3))-(i==j))>1e-6 for i in range(3) for j in range(3)):
        raise ValueError('通道旋转不是正交矩阵')
    det=sum(r[0][i]*(r[1][(i+1)%3]*r[2][(i+2)%3]-r[1][(i+2)%3]*r[2][(i+1)%3]) for i in range(3))
    if abs(det-1)>1e-6:raise ValueError('通道旋转必须为 SO(3)')
    return transform


OFFICIAL_POLICY_REVISION='official_ik_unmodified_physics_off_v1'

def build_kinematic_config(text,package_root,runtime,side="left"):
    """Use the official calibration and assets without local IK corrections.

    The sole behavioral exception is the user's existing no-dynamics setting.
    A resource path is resolved to the installed vendor assets. Saved local
    registrations are deliberately never read, even if they claim acceptance.
    """
    package_root=Path(package_root).resolve();runtime=Path(runtime);runtime.mkdir(parents=True,exist_ok=True)
    source=package_root/'share/qnbot_hand'
    if '__moxian_input_registration__' in text:
        raise ValueError('官方模式必须读取原始标定文件，不能读取本地派生配置')
    calibration_sha=hashlib.sha256(text.encode()).hexdigest()
    derived,count=re.subn(r'(?m)^([ \t]*physics_enabled:[ \t]*)(true|false)([ \t]*(?:#.*)?)$',r'\1false\3',text)
    if not count:raise ValueError('官方配置缺少动力学开关，无法保证仅运行 IK')
    resource_line='__resource_root__: '+json.dumps(str(source))
    if re.search(r'(?m)^__resource_root__:',derived):
        derived,n=re.subn(r'(?m)^__resource_root__:.*$',lambda _:resource_line,derived)
        if n!=1:raise ValueError('官方资源根目录不唯一')
    else:derived=derived.rstrip()+'\n'+resource_line+'\n'
    assets={}
    for relative in (f'hands/dynamic_openxr_hand/resources/pinocchio/{side}.urdf',
                     f'hands/dynamic_openxr_hand/resources/pinocchio/{side}_skeleton.urdf',
                     'hands/dynamic_openxr_hand/resources/pinocchio/metadata.yaml',
                     'hands/dynamic_openxr_hand/pose_library.yaml'):
        assets[relative]=hashlib.sha256((source/relative).read_bytes()).hexdigest()
    configuration_sha=hashlib.sha256(derived.encode()).hexdigest()
    key=hashlib.sha256(OFFICIAL_POLICY_REVISION.encode()+derived.encode()+json.dumps(assets,sort_keys=True).encode()).hexdigest()

    def vendor_enabled(section):
        block=re.search(r'(?m)^[ \t]*'+re.escape(section)+r':\s*\n((?:[ \t]+[^\n]*\n|\n)*)',text)
        setting=re.search(r'(?m)^\s+enabled:\s*(true|false)',block[1]) if block else None
        return setting[1]=='true' if setting else None

    data={'mode':'official','revision':OFFICIAL_POLICY_REVISION,'runtime_policy_sha256':key,
          'calibration_sha256':calibration_sha,'configuration_sha256':configuration_sha,
          'local_ik_modifications':False,'input_registration':False,'fixed_input_registration':None,
          'thumb_sensor_registration':None,'joint_limits_source':'vendor_original',
          'resource_root':str(source),'asset_sha256':assets,'physics':False,
          'joint_smoother':vendor_enabled('joint_smoother'),'open_error_reseed':vendor_enabled('error_reseed'),
          'engine_lifetime':'persistent_across_frames',
          'exceptions':['physics_disabled_by_user'],'joint_limits':{}}
    path=runtime/f'kinematic-{calibration_sha[:16]}.yaml';temporary=path.with_suffix('.tmp');temporary.write_text(derived);temporary.replace(path)
    manifest=runtime/('official-ik-'+key[:16]+'.json');temporary=manifest.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(manifest)
    return path,data
