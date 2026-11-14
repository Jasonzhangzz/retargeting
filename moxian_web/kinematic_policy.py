"""Official runtime configuration; retired experiments require explicit opt-in."""
from pathlib import Path
import hashlib,json,math,re,shutil,xml.etree.ElementTree as ET

POLICY_REVISION='eight_gesture_fixed_channel_registration_v13'
REGISTRATION_REVISION='eight_gesture_channel_and_mount_v2'

def replace_config_block(text,key,value):
    pattern=r'(?m)^'+re.escape(key)+r':[^\n]*\n(?:(?:[ \t].*|)[\n]|[ \t].*$)*'
    text,count=re.subn(pattern,lambda _:key+': '+json.dumps(value,allow_nan=False)+'\n',text)
    if count!=1:raise ValueError('标定配置项不唯一：'+key)
    return text

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

def load_ik_registration(runtime,calibration_sha):
    path=Path(runtime)/'ik-registration.json'
    if not path.exists():return None
    data=json.loads(path.read_text())
    if data.get('calibration_sha256')!=calibration_sha or data.get('revision')!=REGISTRATION_REVISION:return None
    if not data.get('accepted'):return None
    for name,transform in data['input_transforms'].items():
        if name not in ('thumb','index','middle','ring','pinky'):raise ValueError('未知通道')
        validate_input_transform(transform)
    return data

def apply_ik_registration(text,data):
    # JSON flow mappings are YAML; keep the realtime environment stdlib-only.
    # Replace complete top-level blocks, never use duplicate override keys.
    for key,value in data['configuration'].items():
        if key not in ('tip_mapping','solver','hand'):raise ValueError('不支持的标定配置项')
        text=replace_config_block(text,key,value)
    return text+'\n__moxian_input_registration__: '+json.dumps(data['input_transforms'],allow_nan=False)+'\n'

def load_thumb_registration(runtime,calibration_sha):
    path=Path(runtime)/'thumb-registration.json'
    if not path.exists():return None
    data=json.loads(path.read_text())
    if data.get('calibration_sha256')!=calibration_sha or data.get('policy_revision')!=POLICY_REVISION:return None
    if not data.get('accepted'):
        raise ValueError('拇指固定变换没有通过当前运动学版本的验证')
    p=data['input_tip_offset_local_m'];r=data['input_tip_rotation_offset_local']
    if len(p)!=3 or len(r)!=3 or any(len(row)!=3 for row in r):raise ValueError('拇指固定变换维数错误')
    if not all(math.isfinite(v) for v in p+[v for row in r for v in row]) or sum(v*v for v in p)>.075**2:
        raise ValueError('拇指安装偏移超出范围')
    if any(abs(sum(r[k][i]*r[k][j] for k in range(3))-(i==j))>1e-6 for i in range(3) for j in range(3)):
        raise ValueError('拇指安装旋转不是正交矩阵')
    det=sum(r[0][i]*(r[1][(i+1)%3]*r[2][(i+2)%3]-r[1][(i+2)%3]*r[2][(i+1)%3]) for i in range(3))
    if abs(det-1)>1e-6:raise ValueError('拇指安装旋转必须为 SO(3)')
    return data

def apply_thumb_registration(text,data):
    # Work only inside tip_mapping.left; never replace the right-hand mount.
    start=text.index('\n  left:',text.index('tip_mapping:'));end=re.search(r'(?m)^\S',text[start+1:])
    end=start+1+end.start() if end else len(text)
    left=text[start:end];p=json.dumps(data['input_tip_offset_local_m']);r=json.dumps(data['input_tip_rotation_offset_local'])
    pattern=r'(?m)^(    input_tip_offset_local:\n)([\s\S]*?)(?=^    \S|\Z)'
    def position(match):
        block,count=re.subn(r'(?m)^      thumb:\s*\n(?:        -[^\n]*\n?){3}',lambda _: '      thumb: '+p+'\n',match[2])
        if count!=1:raise ValueError('左手拇指安装位置配置不唯一')
        return match[1]+block
    left,count=re.subn(pattern,position,left)
    if count!=1:raise ValueError('缺少左手安装位置配置')
    pattern=r'(?m)^    input_tip_rotation_offset_local:\n'
    left,count=re.subn(pattern,lambda m:m[0]+'      thumb: '+r+'\n',left)
    if count!=1:raise ValueError('缺少左手安装方向配置')
    return text[:start]+left+text[end:]

def build_experimental_kinematic_config(text,package_root,runtime):
    """Retired local corrections, retained only for reproducing historical audits."""
    package_root=Path(package_root);runtime=Path(runtime);runtime.mkdir(parents=True,exist_ok=True)
    source=package_root/'share/qnbot_hand';urdf=source/'hands/dynamic_openxr_hand/resources/pinocchio/left_skeleton.urdf'
    calibration_sha=hashlib.sha256(text.encode()).hexdigest()
    full_registration=load_ik_registration(runtime,calibration_sha)
    registration=None if full_registration else load_thumb_registration(runtime,calibration_sha)
    registration_key=json.dumps(full_registration or registration,sort_keys=True).encode() if full_registration or registration else b''
    asset_bytes=urdf.read_bytes()+urdf.with_name('metadata.yaml').read_bytes()
    if urdf.with_name('left.urdf').exists():asset_bytes+=urdf.with_name('left.urdf').read_bytes()
    key=hashlib.sha256(POLICY_REVISION.encode()+text.encode()+asset_bytes+registration_key).hexdigest()
    destination=runtime/('resources-'+key[:16]);manifest=destination/'runtime-policy.json'
    if not manifest.exists():
        shutil.copytree(source,destination,dirs_exist_ok=True)
        target=destination/'hands/dynamic_openxr_hand/resources/pinocchio/left_skeleton.urdf';tree=ET.parse(target)
        metadata=target.with_name('metadata.yaml');mt=metadata.read_text();limits={}
        # The solver reads left.urdf, not left_skeleton.urdf. Keep both copies
        # identical in their joint tree; the latter is also used by diagnostics.
        joints={j.get('name'):j for j in tree.getroot().findall('joint')}
        axial=joints.get('left_thumb_mcp_abduction');flex=joints.get('left_thumb_mcp_flexion')
        if axial is not None and flex is not None:
            # An axial twist must rotate about the incoming metacarpal, before
            # the MCP's static flexion. Moving that fixed rotation preserves
            # every zero-DOF frame but removes a twist-induced geometric kink.
            rpy=axial.find('origin').get('rpy','0 0 0')
            if flex.find('origin').get('rpy','0 0 0')!='0 0 0':raise ValueError('不支持的拇指 MCP 零位')
            if axial.find('axis').get('xyz')!='1 0 0':raise ValueError('不支持的拇指轴向关节')
            axial.find('origin').set('rpy','0 0 0');flex.find('origin').set('rpy',rpy)
        for joint in tree.getroot().findall('joint'):
            name=joint.get('name')
            if not (any('_'+kind+'_flexion' in name for kind in ('pip','dip')) or name in ('left_thumb_mcp_flexion','left_thumb_ip_flexion')):continue
            if joint.find('axis').get('xyz')!='0 0 1':raise ValueError('未支持的官方指间关节轴，停止生成限位')
            # q=0 is already bent by URDF origin yaw. Bound physical extension,
            # not raw q: setting q>=0 would leave a 15–21 degree bent finger.
            yaw=float(joint.find('origin').get('rpy','0 0 0').split()[2]);lower=-yaw-math.radians(3.)
            joint.find('limit').set('lower',str(lower))
            pattern=r'(?m)^(\s*'+re.escape(name)+r':\s*)\[([^\]]+)\]'
            def replace(match):return match[1]+f'[{lower:.12f}, {match[2].split(",")[1].strip()}]'
            mt,count=re.subn(pattern,replace,mt)
            if count!=1:raise ValueError('官方限位元数据不匹配：'+name)
            limits[name]={'q_min_rad':lower,'origin_yaw_rad':yaw,'physical_extension_deg':3.}
        ET.indent(tree);tree.write(target,encoding='unicode');metadata.write_text(mt)
        actual=target.with_name('left.urdf')
        if actual.exists():
            solver_tree=ET.parse(actual);updated={j.get('name'):j for j in tree.getroot().findall('joint')}
            for joint in solver_tree.getroot().findall('joint'):
                peer=updated.get(joint.get('name'))
                if peer is None:raise ValueError('求解骨架与显示骨架关节不一致')
                for tag in ('origin','axis','limit'):
                    if joint.find(tag) is not None:joint.find(tag).attrib.update(peer.find(tag).attrib)
            ET.indent(solver_tree);solver_tree.write(actual,encoding='unicode')
        info=None
        if full_registration:
            info={k:full_registration[k] for k in ('revision','accepted','calibration_sha256','capture_sha256','input_transforms')}
            report=full_registration['report']
            info['quality']={k:report[k] for k in ('mean_target_error_before_mm','mean_target_error_after_mm','max_flat_thumb_bend_deg','max_pinch_gap_mm','physical_validation','validation_scope')}
        data={'revision':POLICY_REVISION,'runtime_policy_sha256':key,'calibration_sha256':calibration_sha,'physics':False,'joint_smoother':False,'open_error_reseed':False,'engine_lifetime':'persistent_across_frames','joint_limits':limits,'thumb_sensor_registration':registration,'fixed_input_registration':info,'solver_urdf':'left.urdf','thumb_axial_joint_before_static_flexion':axial is not None}
        manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    data=json.loads(manifest.read_text())
    derived=text.replace('physics_enabled: true','physics_enabled: false')
    if registration:derived=apply_thumb_registration(derived,registration)
    derived,count=re.subn(r'(joint_smoother:\s*\n\s+enabled:)\s*true',r'\1 false',derived)
    derived,count2=re.subn(r'(error_reseed:\s*\n\s+enabled:)\s*true',r'\1 false',derived)
    if count!=1 or count2!=1:raise ValueError('官方配置缺少明确的平滑／重置开关')
    if full_registration:derived=apply_ik_registration(derived,full_registration)
    derived,count3=re.subn(r'(?m)^__resource_root__:.*$',lambda _: '__resource_root__: '+json.dumps(str(destination.resolve())),derived)
    if count3!=1:raise ValueError('官方配置缺少资源根目录')
    path=runtime/f'kinematic-{calibration_sha[:16]}.yaml';temp=path.with_suffix('.tmp');temp.write_text(derived);temp.replace(path)
    return path,data


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
