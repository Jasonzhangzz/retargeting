"""Derived official runtime assets; original SDK/calibration files stay immutable."""
from pathlib import Path
import hashlib,json,math,re,shutil,xml.etree.ElementTree as ET

POLICY_REVISION='physical_extension_no_open_reseed_low_latency_v11'

def build_kinematic_config(text,package_root,runtime):
    package_root=Path(package_root);runtime=Path(runtime);runtime.mkdir(parents=True,exist_ok=True)
    source=package_root/'share/qnbot_hand';urdf=source/'hands/dynamic_openxr_hand/resources/pinocchio/left_skeleton.urdf'
    calibration_sha=hashlib.sha256(text.encode()).hexdigest()
    key=hashlib.sha256(POLICY_REVISION.encode()+text.encode()+urdf.read_bytes()+urdf.with_name('metadata.yaml').read_bytes()).hexdigest()
    destination=runtime/('resources-'+key[:16]);manifest=destination/'runtime-policy.json'
    if not manifest.exists():
        shutil.copytree(source,destination,dirs_exist_ok=True)
        target=destination/'hands/dynamic_openxr_hand/resources/pinocchio/left_skeleton.urdf';tree=ET.parse(target)
        metadata=target.with_name('metadata.yaml');mt=metadata.read_text();limits={}
        for joint in tree.getroot().findall('joint'):
            name=joint.get('name')
            if not any('_'+kind+'_flexion' in name for kind in ('pip','dip')):continue
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
        data={'revision':POLICY_REVISION,'runtime_policy_sha256':key,'calibration_sha256':calibration_sha,'physics':False,'joint_smoother':False,'open_error_reseed':False,'joint_limits':limits}
        manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    data=json.loads(manifest.read_text())
    derived=text.replace('physics_enabled: true','physics_enabled: false')
    derived,count=re.subn(r'(joint_smoother:\s*\n\s+enabled:)\s*true',r'\1 false',derived)
    derived,count2=re.subn(r'(error_reseed:\s*\n\s+enabled:)\s*true',r'\1 false',derived)
    if count!=1 or count2!=1:raise ValueError('官方配置缺少明确的平滑／重置开关')
    derived,count3=re.subn(r'(?m)^__resource_root__:.*$',lambda _: '__resource_root__: '+json.dumps(str(destination.resolve())),derived)
    if count3!=1:raise ValueError('官方配置缺少资源根目录')
    path=runtime/f'kinematic-{calibration_sha[:16]}.yaml';temp=path.with_suffix('.tmp');temp.write_text(derived);temp.replace(path)
    return path,data
