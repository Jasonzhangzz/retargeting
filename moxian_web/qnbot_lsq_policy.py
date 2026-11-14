"""Resolve the actual SDK seed policy in its isolated IK interpreter."""
import argparse,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def describe(home,side,fingerprint):
    from qnbot_sdk.glove._resources import builtin_skeleton_package_root
    from moxian_web.kinematic_policy import build_kinematic_config
    from moxian_web.thumb_coordinates import build_thumb_coordinate_config
    from moxian_web.emf_slsqp import SLSQPSkeleton
    home=Path(home);os.environ['QNBOT_SDK_HOME']=str(home)
    artifacts=list(home.glob('glove/calibrations/*/artifacts/hand_system.yaml'))
    source=next((p for p in artifacts if hashlib.sha256(p.read_bytes()).hexdigest()==fingerprint),None)
    if source is None:raise ValueError('历史记录缺少匹配初始化原件，不能伪造骨架版本')
    text=source.read_text();package=builtin_skeleton_package_root()
    if side=='left':derived,policy=build_thumb_coordinate_config(text,package,home/'runtime')
    else:derived,policy=build_kinematic_config(text,package,home/'runtime',side=side)
    derived.write_text(derived.read_text()+'\n__moxian_slsqp__: true\n')
    personalized=None;profile_path=home.parents[1]/'hand_model'/home.name/'current.json'
    if profile_path.exists():
        from moxian_web.hand_model import calibration_report_valid
        saved=json.loads(profile_path.read_text())
        if saved.get('calibration_fingerprint')==fingerprint and saved.get('side')==side and saved.get('calibrated') and calibration_report_valid(saved.get('report',{})):
            personalized=saved.get('skeleton_seed_model') if saved.get('runtime_backend') in ('fixed_mano_v1','gpu_mano_v1') else saved.get('skeleton_model')
    engine=SLSQPSkeleton(package,derived,side=side,personalized=personalized)
    try:actual=engine.policy(policy)
    finally:engine.close()
    return dict(calibration_fingerprint=fingerprint,skeleton_runtime_revision=actual['runtime_policy_sha256'],config=str(derived),source=str(source),side=side)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',type=Path,required=True);p.add_argument('--side',choices=('left','right'),required=True);p.add_argument('--fingerprint',required=True);a=p.parse_args()
    print(json.dumps(describe(a.home,a.side,a.fingerprint)))
