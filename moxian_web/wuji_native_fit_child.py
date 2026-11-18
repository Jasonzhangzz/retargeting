"""Disposable full-native fitter. Native allocations are released at process exit."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from moxian_web.wuji_native_abi import NativeCalibrationABI,SDK_POSES

def main():
    source=Path(sys.argv[1]);destination=Path(sys.argv[2]);data=json.loads(source.read_text())
    abi=NativeCalibrationABI();results={};began=time.monotonic()
    for profile,name in ((0,'wujihand'),(1,'wujihand2')):
        print(json.dumps(dict(event='profile',profile=name)),flush=True)
        poses={name:data['pose_frames'][name] for name in SDK_POSES}
        results[name]=abi.fit_profile(poses,data['side'],profile)
    destination.write_text(json.dumps(dict(profiles=results,active_hand_profile='wujihand',seconds=time.monotonic()-began)))

if __name__=='__main__':main()
