#!/usr/bin/env python3
"""Install pinned full-calibration and Python 3.12 offline-IK runtimes."""
import hashlib,subprocess,sys,tempfile,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAME='wuji_sdk-2026.7.1-cp312-cp312-manylinux_2_34_x86_64.whl'
SHA='89f4421167daf892adae2f57eade363047b403966ae3956b57e7f10762b11698'
URL='https://files.pythonhosted.org/packages/79/38/c1cd916749b78728fe6310c099bd66f1bb149a12369c026ea6d5d06e1044/'+NAME

def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/setup_wuji_calibration_solver.py')],check=True)
    subprocess.run(['/usr/bin/python3.12','-c','import sys; assert sys.version_info[:2]==(3,12)'],check=True)
    destination=ROOT/'moxian_web/vendor/wuji-calibration-native/python312'
    with tempfile.TemporaryDirectory() as temp:
        wheel=Path(temp)/NAME
        if not wheel.exists():urllib.request.urlretrieve(URL,wheel)
        if hashlib.sha256(wheel.read_bytes()).hexdigest()!=SHA:raise ValueError('Wuji SDK wheel指纹不匹配')
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith('/') or not (name.startswith(('wuji_sdk/','wuji_sdk.libs/')) or '.dist-info/licenses/' in name):continue
                path=(destination/name).resolve();path.relative_to(destination.resolve());path.parent.mkdir(parents=True,exist_ok=True);data=archive.read(name)
                if path.exists() and path.read_bytes()==data:continue
                with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:stream.write(data);pending=Path(stream.name)
                pending.replace(path)
    print(destination)

if __name__=='__main__':main()
