#!/usr/bin/env python3
"""Install the pinned native calibration engine, without importing the SDK."""
import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WHEEL_NAME='wuji_sdk-2026.7.1-cp310-cp310-manylinux_2_34_x86_64.whl'
WHEEL_SHA256='c19866e7c564762dcb7e6a2f8fb3eae07d04c649ae0b060e4e1b1167b9ab7a46'
URL='https://files.pythonhosted.org/packages/56/43/0d4587bfa0fadc9e08fb62dfae2f935dba32b7c98ac32c98016836f5865e/'+WHEEL_NAME
ELF_SHA256='8041b5ee059de6e8aa44641f5e7f323d0997604ff908d2dc108d18976631300c'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',type=Path)
    parser.add_argument('--destination',type=Path,default=ROOT/'moxian_web/vendor/wuji-calibration-native')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='wuji-calibration-') as temporary:
        wheel=args.wheel or Path(temporary)/WHEEL_NAME
        if not wheel.exists():urllib.request.urlretrieve(URL,wheel)
        with wheel.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        if checksum!=WHEEL_SHA256:raise ValueError('Wuji wheel SHA256不匹配')
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith('/') or not (name=='wuji_sdk/_wuji_sdk.cpython-310-x86_64-linux-gnu.so' or name.startswith('wuji_sdk.libs/') or '.dist-info/licenses/' in name):continue
                target=(args.destination/name).resolve()
                target.relative_to(args.destination.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                data=archive.read(name)
                if target.exists() and target.read_bytes()==data:continue
                # A running worker may have this ELF mmap'ed: never truncate
                # its inode while another process executes the native solver.
                with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as stream:
                    stream.write(data);pending=Path(stream.name)
                pending.replace(target)
        elf=args.destination/'wuji_sdk/_wuji_sdk.cpython-310-x86_64-linux-gnu.so'
        with elf.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        if checksum!=ELF_SHA256:raise ValueError('Wuji ELF SHA256不匹配')
        print(elf.resolve());print('SDK 2026.7.1 / vendored NLopt 2.9.1 / '+checksum)


if __name__=='__main__':main()
