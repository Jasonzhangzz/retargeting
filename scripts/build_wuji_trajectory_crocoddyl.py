#!/usr/bin/env python3
"""Build analytic C++ action models against the isolated pinned Crocoddyl wheel."""
from pathlib import Path
import argparse,subprocess,sys,sysconfig,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
site=Path(sysconfig.get_paths()['purelib']);prefix=site/'cmeel.prefix';includes=[prefix/'include',prefix/'include/eigen3',site/'pybind11/include',Path(sysconfig.get_paths()['include'])];source=r/'runtime/retarget/learning/wuji_ik/native/crocoddyl_trajectory.cpp';dest=r/'native';dest.mkdir(exist_ok=True);out=dest/('wuji_crocoddyl'+sysconfig.get_config_var('EXT_SUFFIX'))
argv=['g++','-O2','-g0','-std=c++17','-shared','-fPIC','-fopenmp',*[f'-I{x}' for x in includes],str(source),'-o',str(out),'-L'+str(prefix/'lib'),'-Wl,--disable-new-dtags','-Wl,-rpath,'+str(prefix/'lib'),'-lcrocoddyl','-lpinocchio_default']
print('BUILD',argv,flush=True);subprocess.run(argv,check=True);(dest/'build.json').write_text(json.dumps(dict(argv=argv,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),module_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),crocoddyl_version='3.2.1'),indent=2));print('BUILT',out,flush=True)
