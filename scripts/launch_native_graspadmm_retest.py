#!/usr/bin/env python3
"""Wait for the owned pinned-environment installer, then run the frozen protocol once."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback
import native_graspadmm_retest as runner
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--setup-pid',type=int,required=True)
a=p.parse_args();root=a.root.resolve();upstream=root/'upstream'
lock=(root/'.launcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
production=root.parent/'retarget';source_manifest=root.parent/'retarget_solver_compare_20261104/output/native_admm_inputs_20261104/manifest.json'
proc=Path('/proc',str(a.setup_pid));identity=None
if proc.exists():
    if str(root/'setup_native_graspadmm.sh').encode() not in (proc/'cmdline').read_bytes():raise RuntimeError('Setup PID identity mismatch')
    identity=(proc/'stat').read_text().split()[21]
runner.write(root/'launcher.json',dict(pid=os.getpid(),identity=Path('/proc/self/stat').read_text().split()[21],
    setup_pid=a.setup_pid,setup_identity=identity,script_sha256=runner.sha(__file__),started_utc=runner.stamp()))
try:
    while proc.exists() and (proc/'stat').read_text().split()[21]==identity:
        runner.stop_check(root);runner.status(root,'environment_setup',setup_pid=a.setup_pid);time.sleep(10)
    python=root/'venv/bin/python'
    check='import importlib.metadata,json; print(json.dumps({k:importlib.metadata.version(k) for k in '+repr(runner.VERSIONS)+'}))'
    result=subprocess.run([str(python),'-c',check],check=True,capture_output=True,text=True)
    versions=json.loads(result.stdout)
    for name,expected in runner.VERSIONS.items():
        if versions[name].split('+')[0]!=expected:raise RuntimeError('Environment incomplete: '+name)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    runner.status(root,'prepare')
    command=[str(python),str(root/'native_graspadmm_retest.py'),'--root',str(root),'--upstream',str(upstream),'--production',str(production)]
    if not (root/'initialization.json').exists():
        subprocess.run(command+['--source-manifest',str(source_manifest),'--stage','prepare'],check=True)
    subprocess.run(command+['--stage','all'],check=True)
except BaseException as error:
    runner.status(root,'failed',error=str(error),traceback=traceback.format_exc());raise
