#!/usr/bin/env python3
"""Bounded completion handoff: copy the finished experiment and start local Viser."""
import argparse,fcntl,hashlib,json,os,shlex,signal,socket,subprocess,sys,time
from pathlib import Path


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fingerprint',required=True)
    p.add_argument('--remote-root',default='/mnt/cpfs/zbl-cpfs-new/USERS/jasonzhang/retarget_mano_native_20261104');p.add_argument('--port',type=int,default=8104)
    p.add_argument('--timeout',type=int,default=3600);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.delivery.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    repo=Path(__file__).resolve().parents[1];remote=a.remote_root
    env=os.environ.copy();env['SSH_AUTH_SOCK']='/run/user/1000/keyring/.ssh'
    ssh=['ssh','-p','1016','-o','BatchMode=yes','-o','ConnectTimeout=15','8.147.67.185']
    deadline=time.monotonic()+a.timeout;failures=0
    receipt=dict(pid=os.getpid(),start_identity=Path('/proc/self/stat').read_text().split()[21],expected_manifest_sha256=a.fingerprint,timeout_seconds=a.timeout,source=remote)
    query="import pathlib,json,hashlib;p=pathlib.Path("+repr(remote)+");print(json.dumps(dict(status=json.loads((p/'status.json').read_text()),fingerprint=hashlib.sha256((p/'MANIFEST.json').read_bytes()).hexdigest())))"
    while time.monotonic()<deadline:
        try:
            raw=subprocess.check_output(ssh+['python3 -c '+shlex.quote(query)],env=env,text=True,timeout=30)
            data=json.loads(raw);assert data['fingerprint']==a.fingerprint,'Experiment fingerprint changed'
            status=data['status'];write(out/'delivery_status.json',dict(receipt,state='waiting_for_experiment',remote_status=status))
            failures=0
            if status['stage']=='failed':raise RuntimeError('Remote experiment failed: '+json.dumps(status,ensure_ascii=False))
            if status['stage']=='complete':break
        except (subprocess.SubprocessError,OSError,json.JSONDecodeError) as e:
            failures+=1
            write(out/'delivery_status.json',dict(receipt,state='transport_retry',failures=failures,error=str(e)))
            if failures>=3:raise
        time.sleep(30)
    else:raise TimeoutError('Completion handoff timed out; remote experiment was not stopped')
    (out/'results').mkdir(exist_ok=True)
    rsync=['rsync','-az','--timeout=60','-e','ssh -p 1016 -o BatchMode=yes -o ConnectTimeout=15']
    subprocess.run(rsync+['--include=/*.json','--include=/*.md','--include=/*.csv','--exclude=*','8.147.67.185:'+remote+'/',str(out/'results')+'/'],env=env,timeout=600,check=True)
    subprocess.run(rsync+['8.147.67.185:'+remote+'/viewer/',str(out/'viewer')+'/'],env=env,timeout=600,check=True)
    assert hashlib.sha256((out/'results/MANIFEST.json').read_bytes()).hexdigest()==a.fingerprint
    manifest=json.loads((out/'viewer/manifest.json').read_text());assert 'mano' in manifest['model'].lower()
    server_path=out/'viewer/server.json'
    if server_path.exists():
        previous=json.loads(server_path.read_text());pid=previous['pid'];proc=Path('/proc',str(pid))
        if proc.exists():
            command=(proc/'cmdline').read_bytes().split(b'\0')
            if str(repo/'scripts/serve_native_graspadmm.py').encode() not in command or str(out/'viewer').encode() not in command:
                raise RuntimeError('Existing viewer PID belongs to another process; not restarting it')
            os.kill(pid,signal.SIGTERM)
            for _ in range(100):
                if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
                time.sleep(.05)
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(('127.0.0.1',a.port));sock.close()
    with (out/'viewer/server.log').open('ab') as log:
        process=subprocess.Popen([str(repo/'.venv-graspadmm/bin/python'),str(repo/'scripts/serve_native_graspadmm.py'),'--data',str(out/'viewer'),'--port',str(a.port)],cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(40):
        if process.poll() is not None:raise RuntimeError('Viser startup failed')
        server_path=out/'viewer/server.json'
        if server_path.exists() and json.loads(server_path.read_text()).get('pid')==process.pid:break
        time.sleep(.5)
    else:raise RuntimeError('Viser did not write a startup receipt')
    write(out/'delivery_status.json',dict(receipt,state='complete',viewer_pid=process.pid,url=f'http://127.0.0.1:{a.port}/',samples=manifest['samples']))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if '--output' in sys.argv:
            target=Path(sys.argv[sys.argv.index('--output')+1]).resolve();target.mkdir(parents=True,exist_ok=True)
            write(target/'delivery_status.json',dict(state='failed',error=str(e),pid=os.getpid()))
        raise
