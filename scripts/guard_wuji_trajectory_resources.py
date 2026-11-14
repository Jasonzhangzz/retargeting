#!/usr/bin/env python3
"""Resident GPU-memory guard bound to one owned finite campaign controller."""
import argparse,csv,fcntl,json,os,signal,subprocess,time
from pathlib import Path


def read_json(path):
    for attempt in range(8):
        try:return json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):
            if attempt==7:raise
            time.sleep(.05*(attempt+1))


def owned(path,token):
    try:return path.stat().st_uid==os.getuid() and token in (path/'cmdline').read_bytes()
    except OSError:return False


def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--master-pid',type=int,required=True);p.add_argument('--limit-mib',type=int,default=8192);a=p.parse_args();r=a.root;lock=(r/'resource_guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 master=Path('/proc')/str(a.master_pid);assert owned(master,b'launch_wuji_trajectory_campaign.py')
 signature=read_json(r/'STATUS.json')['candidate_signature'];jobs={};peak=0;errors=[];reason='controller_finished'
 previous=read_json(r/'resource_usage.json') if (r/'resource_usage.json').exists() else {}
 if previous.get('candidate_signature')==signature:
  jobs=dict(previous.get('job_peaks_mib',{}));peak=max(jobs.values(),default=0)
 while master.exists():
  try:status=read_json(r/'STATUS.json')
  except (OSError,json.JSONDecodeError) as exc:
   reason='status_read_failure';errors.append(dict(time=time.time(),error=str(exc)))
   if owned(master,b'launch_wuji_trajectory_campaign.py'):os.kill(a.master_pid,signal.SIGTERM)
   break
  if status['pid']!=a.master_pid or status['candidate_signature']!=signature:reason='binding_changed';break
  if status['status']!='running':reason=status['status'];break
  child=status.get('child_pid');proc=Path('/proc')/str(child)
  if owned(proc,b'benchmark_wuji_trajectory.py'):
   try:
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=4)
    usage=sum(int(row[1].strip()) for row in csv.reader(result.stdout.splitlines()) if int(row[0].strip())==child and row[1].strip().isdigit());job=status['active_job']['id'];jobs[job]=max(jobs.get(job,0),usage);peak=max(peak,usage)
    atomic(r/'resource_usage.json',dict(master_pid=a.master_pid,guard_pid=os.getpid(),candidate_signature=signature,limit_mib=a.limit_mib,peak_resident_mib=peak,job_peaks_mib=jobs,last_sample=time.time(),scope='nvidia-smi resident process memory, includes Torch, Warp and CUDA context; sampled every5seconds',query_errors=errors,neural_optimizer_updates=0))
    if usage>a.limit_mib:
     reason='resident_gpu_memory_limit_exceeded';atomic(r/'RESOURCE_VIOLATION.json',dict(job=job,pid=child,resident_mib=usage,limit_mib=a.limit_mib,time=time.time(),action='SIGTERM owned campaign controller'))

     if owned(master,b'launch_wuji_trajectory_campaign.py'):os.kill(a.master_pid,signal.SIGTERM)
     break
   except (subprocess.SubprocessError,ValueError) as exc:errors.append(dict(time=time.time(),error=str(exc)))
  time.sleep(5)
 atomic(r/'resource_guard_exit.json',dict(reason=reason,master_pid=a.master_pid,peak_resident_mib=peak,job_peaks_mib=jobs,query_errors=errors,neural_optimizer_updates=0))


if __name__=='__main__':main()
