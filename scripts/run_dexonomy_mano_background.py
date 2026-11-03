#!/usr/bin/env python3
"""Launch the production LM/CEM solver with bounded retries for transient CPFS I/O."""
from pathlib import Path
import errno,functools,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import retarget_dexonomy_mano as solver

def resilient(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        for attempt in range(7):
            try:return function(*args,**kwargs)
            except OSError as error:
                if error.errno not in (errno.ESTALE,errno.EAGAIN,errno.EIO) or attempt==6:raise
                print('CPFS_RETRY',function.__name__,'attempt',attempt+1,'errno',error.errno,flush=True)
                time.sleep(.05*2**attempt)
    return wrapped

for name in ('read','write','sha','source_groups','export_shard'):
    setattr(solver,name,resilient(getattr(solver,name)))

if __name__=='__main__':solver.main()
