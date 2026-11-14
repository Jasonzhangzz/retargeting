"""Shared scratch cache for immutable source meshes across hand variants."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np


class GeometryCache:
    def __init__(self, loader, directory, budget_bytes=8*1024**3):
        self.loader=loader;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.budget=budget_bytes

    def get(self, entry):
        identity=hashlib.sha256((str(self.loader.dataset)+':'+entry['scene_path']).encode()).hexdigest()
        path=self.directory/(identity+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as a:
                value=json.loads(str(a['metadata']));value.update(vertices=a['vertices'],faces=a['faces'])
            stat=Path(value['source_mesh']).stat()
            if [stat.st_size,stat.st_mtime_ns]!=value.pop('source_mesh_stat'):
                raise ValueError('Immutable source geometry changed: '+value['source_mesh'])
            return (value['source_mesh'],tuple(value['base_scale'])),value
        key,value=self.loader.get(entry)
        temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        metadata={k:v for k,v in value.items() if k not in ('vertices','faces')}
        stat=Path(value['source_mesh']).stat();metadata['source_mesh_stat']=[stat.st_size,stat.st_mtime_ns]
        with temporary.open('wb') as f:
            np.savez(f,vertices=value['vertices'],faces=value['faces'],metadata=json.dumps(metadata))
        # A bounded, shared cache is an optimization, never a source of labels.
        # Atomic publication makes simultaneous first reads safe.
        import fcntl
        with (self.directory/'budget.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=self.directory/'budget.json'
            used=json.loads(state.read_text())['bytes'] if state.exists() else 0
            if not path.exists() and used+temporary.stat().st_size<=self.budget:
                size=temporary.stat().st_size;os.replace(temporary,path)
                state_tmp=state.with_name('budget.'+str(os.getpid())+'.tmp')
                state_tmp.write_text(json.dumps(dict(bytes=used+size)));os.replace(state_tmp,state)
        temporary.unlink(missing_ok=True)
        return key,value
