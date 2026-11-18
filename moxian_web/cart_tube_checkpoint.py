"""Append-only frame checkpoints and provisional paths for long tube jobs."""
import json
from pathlib import Path

import numpy as np

from retarget.baseline.geort.training import atomic_json


def prefix_tube(events,metadata):
    nodes=[event['node'] for event in events];paths=[];end=None
    def finish():
        nonlocal end
        if end is None:return
        i=end;k=int(np.argmin(nodes[i]['cumulative_costs']));path=[]
        while True:
            path.append((i,k));parent=nodes[i]['parents'][k]
            if parent is None:break
            i-=1;k=parent
        paths.append(path[::-1]);end=None
    for i,node in enumerate(nodes):
        if not node['q']:finish();continue
        if node['parents'][0] is None:finish()
        end=i
    component_end=end
    finalized=len(paths)
    finish()
    trajectory=[None]*len(nodes);segments=[None]*len(nodes)
    for segment,path in enumerate(paths):
        for i,k in path:trajectory[i]=nodes[i]['q'][k];segments[i]=segment
    return dict(policy=metadata['policy'],settings=metadata['settings'],nodes=nodes,
        edges=[node['edges'] for node in nodes],paths=paths,trajectory=trajectory,segments=segments,
        gaps=[event['gap'] for event in events if event['gap'] is not None],
        boundaries=[event['boundary'] for event in events if event['boundary'] is not None],
        frames=len(nodes),total_frames=metadata['total_frames'],completed_frames=len(nodes),complete=False,
        feasible_frames=sum(q is not None for q in trajectory),input_fingerprint=metadata['input_fingerprint'],
        provisional_path=component_end is not None,
        resume_state=dict(component_end=component_end,finalized_paths_count=finalized,rng_state=metadata['rng_state']),
        construction='causal_constraints_inside_each_cem_generation',posthoc_connection=False,
        history_used_for_sampling=False,device=metadata.get('device'),gpu_parallel=metadata.get('gpu_parallel'),
        swept_check='adaptive_samples_not_continuous_intersection_proof')


class JournalReader:
    def __init__(self,folder):
        self.folder=Path(folder);self.events=[];self.offset=0;self.identity=None;self.signature=None;self.result=None

    def load(self):
        path=self.folder/'cart_tube_progress.json'
        signature=path.stat().st_mtime_ns
        if signature==self.signature:return self.result
        metadata=json.loads(path.read_text());offset=metadata['byte_offset']
        identity=(metadata['input_fingerprint'],metadata['generation'])
        if identity!=self.identity or offset<self.offset:
            self.events=[];self.offset=0;self.identity=identity
        if offset>self.offset:
            with (self.folder/'cart_tube_nodes.jsonl').open('rb') as stream:
                stream.seek(self.offset);raw=stream.read(offset-self.offset)
            if len(raw)!=offset-self.offset or not raw.endswith(b'\n'):
                raise ValueError('Tube 检查点尚未完整写入，请稍后刷新')
            incoming=[json.loads(line) for line in raw.splitlines()]
            for event in incoming:
                if event['index']!=len(self.events):raise ValueError('Tube 检查点帧号不连续')
                self.events.append(event)
        if len(self.events)!=metadata['completed_frames']:raise ValueError('Tube 检查点帧数不一致')
        self.offset=offset;self.signature=signature;self.result=prefix_tube(self.events,metadata)
        return self.result


class TubeCheckpoint:
    def __init__(self,folder,fingerprint,settings,total_frames,policy,device):
        import uuid
        self.folder=Path(folder);self.path=self.folder/'cart_tube_progress.json';self.resume=None
        self.metadata=dict(input_fingerprint=fingerprint,settings=settings,policy=policy,total_frames=total_frames,
            device=str(device),gpu_parallel=str(device).startswith('cuda'),completed_frames=0,byte_offset=0,
            generation=uuid.uuid4().hex,rng_state=None,complete=False)
        try:
            metadata=json.loads(self.path.read_text())
            if all(metadata.get(k)==self.metadata[k] for k in ('input_fingerprint','settings','policy','total_frames')):
                self.resume=JournalReader(folder).load();self.metadata=metadata
        except (OSError,ValueError,KeyError):pass
        self.stream=(self.folder/'cart_tube_nodes.jsonl').open('r+b' if self.resume is not None else 'w+b')
        self.stream.truncate(self.metadata['byte_offset']);self.stream.seek(self.metadata['byte_offset'])
        atomic_json(self.path,self.metadata)

    def __call__(self,index,node,boundary,gap,rng_state):
        if index!=self.metadata['completed_frames']:raise ValueError('Tube 检查点写入顺序错误')
        event=dict(index=index,node=node,boundary=boundary,gap=gap)
        self.stream.write((json.dumps(event,allow_nan=False,separators=(',',':'))+'\n').encode());self.stream.flush()
        self.metadata.update(completed_frames=index+1,byte_offset=self.stream.tell(),rng_state=rng_state.cpu().tolist())
        atomic_json(self.path,self.metadata)

    def close(self):self.stream.close()
