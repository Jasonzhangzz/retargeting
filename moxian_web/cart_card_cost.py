"""Immutable native CARD source targets for the current normalized reach cost."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
TARGET_VERSION = 'card_source_targets_v1'

def attach_card_targets(root, sessions, device, check=lambda:None, progress=None):
    """A separate, checksummed target cache; existing capture/IK bytes untouched."""
    from .cart_data import file_hash
    from .qnbot_surface import QnbotSurface
    from .wuji_contact_source import MoxianContactSource, MoxianCARDSource
    from retarget.baseline.card.retarget import TactileRetarget
    from retarget.baseline.geort.adapter import SourceAdapter, digest
    reports=[]
    for meta, rows, _ in sessions:
        folder=Path(root)/'recordings'/meta['id']
        source_root=Path(__file__).resolve().parents[1]
        inputs={n:file_hash(folder/n) for n in ('motion.npz','frames.jsonl','profile.json','topology.json','cart_targets.jsonl')}
        implementation={n:file_hash(source_root/n) for n in ('moxian_web/cart_card_cost.py','moxian_web/wuji_contact_source.py',
            'moxian_web/qnbot_surface.py','retarget/baseline/card/retarget.py','retarget/baseline/card/geometry.py',
            'retarget/baseline/card/math.py')}
        fingerprint=digest(dict(version=TARGET_VERSION,inputs=inputs,implementation=implementation))
        path=folder/('card_cost_targets_'+fingerprint[:16]+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as saved: arrays={k:saved[k] for k in saved.files}
            if str(arrays.pop('fingerprint'))!=fingerprint:raise ValueError('CARD target cache fingerprint mismatch')
        else:
            topology=json.loads((folder/'topology.json').read_text());profile=json.loads((folder/'profile.json').read_text())
            adapter=SourceAdapter(topology);material=MoxianContactSource(topology);extractor=MoxianCARDSource(material)
            source=QnbotSurface(topology['hand_side'],device,finger_twist=True)
            source.load_artifact(profile);source.fixed=True
            with np.load(folder/'motion.npz',allow_pickle=False) as motion:qpos=motion['qpos']
            if len(qpos)!=len(rows):raise ValueError('CARD motion/observations length mismatch')
            arrays={k:np.zeros(shape,np.float32) for k,shape in dict(tips=(len(rows),5,3),normals=(len(rows),5,3),
                directions=(len(rows),5,3,3),anchor_position=(len(rows),16,3),anchor_normal=(len(rows),16,3),
                force=(len(rows),16),u=(len(rows),16)).items()}
            arrays['mask']=np.zeros((len(rows),16),bool)
            arrays['timestamps']=np.array([r['timestamp_ms'] for r in rows],np.float64)
            owner=SimpleNamespace(extractor=extractor,tensor=lambda a:np.asarray(a))
            # The extractor's source() normally also constructs unused envelope
            # diagnostics. Supply exactly its mirrored physical geometry here.
            geometry=[]
            extractor.source=lambda frame:geometry[0]
            with (folder/'frames.jsonl').open() as stream:
                for start in range(0,len(rows),64):
                    check();q=torch.as_tensor(qpos[start:start+64],device=device,dtype=torch.float32)
                    with torch.no_grad():vertices,joints,_=source(q)
                    vertices=vertices.cpu().numpy();joints=joints.cpu().numpy()
                    for offset,(v,j) in enumerate(zip(vertices,joints)):
                        i=start+offset;raw=json.loads(next(stream))
                        if raw['timestamp_ms']!=rows[i]['timestamp_ms']:raise ValueError('CARD raw timestamp mismatch')
                        p,n=material.taxels(v);sj=j[material.landmarks]@material.mirror.T;sv=v@material.mirror.T
                        geometry[:]=[(sv,sj,p,n)]
                        _,_,anchors=extractor.extract(raw)
                        target=TactileRetarget.style_targets(owner,sv,sj)
                        canonical=(sj-sj[0])@adapter.basis
                        if not np.allclose(canonical,rows[i]['joints_m'],atol=2.e-6):raise ValueError('CARD source reconstruction disagrees with capture')
                        arrays['tips'][i]=(target['tips']-sj[0])@adapter.basis
                        arrays['normals'][i]=target['normals']@adapter.basis
                        arrays['directions'][i]=target['directions']@adapter.basis
                        for a in anchors:
                            k=a['region'];arrays['mask'][i,k]=True
                            arrays['anchor_position'][i,k]=(np.asarray(a['position_m'])-sj[0])@adapter.basis
                            arrays['anchor_normal'][i,k]=np.asarray(a['direction'])@adapter.basis
                            arrays['force'][i,k]=a['force'];arrays['u'][i,k]=a['u']
                    if progress:progress('card_targets',dataset=meta['id'],completed=min(start+64,len(rows)),total=len(rows))
                if stream.readline():raise ValueError('CARD raw journal has extra frames')
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as output:np.savez(output,fingerprint=fingerprint,**arrays)
            temp.replace(path);del source
        if len(arrays['timestamps'])!=len(rows) or not np.array_equal(arrays['timestamps'],[r['timestamp_ms'] for r in rows]):
            raise ValueError('CARD cached target timestamps mismatch')
        tensors={k:torch.as_tensor(v,device=device) for k,v in arrays.items() if k!='timestamps'}
        for i,row in enumerate(rows):row['_card_targets']={k:v[i] for k,v in tensors.items()}
        reports.append(dict(dataset=meta['id'],fingerprint=fingerprint,sha256=file_hash(path),frames=len(rows)))
    return reports
