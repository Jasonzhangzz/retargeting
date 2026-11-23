"""CARD source targets and shared semantic-pad geometry for v9 and wrist CART.

The two supported objectives share source charts, pad geometry and CARD
weights. This module constructs immutable observed targets; the policy loss
is implemented in cart_card_neural_surface and cart_card_self_contact.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from torch.nn import functional as F

from retarget.baseline.card.math import CardSettings
from retarget.baseline.card.geometry import closest_surface_points

TARGET_VERSION = 'card_source_targets_v1'


@dataclass(frozen=True)
class CardCostSettings:
    loss_kind: str = 'card_raw_neural_self_contact_v4_actuated'
    contact_scale_m: float = .01
    contact_limit_m: float = .0005
    penetration_limit_m: float = .005
    normal_deg: float = 30.
    link_continuity_weight: float = 2.5
    penetration_cap_weight: float = 40.

    def payload(self):
        return dict(asdict(self), card=CardSettings().payload(),
                    reduction='sum_regions_bones_queries_then_mean_window_frames',
                    collision='neural_signed_surface_clearance_and_active_taxel_penetration',
                    fixed_wrist=True, source_contacts='CARD_material_anchors_no_pair_inference')


def selected_surface(query, triangles):
    """Exact CARD closest feature, with its local derivative and bounded memory.

    Discrete face/edge selection is piecewise constant. Recompute the selected
    feature projection with autograd, including the CoP motion within a face.
    """
    with torch.no_grad():
        local, index, bary = closest_surface_points(query.detach(), triangles)
        active = bary > 1.e-7
    tri = triangles[index]
    a, b, c = tri.unbind(1)
    ab, ac, delta = b-a, c-a, query-a
    aa, bb, cc = ab.square().sum(-1), (ab*ac).sum(-1), ac.square().sum(-1)
    det = (aa*cc-bb.square()).clamp_min(1.e-24)
    ad, cd = (delta*ab).sum(-1), (delta*ac).sum(-1)
    v, w = (cc*ad-bb*cd)/det, (aa*cd-bb*ad)/det
    projected = a+v[:,None]*ab+w[:,None]*ac
    for i, k in ((0,1),(1,2),(2,0)):
        edge = tri[:,k]-tri[:,i]
        t = ((query-tri[:,i])*edge).sum(-1)/edge.square().sum(-1).clamp_min(1.e-24)
        point = tri[:,i]+t.clamp(0,1)[:,None]*edge
        use = (active.sum(-1)==2)&active[:,i]&active[:,k]
        projected = torch.where(use[:,None],point,projected)
    vertex = tri[torch.arange(len(query),device=query.device),bary.argmax(-1)]
    projected = torch.where((active.sum(-1)==1)[:,None],vertex,projected)
    # Preserve the reference forward value to its original floating precision.
    return local+(projected-projected.detach()), index


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


class CardSurfaceGeometry:
    def __init__(self,hand,collision):
        self.hand=hand;self.settings=CardCostSettings();self.collision=collision
        self.vectorized=True;self.cache_rows=False
        self.card=CardSettings();self.region_owners=[0]+[2+4*f+k for f in range(5) for k in range(3)]
        self.pad_triangles=[];self.pad_normals=[];apices=[];psi=[]
        for owner in self.region_owners:
            pad=hand.surface.contact_surface(owner);pool=hand.surface.candidates(owner)['points']
            self.pad_triangles.append(hand.tensor(pad['triangles']));self.pad_normals.append(hand.tensor(pad['normals']))
            apices.append(pool[np.argmax(pool@hand.surface.axes[owner])]);psi.append(pool.mean(0))
        self.apices=hand.tensor(apices);self.palm_psi=hand.tensor(psi[0])
