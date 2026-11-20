"""Fixed-wrist CARD raw energy, with articulated Neural SDF collision queries.

Style and contact use CARD's actual semantic pads, source charts, CoP query,
weights and sum reductions. Collision replaces capsule/object representations
with signed distances of moving robot surfaces and observed active taxels.
Hard native-mesh acceptance stays outside the differentiable scalar objective.
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
from .cart_objective import CartObjective

VERSION = 'card_raw_neural_collision_v1'


@dataclass(frozen=True)
class CardCostSettings:
    loss_kind: str = VERSION
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
        fingerprint=digest(dict(version=VERSION,inputs=inputs,implementation=implementation))
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


class CardCostObjective(CartObjective):
    def __init__(self,hand,collision):
        super().__init__(hand,CardCostSettings(),collision)
        self.card=CardSettings();self.region_owners=[0]+[2+4*f+k for f in range(5) for k in range(3)]
        self.pad_triangles=[];self.pad_normals=[];apices=[];psi=[]
        for owner in self.region_owners:
            pad=hand.surface.contact_surface(owner);pool=hand.surface.candidates(owner)['points']
            self.pad_triangles.append(hand.tensor(pad['triangles']));self.pad_normals.append(hand.tensor(pad['normals']))
            apices.append(pool[np.argmax(pool@hand.surface.axes[owner])]);psi.append(pool.mean(0))
        self.apices=hand.tensor(apices);self.palm_psi=hand.tensor(psi[0])

    def contact_points(self,frames,targets):
        p,r=frames[:2];positions=[];normals=[];locals_=[]
        for region,owner in enumerate(self.region_owners):
            ids=targets['mask'][:,region].nonzero().flatten()
            if not len(ids):positions.append((ids,None));normals.append(None);locals_.append(None);continue
            target=targets['anchor_position'][ids,region]
            query=(r[ids,owner].transpose(-1,-2)@(target-p[ids,owner])[...,None]).squeeze(-1)
            psi=self.hand.tensor(self.hand.surface.axes[owner]*self.hand.surface.lengths[owner])[None]*targets['u'][ids,region,None]
            if region==0:psi=self.palm_psi[None].expand(len(ids),-1)
            else:
                n=self.hand.tensor(self.hand.surface.normals[owner]);pool=self.hand.tensor(self.hand.surface.candidates(owner)['points'])
                psi=psi+n[None]*((pool[None]-psi[:,None])*n).sum(-1).mean(-1)[:,None]
            alpha=.15 if region and (region-1)%3 in (1,2) else 0.
            local,index=selected_surface((query+alpha*psi)/(1+alpha),self.pad_triangles[region])
            pos=(r[ids,owner]@local[...,None]).squeeze(-1)+p[ids,owner]
            normal=(r[ids,owner]@self.pad_normals[region][index][...,None]).squeeze(-1)
            positions.append((ids,pos));normals.append(normal);locals_.append(local)
        return positions,normals,locals_

    def collision_cost(self,q,rows,frames):
        """CARD squared-mm clearance; Neural SDF replaces capsules/solid object.

        Sum all configured self sample/link queries and all active taxel/link
        queries per frame. No maximum, exponential kernel or semantic average.
        """
        oracle=self.collision.oracle;s=self.card;size=len(q)
        local,links,valid=oracle.self_queries(frames)
        poses=torch.arange(size,device=q.device)[:,None].expand_as(links)
        sdf=self.collision.distance(local,links,configuration=q,pose_ids=poses,frames=frames)
        cost_self=s.wself**2*torch.where(valid,(s.margin_mm-sdf*1000).clamp_min(0).square(),0.).sum(-1)
        tactile,tlinks,counts=oracle.flat_tactile_queries(frames,rows)
        ids=torch.arange(size,device=q.device).repeat_interleave(q.new_tensor(counts,dtype=torch.long))
        tposes=ids[:,None].expand_as(tlinks)
        values=self.collision.distance(tactile,tlinks,configuration=q,pose_ids=tposes,frames=frames)
        penalties=s.wobj_pseudo**2*(-values*1000-s.allowance_mm).clamp_min(0).square().sum(-1)
        cost_object=q.new_zeros(size).index_add(0,ids,penalties)
        depth=torch.where(valid,(-sdf).clamp_min(0),0.).amax(-1)
        if len(values):depth=depth.scatter_reduce(0,ids,(-values).clamp_min(0).amax(-1),reduce='amax',include_self=True)
        return cost_self,cost_object,depth

    def batch(self,q,rows,references=None,targets=None,acceleration=None,previous_rows=None,audit=False,normalizers=None,*,collect_metrics=True):
        if any('_card_targets' not in r for r in rows):raise ValueError('Missing verified CARD source targets')
        s=self.card;h=self.hand;size=len(q);frames=h.frames(q);p,r=frames[:2]
        c={k:torch.stack([row['_card_targets'][k] for row in rows]) for k in rows[0]['_card_targets']}
        positions=p[:,self.region_owners[1:]].reshape(size,5,3,3)
        distal=[3,6,9,12,15];owners=[self.region_owners[i] for i in distal]
        apex=p[:,owners]+(r[:,owners]@self.apices[distal,:,None]).squeeze(-1)
        chains=torch.cat((positions,apex[:,:,None,:]),2)
        dirs=F.normalize(chains[:,:,1:]-chains[:,:,:-1],dim=-1)
        pad_normals=(r[:,owners]@h.tensor(h.surface.normals[owners])[...,None]).squeeze(-1)
        free=torch.ones((size,5),device=q.device)  # CARD default contact_style_weight=1
        bone_dot=(dirs*c['directions']).sum(-1).clamp(-1,1)
        parts=dict(style_bone=s.wb**2*(1-bone_dot).mul(free[:,:,None]).sum((-1,-2)),
            style_tip=s.wtip**2*((apex-c['tips'])*1000).square().sum(-1).mul(free).sum(-1),
            style_normal=s.wn**2*(1-(pad_normals*c['normals']).sum(-1).clamp(-1,1)).mul(free).sum(-1),
            smooth_joint=s.wtheta**2*(q-references).square().sum(-1) if references is not None else q.new_zeros(size))
        selected,nn,local=self.contact_points(frames,c)
        old=None
        if references is not None and previous_rows and all(row is not None for row in previous_rows):
            old_c={k:torch.stack([row['_card_targets'][k] for row in previous_rows]) for k in c}
            _,_,old=self.contact_points(h.frames(references),old_c)
        for k in ('contact_position','contact_direction','contact_fix'):parts[k]=q.new_zeros(size)
        parts['link_continuity']=q.new_zeros(size)
        max_gap=q.new_zeros(size);max_angle=q.new_zeros(size);keep_cost=q.new_zeros(size);keep_counts=[0]*size
        for region,((ids,pos),normal,material) in enumerate(zip(selected,nn,local)):
            if pos is None:continue
            rho=1+.5*c['force'][ids,region]/c['force'][ids].amax(-1).clamp_min(1.e-20)
            error=(pos-c['anchor_position'][ids,region]).square().sum(-1)
            dot=(normal*c['anchor_normal'][ids,region]).sum(-1).clamp(-1,1)
            parts['contact_position']=parts['contact_position'].index_add(0,ids,rho.square()*s.wc**2*1.e6*error)
            parts['contact_direction']=parts['contact_direction'].index_add(0,ids,rho.square()*s.wc**2*s.direction_price*s.radius_mm**2*(1-dot))
            max_gap=max_gap.index_copy(0,ids,torch.maximum(max_gap[ids],error.detach().sqrt()))
            max_angle=max_angle.index_copy(0,ids,torch.maximum(max_angle[ids],torch.rad2deg(torch.acos(dot.detach()))))
            if old is not None and old[region] is not None:
                # Dense correspondence through row indices; no detached earlier
                # predicted state, so retention also preserves the BPTT path.
                old_ids=old_c['mask'][:,region].nonzero().flatten();dense=q.new_zeros((size,3)).index_copy(0,old_ids,old[region])
                continuous=q.new_tensor([0<row['timestamp_ms']-prev['timestamp_ms']<=250 and row.get('stream_epoch')==prev.get('stream_epoch')
                    for row,prev in zip(rows,previous_rows)],dtype=torch.bool)
                retained=old_c['mask'][ids,region]&continuous[ids]
                delta=(material-dense[ids]).square().sum(-1)
                fix=torch.where(retained,rho.square()*s.wfix**2*1.e6*delta,0.)
                parts['contact_fix']=parts['contact_fix'].index_add(0,ids,fix)
                # Keep the same semantic link in contact, while allowing the
                # material CoP to slide. Use its closest physical pad surface,
                # independently of CARD's material fix and segment-u prior.
                owner=self.region_owners[region]
                query=(r[ids,owner].transpose(-1,-2)@(c['anchor_position'][ids,region]-p[ids,owner])[...,None]).squeeze(-1)
                hit,_=selected_surface(query,self.pad_triangles[region])
                gap_mm=(query-hit).norm(dim=-1)*1000
                link=self.settings.link_continuity_weight**2*rho.square()*gap_mm.square()
                parts['link_continuity']=parts['link_continuity'].index_add(0,ids,torch.where(retained,link,0.))
                keep_cost=keep_cost.index_add(0,ids,torch.where(retained,1-torch.exp(-delta/.01**2),0.))
                for i in ids[retained].tolist():keep_counts[i]+=1
        parts['collision_self'],parts['collision_object'],depth=self.collision_cost(q,rows,frames)
        for name in ('smooth_translation','smooth_rotation','motion_joint','contact_self_closure',
                     'contact_force_vector','contact_normal_force','collision_mesh'):
            parts[name]=q.new_zeros(size)  # fixed wrist and CARD default profile
        parts['penetration_cap']=self.settings.penetration_cap_weight**2*((depth-self.settings.penetration_limit_m)*1000).clamp_min(0).square()
        total=sum(parts.values())
        means={k:v.mean() for k,v in parts.items()}
        # Familiar chart aliases; sum the primary cost parts, not their aliases.
        means.update(direction=means['style_bone'],smooth=means['smooth_joint'],contact=means['contact_position'],
            normal=means['contact_direction'],penetration=means['collision_self']+means['collision_object'],contact_keep=means['contact_fix'])
        if acceleration is not None:means['acceleration']=acceleration.square().mean()  # monitor only
        if not collect_metrics:return total.mean(),means,[]
        valid=torch.ones(size,device=q.device,dtype=torch.bool)
        if audit:depth,valid=self.collision.audit(q,rows,frames)
        feasible=valid&(depth<=self.settings.penetration_limit_m)&(max_gap<=self.settings.contact_limit_m)&(max_angle<=self.settings.normal_deg)
        feasible=feasible&(q>=h.low-1.e-6).all(-1)&(q<=h.high+1.e-6).all(-1)&torch.isfinite(total)
        dir_error=torch.rad2deg(torch.acos(bone_dot.detach())).mean((-1,-2))
        metrics=[dict(feasible=feasible[i:i+1],contact_count=int(c['mask'][i].sum()),max_gap_m=max_gap[i:i+1],
            max_normal_deg=max_angle[i:i+1],max_penetration_m=depth[i:i+1],bone_direction_deg=dir_error[i:i+1],
            query_valid=valid[i:i+1],contact_query_valid=valid[i:i+1],unresolved_contact_count=q.new_zeros(1,dtype=torch.long),
            points=[],groups={},contact_keep_count=keep_counts[i],
            contact_keep_loss=keep_cost[i:i+1]/max(1,keep_counts[i])) for i in range(size)]
        return total.mean(),means,metrics
