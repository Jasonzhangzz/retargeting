"""CARD contact-anchored MANO -> native WUJI2, fixed-budget GPU CEM."""
import math,time,hashlib,json,uuid
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from retarget.env.grasp.pose_gpu import ModelFK,WujiPoseAdapter
from .wuji_surface import WujiSurface,mirror_matrix,unit
from .card_geometry import AnchorExtractor,PseudoObject,contact_weights,segment_gap,closest_surface_points,source_wrist
from .card_math import CardSettings,ALGORITHM,cem,log_so3,continuous_euler_xyz
from .robokit_initializer import RobokitInitializer
from .wuji_collision import WujiMeshCollision
from .wuji_tactile import pinch_hypothesis,pinch_seed
from .contact_force import fit_coulomb_force

RetargetSettings=CardSettings


class TactileRetarget:
    def __init__(self,topology,cache=None,device='cuda:0',settings=None,rebuild=False,object_sdf=None):
        self.settings=settings or CardSettings();self.device=torch.device(device);self.source_topology=topology
        self.surface=WujiSurface();self.model=self.surface.model
        if topology.get('surface_backend')=='qnbot':
            from .wuji_contact_source import MoxianContactSource,MoxianCARDSource,robot_mapping
            material=MoxianContactSource(topology);self.mapping=robot_mapping(material,self.surface,cache,rebuild)
            self.mirror=material.mirror;self.extractor=MoxianCARDSource(material)
        else:
            self.mapping=self.surface.mapping(topology,cache,rebuild)
            self.mirror=mirror_matrix(topology);self.extractor=AnchorExtractor(topology,self.mirror)
        self.patches=self.extractor.patches
        self.adapter=WujiPoseAdapter(self.model,self.device)
        for name in ('site_local','low','high','neutral','robot_origin','robot_basis'):setattr(self.adapter,name,getattr(self.adapter,name).float())
        self.adapter.site_ids=torch.tensor(self.adapter.site_ids,device=self.device)
        self.fk=ModelFK(self.model,self.device,torch.float32);self.adapter.fk=self.fk
        self.seed_fk=ModelFK(self.model,torch.device('cpu'),torch.float32)
        self.initializer=RobokitInitializer(self.surface,self.adapter)
        self.mesh_collision=WujiMeshCollision(self.surface,self.device)
        self.region_owners=np.array([0]+[2+4*f+k for f in range(5) for k in range(3)])
        self.region_body=self.tensor(np.array(self.surface.body_ids)[self.region_owners],torch.long)
        self.pools=[];self.pool_normals=[];self.pool_squares=[];self.pad_triangles=[];self.pad_face_ids=[];self.pad_normals=[];self.apices=[];collision_points=[];collision_owners=[]
        cap_a=[];cap_b=[];radii=[]
        for region,owner in enumerate(self.region_owners):
            pool=self.surface.candidates(int(owner))
            if not len(pool['points']):raise ValueError(f'No palmar search surface for CARD region {region}')
            points=pool['points'];normal=pool['normals'];self.pools.append(self.tensor(points));self.pool_normals.append(self.tensor(normal));self.pool_squares.append(self.tensor((points*points).sum(-1)))
            pad=self.surface.contact_surface(int(owner));self.pad_triangles.append(self.tensor(pad['triangles']))
            self.pad_face_ids.append(self.tensor(pad['face_ids']+self.surface.face_offsets[owner],torch.long));self.pad_normals.append(self.tensor(pad['normals']))
            axis=self.surface.axes[owner];tip=points[np.argmax(points@axis)];self.apices.append(tip)
            # Deterministic distributed pad samples, independent of contact activation.
            chosen=np.linspace(0,len(points)-1,min(64,len(points)),dtype=int)
            collision_points.extend(points[chosen]);collision_owners.extend([region]*len(chosen))
            verts=np.asarray(self.surface.meshes[owner].vertices);z=verts@axis;radial=verts-z[:,None]*axis
            radius=float(np.max(np.linalg.norm(radial,axis=1)))
            # Capsule axial extents stay inside the link's native longitudinal bounds.
            length=max(float(z.max()-z.min()),1e-6);radius=min(radius,length/2)
            lo=float(z.min()+radius);hi=float(z.max()-radius)
            cap_a.append(axis*lo);cap_b.append(axis*hi);radii.append(radius)
        self.apices=self.tensor(np.array(self.apices));self.cap_a=self.tensor(np.array(cap_a));self.cap_b=self.tensor(np.array(cap_b));self.radii=self.tensor(radii)
        self.collision_local=self.tensor(np.array(collision_points));self.collision_regions=self.tensor(collision_owners,torch.long)
        self.cross_pairs=[(i,j) for i in range(1,16) for j in range(i+1,16) if (i-1)//3!=(j-1)//3]
        self.palm_pairs=[(0,i) for i in range(1,16) if (i-1)%3>0] # proximal links are mechanically attached
        self.pair_a=self.tensor([p[0] for p in self.cross_pairs+self.palm_pairs],torch.long);self.pair_b=self.tensor([p[1] for p in self.cross_pairs+self.palm_pairs],torch.long)
        records=self.mapping['records'];self.owners=np.array([r['owner'] if r else 0 for r in records]);self.local=np.array([r['local_position_m'] if r else [0,0,0] for r in records]);self.normal=np.array([r['local_normal'] if r else [0,1,0] for r in records])
        self.external_sdf=object_sdf;self.lower_np=self.adapter.low.cpu().numpy();self.upper_np=self.adapter.high.cpu().numpy()
        self.topology_version=self.topology()["topology_version"]
        self.robokit_cem=None
        self.reset()

    def tensor(self,a,dtype=torch.float32):return torch.as_tensor(np.asarray(a) if isinstance(a,(list,tuple)) else a,device=self.device,dtype=dtype)
    def topology(self):
        model=self.surface.topology(self.mapping)
        descriptor=self.external_sdf.descriptor() if self.external_sdf is not None else None
        geometry={k:v for k,v in descriptor.items() if k!='transform'} if descriptor else None
        version=hashlib.sha256(json.dumps([self.mapping["version"],self.settings.payload(),geometry],sort_keys=True).encode()).hexdigest()[:24]
        return {**model,"algorithm":ALGORITHM,"topology_version":version,"settings":self.settings.payload(),"object_sdf":descriptor,"initializer_quality":{k:v for k,v in self.initializer.artifact.items() if k not in ("matrix","bias")},"sampling_geometry":{"search_candidates_per_region":[len(p) for p in self.pools],"object_samples_per_region":[int((self.collision_regions==i).sum()) for i in range(16)]},
            "contact_surfaces":{"policy":"continuous_native_semantic_triangles","physical_pad_verified":False,"face_ids":[p.cpu().tolist() for p in self.pad_face_ids],"region_owners":self.region_owners.tolist()},
            "mesh_collision":{"hard_limit_mm":self.settings.self_penetration_mm,"coarse_samples":len(self.mesh_collision.samples[False][0]),"dense_samples":len(self.mesh_collision.samples[True][0]),"mechanical_MCP_collar_radii_m":self.mesh_collision.collar_radii_m.tolist(),"tactile_exemptions":False,"continuous_intersection_proof":False},
            "capsules":{"region_owners":self.region_owners.tolist(),"a_m":self.cap_a.cpu().tolist(),"b_m":self.cap_b.cpu().tolist(),"radii_m":self.radii.cpu().tolist()}}

    def reset(self):
        self.session_id=uuid.uuid4().hex
        self.initializer.reset()
        self.previous=None;self.previous_initial_joint=None;self.previous_source_wrist=None;self.previous_timestamp=None;self.last_seen_timestamp=None;self.previous_contacts={};self.force_max=0.;self.last_output=None
        self.sdf=self.external_sdf or PseudoObject(self.device);self.generator=torch.Generator(device=self.device).manual_seed(self.settings.seed)
        self.offline_anchors=None;self.offline_force_max=None
        self.self_contact_pairs=[];self.self_contact_targets={};self.contact_regime=None;self.last_external_stamp=None
        self.paired_local={};self.paired_normal={};self.paired_candidates={};self.contact_association={}
        self.source_calibration=None
    def source(self,frame):return self.extractor.source(frame)

    def update_object_transform(self,frame):
        if self.external_sdf is None or frame.get('object_transform') is None:return
        transform=np.asarray(frame['object_transform'],float)
        if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(transform[:3,:3])<.99 or not np.allclose(transform[3],[0,0,0,1]):raise ValueError('Invalid tracked object rigid transform')
        self.sdf.transform=transform

    def frames(self,t,r,theta):
        q=torch.cat((torch.zeros((*theta.shape[:-1],6),device=theta.device,dtype=theta.dtype),theta),-1)
        p,rr=self.fk.frames(q)
        return t[...,None,:]+(r[...,None,:,:]@p[...,None]).squeeze(-1),r[...,None,:,:]@rr

    def positions(self,q):
        from retarget.env.grasp.pose_gpu import euler_xyz_to_matrix
        p,r=self.frames(q[:3],euler_xyz_to_matrix(q[3:6]),q[6:]);ids=self.tensor(np.array(self.surface.body_ids)[self.owners],torch.long)
        return p[ids]+(r[ids]@self.tensor(self.local)[...,None]).squeeze(-1),(r[ids]@self.tensor(self.normal)[...,None]).squeeze(-1)

    def style_targets(self,vertices,joints):
        normals=[];apices=[]
        for f in range(5):
            region=3+3*f;o,x,n,tri=self.extractor.chart(region,vertices,joints)
            coords=tri.reshape(-1,3);apices.append(coords[np.argmax((coords-o)@x)])
            normals.append(n)
        bones=joints[1:].reshape(5,4,3)
        return dict(tips=self.tensor(apices),normals=self.tensor(normals),directions=self.tensor(unit(np.diff(bones,axis=1))))

    def context(self,vertices,joints,anchors,initial=None):
        target=self.style_targets(vertices,joints);active=np.zeros(16,bool)
        for a in anchors:active[a['region']]=True
        free=np.array([self.settings.contact_style_weight if active[1+3*f:4+3*f].any() else 1. for f in range(5)])
        motion_joint=None
        if (self.settings.wmotion_joint>0 and initial is not None and self.previous is not None and self.previous_initial_joint is not None
                and not self.initializer.last_report.get('history_reset_reason')):
            motion_joint=(self.previous[2]+initial[2]-self.previous_initial_joint).clamp(self.adapter.low[6:],self.adapter.high[6:])
        return dict(**target,vertices=vertices,joints=joints,anchors=anchors,active=active,free=self.tensor(free),
                    sdf=self.sdf,previous=self.previous,motion_joint=motion_joint,previous_source_wrist=self.previous_source_wrist,self_pairs=self.self_contact_pairs,pair_targets=self.self_contact_targets,
                    paired_local=self.paired_local,paired_normal=self.paired_normal,paired_candidates=self.paired_candidates)

    def update_self_contacts(self,frame,anchors,initial=None):
        if not self.settings.experimental_self_contact:
            self.self_contact_pairs=[];self.self_contact_targets={}
            self.paired_local={};self.paired_normal={};self.paired_candidates={}
            self.contact_association=dict(policy='source_material_anchors_only',selected=None,candidates=[],
                reason='no_pair_inference_no_midpoint_replacement')
            return set()
        active={a['region'] for a in anchors};pairs=[];targets={}
        for pair in frame.get('self_contact_pairs') or []:
            regions=pair.get('regions') or []
            if len(regions)==2 and regions[0]!=regions[1] and set(regions)<=active:
                key=tuple(sorted(map(int,regions)));pairs.append(key)
                if pair.get('point_a_m') is not None and pair.get('point_b_m') is not None:
                    centre=(np.asarray(pair['point_a_m'])+pair['point_b_m'])*.5
                    if centre.shape==(3,) and np.isfinite(centre).all():targets[key]=self.tensor(centre@self.mirror.T)
        self.paired_local={};self.paired_normal={};self.paired_candidates={}
        if anchors:
            if initial is None:initial=self.initializer(self.extractor.source(frame)[1])
            p,r=self.frames(*initial);native={};local={};normal={}
            response=np.clip(np.nan_to_num(np.asarray((frame.get('source_contact') or {}).get('normalized') or [0.]*460)),0,1)
            for a in anchors:
                region=a['region'];ids=[i for i in a['sids'] if self.mapping['records'][i] is not None]
                if not ids:continue
                weights=response[ids];weights=weights if weights.sum()>1e-12 else np.ones(len(ids))
                centre=self.tensor(np.average(self.local[ids],axis=0,weights=weights))
                index=(self.pools[region]-centre).square().sum(-1).argmin()
                local[region]=self.pools[region][index];normal[region]=self.pool_normals[region][index]
                body=self.region_body[region];native[region]=((p[body]+r[body]@local[region]).cpu().numpy(),(r[body]@normal[region]).cpu().numpy())
            pair,self.contact_association=pinch_hypothesis(anchors,native,self.external_sdf is not None)
            existing=next((old for old in pairs if 3 in old and all(region in (3,6,9,12,15) for region in old)),None)
            if existing is not None:self.contact_association.update(selected=list(existing),reason='upstream_bilateral_material_contact')
            elif pair is not None and not any(3 in old for old in pairs):pairs.append(pair)
            elif any(3 in old for old in pairs):self.contact_association['selected']=None
            for a,b in pairs:
                if a and b:
                    for region in (a,b):
                        if region in local:self.paired_local[region]=local[region];self.paired_normal[region]=normal[region]
                        ids=[i for anchor in anchors if anchor['region']==region for i in anchor['sids'] if self.mapping['records'][i] is not None]
                        if region in local and ids:
                            # Pressure locates a patch, not an exact robot CoP.
                            # Restrict material search to that sensed footprint.
                            radius=min(.008,max(.003,float(np.linalg.norm(self.local[ids]-local[region].cpu().numpy(),axis=1).max())+.0015))
                            distance=(self.pools[region]-local[region]).square().sum(-1)
                            candidates=torch.argsort(distance);candidates=candidates[distance[candidates]<=radius**2][:32]
                            self.paired_candidates[region]=(self.pools[region][candidates],self.pool_normals[region][candidates])
        else:self.contact_association=dict(policy='bilateral_tactile_robokit_pinch_v1',reason='no_active_anchors',selected=None,candidates=[])
        updated=sorted(set(pairs))
        if updated!=self.self_contact_pairs:
            changed={r for pair in self.self_contact_pairs+updated for r in pair}
            self.previous_contacts={r:p for r,p in self.previous_contacts.items() if r not in changed}
        self.self_contact_pairs=updated
        self.self_contact_targets=targets
        return {r for pair in pairs for r in pair}

    @torch.no_grad()
    def evaluate(self,t,r,theta,c,details=False,search=False):
        s=self.settings;batch=len(theta);full_p,full_r=self.frames(t,r,theta);p=full_p[:,self.region_body];rr=full_r[:,self.region_body]
        mesh_depth,mesh_valid=self.mesh_collision.query(full_p,full_r,dense=details)
        mesh_max=mesh_depth.amax((-1,-2));hard_mesh=(mesh_max>s.self_penetration_mm)|~mesh_valid
        zero=theta.new_zeros(batch);parts={k:zero.clone() for k in ('contact_position','contact_direction','contact_force_vector','contact_normal_force','contact_fix','contact_self_closure','style_tip','style_normal','style_bone','smooth_joint','smooth_translation','smooth_rotation','motion_joint','collision_self','collision_mesh','collision_object')}
        parts['collision_mesh']=s.wmesh**2*mesh_depth.square().sum((-1,-2))
        selected={};selected_normals={};local_selected={};material_selected={};force_support={};targets={a['region']:self.tensor(a['position_m']) for a in c['anchors']}
        # A bilaterally supported hand-hand contact is one shared contact
        # location. It is not the boundary of a solid pseudo-object.
        for a,b in c.get('self_pairs',[]):
            if a in targets and b in targets:
                midpoint=c.get('pair_targets',{}).get((a,b),(targets[a]+targets[b])*.5)
                if a!=0:targets[a]=midpoint
                targets[b]=midpoint
        paired_selection={}
        for a,b in c.get('self_pairs',[]):
            pools=c.get('paired_candidates',{})
            if a not in pools or b not in pools:continue
            la,na=pools[a];lb,nb=pools[b]
            wa=p[:,a,None]+(rr[:,a,None]@la[None,:,:,None]).squeeze(-1)
            wb=p[:,b,None]+(rr[:,b,None]@lb[None,:,:,None]).squeeze(-1)
            scores=(wa[:,:,None]-wb[:,None]).square().sum(-1)
            scores+=.02*((la-c['paired_local'][a]).square().sum(-1)[:,None]+(lb-c['paired_local'][b]).square().sum(-1)[None])
            index=scores.flatten(1).argmin(-1);ia=index//len(lb);ib=index%len(lb)
            paired_selection[a]=(la[ia],na[ia]);paired_selection[b]=(lb[ib],nb[ib])
        for anchor in c['anchors']:
            region=anchor['region'];pool=self.pools[region];normal=self.pool_normals[region]
            target=targets[region];direction=self.tensor(anchor['direction'])
            clocal=((target-p[:,region])[:,None,:]@rr[:,region]).squeeze(1)
            owner=self.region_owners[region];psi=self.tensor(self.surface.axes[owner]*self.surface.lengths[owner]*anchor['u'])
            if region==0:psi=pool.mean(0)
            else:psi=psi+self.tensor(self.surface.normals[owner])*((pool-psi)@self.tensor(self.surface.normals[owner])).mean()
            alpha=.15 if anchor['segment'] in (1,2) else 0.
            query=(clocal+alpha*psi)/(1+alpha)
            local,index,bary=closest_surface_points(query,self.pad_triangles[region]);local_normal=self.pad_normals[region][index]
            material_selected[region]=(self.pad_face_ids[region][index],bary)
            paired=region in c.get('paired_local',{})
            if paired:
                material_selected.pop(region,None)
                local=c['paired_local'][region].expand(batch,-1);local_normal=c['paired_normal'][region].expand(batch,-1)
                if region in paired_selection:local,local_normal=paired_selection[region]
            pos=p[:,region]+(rr[:,region]@local[...,None]).squeeze(-1);nn=(rr[:,region]@local_normal[...,None]).squeeze(-1)
            rho=1+.5*(anchor['force']/self.force_max if self.force_max>0 else 0.);rho=min(max(rho,1.),1.5)
            parts['contact_position']+=(.02 if paired else 1.)*rho**2*s.wc**2*((pos-target)*1000).square().sum(-1)
            if not paired:
                estimate=anchor.get('force_estimate')
                valid=estimate.get('direction_valid',False) if estimate is not None else anchor['force']>0
                if valid and (details or s.contact_direction_model=='coulomb_force'):
                    vector=self.tensor(estimate['vector_relative']) if estimate is not None else direction*anchor['force']
                    fitted=fit_coulomb_force(vector,nn,s.friction_coefficient,s.normal_force_weight)
                    magnitude=vector.norm().square().clamp_min(1e-20)
                    force_support[region]=fitted
                    if s.contact_direction_model=='coulomb_force':
                        price=rho**2*s.wc**2*s.direction_price*s.radius_mm**2
                        parts['contact_force_vector']+=price*fitted['vector_error']/magnitude
                        parts['contact_normal_force']+=price*fitted['normal_error']/magnitude
                if s.contact_direction_model=='card_alignment':
                    parts['contact_direction']+=rho**2*s.wc**2*s.direction_price*s.radius_mm**2*(1-(nn*direction).sum(-1).clamp(-1,1))
            if region in self.previous_contacts:
                old=self.previous_contacts[region];fixed=p[:,region]+(rr[:,region]@old[...,None]).squeeze(-1)
                parts['contact_fix']+=rho**2*s.wfix**2*((pos-fixed)*1000).square().sum(-1)
            selected[region]=pos;selected_normals[region]=nn;local_selected[region]=local
        # Three semantic bones per finger; their distal endpoints are native pad apices.
        positions=p[:,1:].reshape(batch,5,3,3);rotations=rr[:,1:].reshape(batch,5,3,3,3)
        apex=p[:,[3,6,9,12,15]]+(rr[:,[3,6,9,12,15]]@self.apices[[3,6,9,12,15]][...,None]).squeeze(-1)
        chains=torch.cat((positions,apex[:,:,None,:]),2);directions=torch.nn.functional.normalize(chains[:,:,1:]-chains[:,:,:-1],dim=-1)
        pad_normals=(rr[:,[3,6,9,12,15]]@self.tensor(self.surface.normals[self.region_owners[[3,6,9,12,15]]])[...,None]).squeeze(-1)
        parts['style_tip']=s.wtip**2*((apex-c['tips'])*1000).square().sum(-1).mul(c['free']).sum(-1)
        parts['style_normal']=s.wn**2*(1-(pad_normals*c['normals']).sum(-1).clamp(-1,1)).mul(c['free']).sum(-1)
        parts['style_bone']=s.wb**2*(1-(directions*c['directions']).sum(-1).clamp(-1,1)).mul(c['free'][:,None]).sum((-1,-2))
        if c['previous'] is not None:
            pt,pr,pj=c['previous'];parts['smooth_joint']=s.wtheta**2*(theta-pj).square().sum(-1)
            parts['smooth_translation']=s.wt**2*((t-pt)*1000).square().sum(-1)
            parts['smooth_rotation']=s.womega**2*log_so3(pr.T@r).square().sum(-1)
        if c.get('motion_joint') is not None:
            parts['motion_joint']=s.wmotion_joint**2*(theta-c['motion_joint']).square().sum(-1)
        a=p+(rr@self.cap_a[...,None]).squeeze(-1);b=p+(rr@self.cap_b[...,None]).squeeze(-1)
        gap=segment_gap(a[:,self.pair_a],b[:,self.pair_a],a[:,self.pair_b],b[:,self.pair_b])-self.radii[self.pair_a]-self.radii[self.pair_b]
        intended=set(c.get('self_pairs',[]));cross_mask=theta.new_tensor([pair not in intended for pair in self.cross_pairs])
        parts['collision_self']=s.wself**2*((s.margin_mm-gap[:,:len(self.cross_pairs)]*1000).clamp_min(0).square()*cross_mask).sum(-1)
        palm_gap=gap[:,len(self.cross_pairs):];palm_mask=torch.tensor([pair not in intended for pair in self.palm_pairs],device=theta.device)
        # Capsule clearance is a preference, not proof of mesh penetration.
        # In particular a positive 0--6 mm palm gap must not reject a pose.
        parts['collision_self']+=s.wself**2*((s.margin_mm-palm_gap*1000).clamp_min(0).square()*palm_mask).sum(-1)
        hard_palm=torch.zeros(batch,dtype=torch.bool,device=theta.device);hard_pair=hard_palm.clone()
        anchor_by_region={a['region']:a for a in c['anchors']};pair_gaps={};pair_points={}
        for a,b in intended:
            if a not in selected or b not in selected:continue
            point_a,point_b=selected[a],selected[b]
            if a==0:
                # A palm has one CARD CoP, but several independently sensed
                # finger-palm contacts must not collapse to that same spot.
                target=c.get('pair_targets',{}).get((a,b),targets[b]);local_target=((target-p[:,0])[:,None,:]@rr[:,0]).squeeze(1)
                index=(self.pool_squares[0][None]-2*local_target@self.pools[0].T).argmin(-1)
                point_a=p[:,0]+(rr[:,0]@self.pools[0][index][...,None]).squeeze(-1)
            pair_points[(a,b)]=(point_a,point_b)
            axis=torch.nn.functional.normalize(self.tensor(anchor_by_region[a]['direction'])-self.tensor(anchor_by_region[b]['direction']),dim=0)
            separation=(point_b-point_a)*1000;signed_gap=(separation*axis).sum(-1);pair_gaps[(a,b)]=signed_gap
            # CARD's original targets describe hand-object contact. A supported
            # hand-hand pair additionally constrains the two selected pads to
            # the same contact, using the existing reach weight and mm units.
            rho=1+.5*max(anchor_by_region[a]['force'],anchor_by_region[b]['force'])/self.force_max if self.force_max else 1.
            parts['contact_self_closure']+=rho**2*s.wpinch**2*separation.square().sum(-1)
            parts['contact_direction']+=rho**2*s.wc**2*s.direction_price*s.radius_mm**2*(1+(selected_normals[a]*selected_normals[b]).sum(-1).clamp(-1,1))
            tangent=(separation-signed_gap[:,None]*axis).norm(dim=-1)
            # Signed CoP separation is not a solid-surface intersection test.
            # The full native mesh query below owns self-penetration rejection.
        hard=hard_palm|hard_pair|hard_mesh
        max_pen=zero.clone()
        if c['sdf'].metadata()['ready']:
            # Link blocks bound memory, while every configured sample is evaluated.
            weight=s.wobj_mesh if c['sdf'].kind=='tracked_mesh' else s.wobj_pseudo
            for region in range(16):
                ids=self.collision_regions==region;local=self.collision_local[ids]
                samples=p[:,region,None]+(rr[:,region,None]@local[None,:,:,None]).squeeze(-1)
                penetration=c['sdf'].penetration(samples,c['joints'])*1000
                max_pen=torch.maximum(max_pen,penetration.amax(-1))
                parts['collision_object']+=weight**2*(penetration-s.allowance_mm).clamp_min(0).square().sum(-1)
            # A hull inferred from unmatched pressure is uncertain geometry,
            # not a measured solid. It can guide search but cannot veto it.
            if c['sdf'].kind=='tracked_mesh':hard|=max_pen>s.hard_penetration_mm
        groups=dict(contact=parts['contact_position']+parts['contact_direction']+parts['contact_force_vector']+parts['contact_normal_force']+parts['contact_fix']+parts['contact_self_closure'],style=parts['style_tip']+parts['style_normal']+parts['style_bone'],smooth=parts['smooth_joint']+parts['smooth_translation']+parts['smooth_rotation']+parts['motion_joint'],collision=parts['collision_self']+parts['collision_mesh']+parts['collision_object'])
        raw_cost=sum(groups.values());cost=torch.where(hard|~torch.isfinite(raw_cost),torch.inf,raw_cost)
        if search:
            violation=(mesh_depth-s.self_penetration_mm).clamp_min(0).square().sum((-1,-2))
            if c['sdf'].kind=='tracked_mesh':violation+=(max_pen-s.hard_penetration_mm).clamp_min(0).square()
            violation=torch.where(mesh_valid&torch.isfinite(raw_cost),violation,torch.inf)
            return raw_cost,violation
        if not details:return cost
        return cost,dict(parts=parts,groups=groups,points=selected,normals=selected_normals,local_points=local_selected,material_points=material_selected,force_support=force_support,hard=hard,hard_palm=hard_palm,hard_pair=hard_pair,hard_mesh=hard_mesh,mesh_depth=mesh_depth,mesh_valid=mesh_valid,hard_object=(max_pen>s.hard_penetration_mm)&(c['sdf'].kind=='tracked_mesh'),pair_gaps=pair_gaps,pair_points=pair_points,max_penetration_mm=torch.maximum(max_pen,mesh_max),object_max_penetration_mm=max_pen,min_palm_gap_mm=palm_gap.amin(-1)*1000)

    def verify_mesh(self,t,r,theta):
        p,rr=self.frames(t,r,theta);depth,valid=self.mesh_collision.query(p,rr,dense=True)
        return valid&(depth.amax((-1,-2))<=self.settings.self_penetration_mm)

    def contact_report(self,detail):
        result=[]
        for pair,gap in detail['pair_gaps'].items():
            a,b=pair;distance=float((detail['pair_points'][pair][1][0]-detail['pair_points'][pair][0][0]).norm()*1000)
            dot=float((detail['normals'][a][0]*detail['normals'][b][0]).sum())
            result.append(dict(regions=list(pair),signed_gap_mm=float(gap[0]),distance_mm=distance,
                normal_dot=dot,satisfied=bool(distance<=3 and dot<-.1),intentional=True,
                evidence=self.contact_association.get('reason') if 3 in pair else 'upstream_bilateral_material_contact'))
        return result

    def force_report(self,detail,region):
        fit=detail['force_support'].get(region)
        if fit is None:return dict(available=False,reason='no_valid_pressure_resultant')
        return dict(available=True,units='relative_response_sum',force_calibrated=False,
            shear_measured=False,friction_coefficient=self.settings.friction_coefficient,
            normal_target_assumption='normal_push_with_resultant_magnitude',
            desired_normal_relative=float(fit['desired_normal'][0]),desired_tangent_relative=float(fit['desired_tangent'][0]),
            desired_force_in_cone=bool(fit['desired_in_cone'][0]),
            fitted_force_relative=fit['force'][0].cpu().tolist(),fitted_normal_relative=float(fit['normal_force'][0]),
            fitted_tangent_relative=float(fit['tangential_force'][0]),
            vector_error_squared=float(fit['vector_error'][0]),normal_error_squared=float(fit['normal_error'][0]))

    @torch.no_grad()
    def step(self,frame,contact_enabled=True):
        began=time.perf_counter();stamp=float(frame['timestamp_ms'])
        def clock():
            # Complete queued CUDA work before attributing wall time to a stage.
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            return time.perf_counter()
        calibration=(frame.get('skeleton_runtime_revision'),frame.get('pose_calibration_version'))
        if self.source_calibration is not None and calibration!=self.source_calibration:
            # A new IK or MANO zero changes the meaning of accumulated contact
            # positions. Rebuild causal object/history in the new pose frame.
            self.reset()
        self.source_calibration=calibration
        if not math.isfinite(stamp) or (self.last_seen_timestamp is not None and stamp<=self.last_seen_timestamp):raise ValueError('non-increasing or nonfinite source timestamp')
        if self.last_seen_timestamp is not None and stamp-self.last_seen_timestamp>1000:
            self.initializer.reset();self.previous=None;self.previous_initial_joint=None;self.previous_source_wrist=None;self.previous_contacts={}
        self.last_seen_timestamp=stamp
        (vertices,joints,points,normals),weights,anchors=self.extractor.extract(frame,contact_enabled)
        if self.offline_anchors is not None:anchors=self.offline_anchors[stamp]
        extracted_at=clock()
        raw_initial=self.initializer(joints)
        initialized_at=clock()
        paired_regions=self.update_self_contacts(frame,anchors,raw_initial)
        initial=raw_initial;seed_report=dict(applied=False,uses_history=False)
        pair=self.contact_association.get('selected')
        if pair is not None:
            initial,seed_report=pinch_seed(self.seed_fk,raw_initial,self.region_body.cpu().tolist(),self.paired_local,self.paired_normal,pair,self.adapter.low[6:],self.adapter.high[6:])
        current_regions={a['region'] for a in anchors};self.previous_contacts={k:v for k,v in self.previous_contacts.items() if k in current_regions}
        if self.previous_timestamp is not None and stamp-self.previous_timestamp>250:self.previous_contacts={}
        self.force_max=self.offline_force_max if self.offline_force_max is not None else max(self.force_max,max([a['force'] for a in anchors],default=0.))
        self.update_object_transform(frame)
        if isinstance(self.sdf,PseudoObject) and not self.sdf.frozen:
            regime='self_contact' if paired_regions else 'external_or_unpaired'
            external=[a for a in anchors if a['region'] not in paired_regions]
            if self.contact_regime is not None and self.contact_regime!=regime:
                self.sdf=PseudoObject(self.device);self.previous_contacts={}
            if not external and self.last_external_stamp is not None and stamp-self.last_external_stamp>500:
                self.sdf=PseudoObject(self.device);self.last_external_stamp=None
            self.contact_regime=regime
            self.sdf.add(external,joints,stamp)
            if any(a['region'] for a in external):self.last_external_stamp=stamp
        c=self.context(vertices,joints,anchors,initial)
        incumbents=[raw_initial] if seed_report['applied'] else []
        if self.previous is None:
            # Evaluate neutral only as an explicit incumbent inside the same
            # Robokit refinement bounds and with unchanged collision gates.
            neutral=self.adapter.neutral[6:].clone();qneutral=torch.cat((neutral.new_zeros(6),neutral))
            tips=self.adapter.keypoints(qneutral)[[4,8,12,16,20]]
            translation=self.tensor(joints[[4,8,12,16,20]]).mean(0)-initial[1]@tips.mean(0)
            incumbents.append((translation,initial[1],neutral))
        cem_started_at=clock()
        use_robokit=self.settings.cem_backend=='robokit' and not self.settings.experimental_self_contact and self.settings.contact_direction_model=='card_alignment'
        cold_start=False
        if use_robokit:
            from .robokit_card import RobokitCardSolver
            if self.robokit_cem is None or self.robokit_cem.settings!=self.settings:
                self.robokit_cem=RobokitCardSolver(self);cold_start=True
            best,diagnostic=self.robokit_cem.solve(initial,c)
        else:
            best,diagnostic=cem(initial,None,self.adapter.low[6:],self.adapter.high[6:],lambda t,r,j:self.evaluate(t,r,j,c,search=True),self.settings,self.generator,incumbents=incumbents,verify=self.verify_mesh)
            diagnostic['backend']='torch_reference'
            if self.settings.cem_backend=='robokit':diagnostic['backend_reason']='explicit_force_cone_or_pair_closure_ablation_uses_reference_backend'
        diagnostic['cold_start']=cold_start
        cem_finished_at=clock()
        diagnostic['initial_incumbent_source']='current_frame_robokit' if use_robokit else 'current_frame_robokit_and_native_neutral' if seed_report['applied'] and self.previous is None else 'current_frame_robokit' if seed_report['applied'] else 'native_neutral' if incumbents else None
        diagnostic['initializer']=dict(self.initializer.last_report)
        diagnostic['style_finger_weights']=c['free'].cpu().tolist()
        diagnostic['style_policy']='all_fingers' if self.settings.contact_style_weight==1 else 'contact_finger_style_scaled'
        diagnostic['joint_motion_reference']='previous_solution_plus_current_Robokit_joint_motion' if c['motion_joint'] is not None else None
        diagnostic['joint_motion_reference_rad']=c['motion_joint'].cpu().tolist() if c['motion_joint'] is not None else None
        diagnostic['joint_motion_weight']=self.settings.wmotion_joint
        diagnostic['smooth_reference']='previous_feasible_solution' if c['previous'] is not None else None
        diagnostic['smooth_joint_units']='rad'
        diagnostic['tactile_seed']=seed_report;diagnostic['contact_association']=self.contact_association
        diagnostic['contact_objective_policy']='experimental_pair_closure' if self.settings.experimental_self_contact else 'CARD_source_CoP_direction_force_and_semantic_pad_search'
        if seed_report['applied']:diagnostic['search_center']='current_frame_robokit_plus_tactile'
        t,r,theta=best
        if use_robokit:detail=self.robokit_cem.details(best,c)
        else:_,detail=self.evaluate(t[None],r[None],theta[None],c,True)
        audited_at=clock()
        if diagnostic['feasible'] and bool(detail['hard'][0]):
            diagnostic.update(feasible=False,final_audit_failed=True)
            diagnostic['output_selection']='current_candidate_failed_final_audit'
        diagnostic['output_self_mesh_satisfied']=bool(detail['mesh_valid'][0] and not detail['hard_mesh'][0])
        if diagnostic['feasible']:
            self.previous=tuple(v.clone() for v in (t,r,theta));self.previous_timestamp=stamp
            self.previous_initial_joint=initial[2].clone()
            self.previous_source_wrist=source_wrist(joints)
        if diagnostic['feasible']:self.previous_contacts={k:v[0].clone() for k,v in detail['local_points'].items()}
        else:self.previous_contacts={}
        p,rr=self.frames(t,r,theta);p=p.cpu().numpy()[self.surface.body_ids];rr=rr.cpu().numpy()[self.surface.body_ids]
        old_euler=np.asarray(self.last_output['qpos'][3:6]) if self.last_output else np.zeros(3)
        euler=continuous_euler_xyz(r.cpu().numpy(),old_euler)
        qn=np.r_[t.cpu().numpy(),euler,theta.cpu().numpy()]
        taxels=p[self.owners]+np.einsum('nij,nj->ni',rr[self.owners],self.local);taxel_normals=np.einsum('nij,nj->ni',rr[self.owners],self.normal)
        anchors_out=[];errors=[];angular=[]
        for a in anchors:
            region=a['region'];pos=detail['points'][region][0].cpu().numpy();nn=detail['normals'][region][0].cpu().numpy()
            error=float(np.linalg.norm(pos-a['position_m'])*1000);angle=float(np.degrees(np.arccos(np.clip(nn@np.array(a['direction']),-1,1))))
            anchors_out.append({**a,'relative_strength':a['force'],'force_units':'relative_response_sum','rho':1+.5*a['force']/self.force_max if self.force_max else 1.,'robot_position_m':pos.tolist(),'robot_normal':nn.tolist(),'error_mm':error,'normal_error_deg':angle})
            material=detail['material_points'].get(region)
            anchors_out[-1].update(robot_owner=int(self.region_owners[region]),robot_link=self.model.body(int(self.region_body[region])).name,
                                  robot_face_id=int(material[0][0]) if material is not None else None,
                                  robot_barycentric=material[1][0].cpu().tolist() if material is not None else None,
                                  force_support=self.force_report(detail,region))
            projected=[self.mapping['records'][sid] for sid in a['sids'] if self.mapping['records'][sid] is not None]
            if projected:
                response=np.array([weights['response'][x['sid']] for x in projected]);response=response if response.sum()>1e-12 else np.ones(len(projected))
                anchors_out[-1]['projected_material_local_m']=np.average([x['local_position_m'] for x in projected],axis=0,weights=response).tolist()
                anchors_out[-1]['projected_material_normal']=unit(np.average([x['local_normal'] for x in projected],axis=0,weights=response)).tolist()
            errors.append(error);angular.append(angle)
        parts={k:float(v[0]) for k,v in detail['parts'].items()};groups={k:float(v[0]) for k,v in detail['groups'].items()}
        self_contacts=self.contact_report(detail)
        registration=frame.get('contact_registration') or self.source_topology.get('contact_registration') or {}
        verified=set(registration.get('verified_sids',[]));active_sids={sid for a in anchors for sid in a['sids']}
        source_geometry=dict(registration=registration,active_verified_sids=sorted(active_sids&verified),
            active_unverified_sids=sorted(active_sids-verified),personalized_mano=bool(frame.get('calibrated')),
            sensor_mounting_status=(frame.get('sensor_mounting') or {}).get('status','unregistered'))
        source_geometry['contact_geometry_ready']=bool(active_sids and not active_sids-verified and source_geometry['personalized_mano'] and source_geometry['sensor_mounting_status']=='multi_pose_calibrated')
        diagnostic.update(used_history_fallback=False,collision_scope='candidate_endpoint_only_no_swept_path_check')
        output=dict(status='ready' if diagnostic['feasible'] else 'infeasible',ready=diagnostic['feasible'],solution_feasible=diagnostic['feasible'],display_available=True,
            timestamp_ms=stamp,source_timestamp_ms=stamp,tactile_timestamp_ms=frame.get('tactile_timestamp_ms'),hand_side='right',source_hand_side=self.source_topology['hand_side'],coordinate_frame='shared_virtual_world',units='m',
            initializer_quality={k:v for k,v in self.initializer.artifact.items() if k not in ('matrix','bias')},solver_session_id=self.session_id,topology_version=self.topology_version,algorithm=ALGORITHM,relative_force_max=self.force_max,force_units='relative_response_sum',settings=self.settings.payload(),solver=diagnostic,contact_anchors=anchors_out,self_contact_pairs=self_contacts,contact_constraints_satisfied=all(p['satisfied'] for p in self_contacts) if self_contacts else None,sdf=self.sdf.metadata(),postprocessing='raw_cem',seed=self.settings.seed,
            source_contact_geometry=source_geometry,
            taxel_projection=dict(policy=self.mapping['policy'],mapped_sids=sum(x is not None for x in self.mapping['records']),position_and_normal=True,dynamic_contact_search='CARD_Eq2_continuous_native_triangles',physical_registration_verified=False,mapping_role='virtual_taxel_display_only_in_default_objective'),
            mapping_version=self.mapping['version'],mapping_ready=self.mapping['ready'],failed_sids=self.mapping['failed_sids'],geometry_version=frame.get('geometry_version'),source_binding_version=frame.get('binding_version'),
            initial_qpos=np.r_[initial[0].cpu().numpy(),Rotation.from_matrix(initial[1].cpu().numpy()).as_euler('XYZ'),initial[2].cpu().numpy()].tolist(),
            robokit_qpos=np.r_[raw_initial[0].cpu().numpy(),Rotation.from_matrix(raw_initial[1].cpu().numpy()).as_euler('XYZ'),raw_initial[2].cpu().numpy()].tolist(),
            joint_names=[self.model.joint(i).name for i in range(6,26)],joint_positions_rad=qn[6:].tolist(),qpos=qn.tolist(),raw_qpos=qn.tolist(),wrist_position_m=t.cpu().tolist(),wrist_rotation=r.cpu().tolist(),
            link_positions_m=p.tolist(),link_rotations=rr.tolist(),taxel_positions_m=taxels.tolist(),taxel_normals=taxel_normals.tolist(),source_taxel_positions_m=points.tolist(),source_taxel_normals=normals.tolist(),source_joints_m=joints.tolist(),source_vertices_m=vertices.tolist(),
            contact_ready=bool(weights['ready'] and diagnostic['feasible']),contact_mask=weights['mask'].tolist() if weights['ready'] and diagnostic['feasible'] else [None]*460,contact_weights=weights['cloud'].tolist(),
            collision=dict(source='native_mesh_hard_guard_plus_CARD_capsules_and_object_SDF',dynamics=False,max_penetration_mm=float(detail['max_penetration_mm'][0]),
                self_mesh=self.mesh_collision.report(detail['mesh_depth'][0],detail['mesh_valid'][0]),self_hard_limit_mm=self.settings.self_penetration_mm,
                object_max_penetration_mm=float(detail['object_max_penetration_mm'][0]),mesh_hard_rejected=bool(detail['hard_mesh'][0]),
                object_hard_gate_enabled=self.sdf.kind=='tracked_mesh',
                min_palm_gap_mm=float(detail['min_palm_gap_mm'][0]),palm_hard_rejected=bool(detail['hard_palm'][0]),pair_hard_rejected=bool(detail['hard_pair'][0]),object_hard_rejected=bool(detail['hard_object'][0])),
            quality=dict(active_taxels=int(weights['mask'].sum()),active_patches=int(weights['active'].sum()),active_anchors=len(anchors),contact_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                contact_max_error_mm=max(errors) if errors else None,patch_center_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,contact_normal_mean_deg=float(np.mean(angular)) if angular else None,
                cost_terms=groups,cost_subterms=parts,source_quality=frame.get('quality'),processing_ms=(time.perf_counter()-began)*1000,joint_bounds_saturated=int(((qn[6:]-self.lower_np[6:]<.01)|(self.upper_np[6:]-qn[6:]<.01)).sum())))
        finished_at=clock()
        diagnostic['timing_ms']=dict(extract=(extracted_at-began)*1000,
            robokit=(initialized_at-extracted_at)*1000,prepare=(cem_started_at-initialized_at)*1000,
            cem=(cem_finished_at-cem_started_at)*1000,final_audit=(audited_at-cem_finished_at)*1000,
            report=(finished_at-audited_at)*1000,total=(finished_at-began)*1000,
            scope='worker_step_wall_clock_cuda_synchronized_cem_includes_candidate_collision_verification')
        output['quality']['processing_ms']=diagnostic['timing_ms']['total']
        self.last_output=output
        return output
