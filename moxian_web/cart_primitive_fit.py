"""Hand-sized Dexonomy primitive with bounded, collision-checked contact facets."""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .cart_primitives import REVISION, _proposals, primitive_target, surface
from .cart_protocol import stage

POLICY=dict(confidence_threshold=.55,gap_m=.002,normal_deg=30.,
            delta_radius_fraction=.5,delta_max_m=.012,delta_min_m=.002,
            contact_scope='every_key_finger_one_annotated_region',confidence_uses_template_error=False,
            budget_scope='candidate_admission')


def finger_patches(patches):
    groups={}
    for name,values in patches.items():groups.setdefault(name.split('_')[0],[]).append(values)
    return {name:tuple(np.concatenate([v[i] for v in values]) for i in range(3))
            for name,values in groups.items()}


def key_contacts(hull,patches,spec):
    from .cart_convex import hull_contacts
    candidates=hull_contacts(hull,patches,{**spec,'patches':list(patches),
        'patch_weights':{p:spec['patch_weights'].get(p,1.) for p in patches}})
    groups={}
    for c in candidates:groups.setdefault(c['patch'].split('_')[0],[]).append(c)
    # A passing contact always beats an otherwise cheaper nonpassing one.
    def score(c):
        return (not(c['gap_m']<=POLICY['gap_m'] and c['normal_error_deg']<=POLICY['normal_deg']),
                (c['gap_m']/POLICY['gap_m'])**2+(c['normal_error_deg']/POLICY['normal_deg'])**2)
    return [min(group,key=score) for group in groups.values()]


def fit_primitive(kind,patches,center,rotation,hand_error_m=0.,obstacles=None,max_nfev=30):
    """Fit metric dimensions to opposing material surfaces, never copy grid sizes.

    Confidence is a geometric quality score, not a calibrated probability.
    Cylinder length uses the observed axial span when no end cap is observed.
    """
    patches=finger_patches(patches)
    cloud=np.concatenate([v[0] for v in patches.values()])
    center=np.asarray(center,float);rotation=np.asarray(rotation,float)
    span=max(float(np.linalg.norm(np.ptp(cloud,axis=0))),.006)
    center=np.clip(center,cloud.min(0)-span*.25,cloud.max(0)+span*.25)
    local=(cloud-center)@rotation
    radius=float(np.clip(np.median(np.linalg.norm(local if kind=='sphere' else local[:,:2],axis=1)),.002,.12))
    height=float(np.clip(np.ptp(local[:,2])+2*radius,.006,.3))
    if kind=='sphere':x=np.r_[center,np.log(radius)]
    else:x=np.r_[center,Rotation.from_matrix(rotation).as_rotvec(),np.log([radius,height])]
    initial=x.copy()
    def parameters(value):
        if kind=='sphere':return value[:3],np.eye(3),np.exp(value[3]),2*np.exp(value[3])
        return value[:3],Rotation.from_rotvec(value[3:6]).as_matrix(),*np.exp(value[6:8])
    lo=np.r_[cloud.min(0)-span*.5, np.log(.001)] if kind=='sphere' else np.r_[cloud.min(0)-span*.5,[-2*np.pi]*3,np.log([.001,.004])]
    hi=np.r_[cloud.max(0)+span*.5, np.log(.15)] if kind=='sphere' else np.r_[cloud.max(0)+span*.5,[2*np.pi]*3,np.log([.15,.35])]
    for _ in range(2):
        c,r,rad,h=parameters(x);points=[];normals=[]
        for p,n,_ in patches.values():
            d,_,out=surface(p,c,r,rad,h,kind)
            score=(d/.005)**2+2*(1+(n*out).sum(1))
            k=int(score.argmin());points.append(p[k]);normals.append(n[k])
        points=np.asarray(points);normals=np.asarray(normals)
        def residual(value):
            c,r,rad,h=parameters(value)
            d,_,out=surface(points,c,r,rad,h,kind)
            regularizer=(value-initial)*.03
            regularizer[:3]/=span
            if kind=='cylinder':regularizer[-1]*=6
            penetration=(np.minimum(surface(obstacles,c,r,rad,h,kind)[0],0)/.002
                         if obstacles is not None else [])
            return np.r_[d/.004,((normals+out)*.6).ravel(),penetration,regularizer]
        fitted=least_squares(residual,np.clip(x,lo+1e-9,hi-1e-9),bounds=(lo,hi),max_nfev=max_nfev)
        x=fitted.x
    c,r,rad,h=parameters(x)
    d,_,out=surface(points,c,r,rad,h,kind)
    normal_error=np.degrees(np.arccos(np.clip(-(normals*out).sum(1),-1,1)))
    rms=float(np.sqrt(np.mean(d*d)));normal_rms=float(np.sqrt(np.mean(normal_error**2)))
    s=np.linalg.svd(normals,compute_uv=False)
    opposing=bool(np.min(normals@normals.T)<-.25)
    # Parallel opposing jaw normals are normal for a prismatic grasp. They
    # constrain a cylinder's diameter or cap separation, despite rank-one n.
    observable=(len(s)>=2 and s[1]/max(s[0],1e-12)>.07) if kind=='sphere' else opposing
    confidence=float(np.exp(-.5*((rms/.005)**2+(normal_rms/45)**2)))
    if not observable or not opposing:confidence=0.
    axial=normals@r[:,2]
    length_observed=kind=='sphere' or bool((axial>.7).any() and (axial<-.7).any())
    if not length_observed:confidence*=.8
    return dict(kind=kind,radius_m=float(rad),height_m=float(h),center_m=c,rotation=r,
        confidence=confidence,confidence_kind='geometric_quality_not_probability',
        gap_rms_m=rms,normal_rms_deg=normal_rms,hand_alignment_rms_m=float(hand_error_m),
        dimensions_observable=bool(observable),opposing_contacts=bool(opposing),
        length_observed_from_contacts=length_observed,
        height_estimator='observed_axial_span_with_radial_margin' if kind=='cylinder' else 'diameter')


def estimate_primitive(vertices,faces,regions,stage_id,full=False,budget_ms=120,witnesses=None,joints=None,previous=None):
    from .cart_convex import _patches
    from .cart_geometry import FreeSpace,hull_data,region_pair
    started=time.perf_counter();deadline=float('inf') if full else started+max(0,budget_ms)/1000
    spec=stage(stage_id)
    if witnesses is None:witnesses=[region_pair(vertices,faces,regions,p) for p in spec['pairs']]
    base=dict(algorithm=REVISION,stage=stage_id,category=spec['category'],category_name=spec['category_name'],
              pairs=witnesses,coordinate_frame='palm_canonical',primitive_types=spec['primitive_types'])
    if spec['category']==1:
        return dict(base,ready=False,status='hand_contact',contact_mode='hand',reference_gap_m=max(w['gap_m'] for w in witnesses))
    # Validate the action surfaces even if no fitting budget remains.
    action_patches=_patches(vertices,faces,regions,spec['envelope'],True)
    cloud=np.concatenate([p for p,_,_ in action_patches.values()])
    attempts=[];best=None;valid=None;tried=[];field=None
    proposals=_proposals(spec,joints,cloud);seen=set();unique=[]
    for proposal in proposals:
        if proposal[1]['type_id'] not in seen:unique.append(proposal);seen.add(proposal[1]['type_id'])
    offset=0 if full or not unique else int((previous or {}).get('next_proposal_offset',0))%len(unique)
    proposals=unique[offset:]+unique[:offset];attempted=0
    v=np.asarray(vertices);tri=v[np.asarray(faces)]
    # Preview limits the number of hypotheses, not the validity of each one.
    # The old sparse preview could reject a pose accepted by preprocessing.
    count=768
    obstacles=np.vstack((v[np.linspace(0,len(v)-1,min(count,len(v))).astype(int)],
                         tri[np.linspace(0,len(tri)-1,min(count,len(tri))).astype(int)].mean(1)))
    for error,ref,center,rotation in proposals:
        if time.perf_counter()>=deadline:break
        attempted+=1
        allowed=ref['required_patches'];fingers={p.split('_')[0] for p in allowed}
        required=[p for p in allowed if len(regions.get(p,[]))]
        if {p.split('_')[0] for p in required}!=fingers:
            attempts.append(dict(type_id=ref['type_id'],status='missing_key_finger_surface'));continue
        patches=_patches(vertices,faces,regions,required,True)
        warm=(previous or {}).get('primitive',{}) if (previous or {}).get('ready') else {}
        if not full and warm.get('reference_type')==ref['type_id']:
            center=np.asarray(warm['center_m']);rotation=np.asarray(warm['rotation'])
        fit=fit_primitive(ref['kind'],patches,center,rotation,np.sqrt(3*error) if joints is not None else 0.,obstacles)
        attempt=dict(type_id=ref['type_id'],confidence=fit['confidence'],gap_rms_m=fit['gap_rms_m'],
                     normal_rms_deg=fit['normal_rms_deg'],hand_alignment_rms_m=fit['hand_alignment_rms_m'],
                     status='low_confidence')
        attempts.append(attempt);tried.append(ref['kind'])
        if fit['confidence']<POLICY['confidence_threshold']:continue
        # Once admitted, a candidate always reaches full validation. Budget
        # exhaustion cannot erase a good fit before it has even been checked.
        if field is None:field=FreeSpace(vertices,faces)
        seed=fit['center_m'];radius=fit['radius_m'];height=fit['height_m'];rotation=fit['rotation']
        limit=max(POLICY['delta_min_m'],min(POLICY['delta_max_m'],POLICY['delta_radius_fraction']*radius))
        current=primitive_target(ref['kind'],radius,height,seed,rotation,ref)
        clear=field.clear(current,seed)
        # Small inward corrections leave room for local contact facets. They
        # cannot turn a poor primitive fit into an accepted estimate.
        for _ in range(5):
            if clear:break
            radius*=.96;height*=.96
            current=primitive_target(ref['kind'],radius,height,seed,rotation,ref)
            clear=field.clear(current,seed)
        if not clear:attempt['status']='primitive_collision';continue
        primitive=dict(current['primitive']);initial_volume=current['volume_m3'];steps=0
        # Add material tangent facets; preserve the fitted primitive as a core.
        # Every accepted addition is checked against complete hand triangles.
        existing=key_contacts(current,patches,spec)
        for finger,(p,n,tangent) in finger_patches(patches).items():
            contact=next(c for c in existing if c['patch'].split('_')[0]==finger)
            if contact['gap_m']<=POLICY['gap_m'] and contact['normal_error_deg']<=POLICY['normal_deg']:continue
            distances,_,out=surface(p,seed,rotation,radius,height,ref['kind'])
            score=(distances/.004)**2+2*(1+(n*out).sum(1))
            for k in np.argsort(score)[:6]:
                added=False
                for margin in (0.,.00025,.00075,.0015):
                    facet=p[k]+tangent[k]+margin*n[k]
                    displacement=surface(facet,seed,rotation,radius,height,ref['kind'])[0]
                    if np.max(displacement)>limit:continue
                    trial=hull_data(np.vstack((current['vertices_m'],facet)))
                    if field.clear(trial,seed):current=trial;steps+=1;added=True;break
                if added:break
        contacts=key_contacts(current,patches,spec)
        selected_patches=[c['patch'] for c in contacts]
        covered=[c['gap_m']<=POLICY['gap_m'] and c['normal_error_deg']<=POLICY['normal_deg'] for c in contacts]
        # Validate the final complete object as well as all key finger segments.
        clear=field.clear(current,seed)
        points=np.asarray(current['vertices_m'])
        offsets,hits,_=surface(points,seed,rotation,radius,height,ref['kind'])
        changed=offsets>1e-7
        delta=max(0.,float(np.max(offsets)))
        fit_metadata={k:v for k,v in fit.items() if k not in ('center_m','rotation')}
        primitive.update(fit_metadata,radius_m=radius,height_m=height,
                         fitted_radius_m=fit['radius_m'],fitted_height_m=fit['height_m'])
        current.update(base,primitive=primitive,initialization=dict(kind=ref['kind'],reference_type=ref['type_id'],
            reference_row=ref['row'],center_m=seed.tolist(),rotation=rotation.tolist()),
            required_patches=selected_patches,required_fingers=sorted(fingers),allowed_contact_patches=allowed,
            patch_weights={c['patch']:c['weight'] for c in contacts},contacts=contacts,contact_mode='hull',
            status='primitive_delta',confidence=fit['confidence'],confidence_threshold=POLICY['confidence_threshold'],
            coverage=float(np.mean(covered)),collision_free=bool(clear),
            ready=bool(clear and all(covered) and delta<=limit+1e-9),
            delta_geometry=dict(kind='bounded_convex_contact_facets',limit_m=limit,max_offset_m=delta,
                                added_facets=steps,core_preserved=True,vertices_m=points[changed].tolist(),
                                offsets_m=(points-hits)[changed].tolist()),
            optimization_steps=steps,initial_volume_m3=initial_volume,free_convex_shape=False,
            reference_gap_m=max(c['gap_m'] for c in contacts),supervision_valid=False)
        current['supervision_valid']=current['ready']
        attempt['status']='accepted' if current['ready'] else 'key_contacts_incomplete'
        # Once every contact passes, numerical zero-gap differences must not
        # outweigh a better supported primitive family and size estimate.
        rank=(-current['coverage'],error if joints is not None else 0.,-fit['confidence'],delta,
              max(c['gap_m'] for c in contacts))
        if best is None or rank<best[0]:best=(rank,current)
        if current['ready'] and (valid is None or rank<valid[0]):valid=(rank,current)
    expired=time.perf_counter()>=deadline
    if valid:result=valid[1]
    else:
        reason=('budget_exhausted' if expired else 'low_confidence' if attempts and all(a['status']=='low_confidence' for a in attempts)
                else 'no_graspable_primitive')
        result=dict(base,ready=False,status=reason,contact_mode='unavailable',supervision_valid=False,
                    reference_gap_m=best[1]['reference_gap_m'] if best else float(np.linalg.norm(np.ptp(cloud,axis=0))),
                    diagnostic_candidate={**best[1],'ready':True,'supervision_valid':False} if best else None)
    next_offset=(offset+attempted)%max(1,len(unique))
    if valid:
        next_offset=next(i for i,p in enumerate(unique) if p[1]['type_id']==result['primitive']['reference_type'])
    result.update(fit_attempts=attempts,initial_shapes_tried=list(dict.fromkeys(tried)),
                  processing_ms=(time.perf_counter()-started)*1000,budget_exhausted=expired,
                  search_profile='full' if full else 'realtime',approximate=True,
                  next_proposal_offset=next_offset)
    if witnesses:result['closure_gap_m']=max(w['gap_m'] for w in witnesses)
    return result
