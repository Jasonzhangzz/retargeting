"""Source material contact evidence. No grasp classes or inferred object hull."""
from itertools import combinations
import numpy as np
import torch
import warp as wp
from .tactile_surface import CurvedTaxels
from .hand_anatomy import closed_surface_parts
from .cart_protocol import FINGERS
from .cart_contact_witness import contact_witness, DEEP_PENETRATION_M

CONTACT_POLICY = 'directional_depenetration_then_contact_v2'


def link_owner(name):
    finger, part = name.split('_', 1)
    f = FINGERS.index(finger)
    k = {'metacarpal': 0, 'proximal': 1, 'distal': 2}[part] if f == 0 else {
        'proximal': 0, 'intermediate': 1, 'distal': 2}[part]
    return 2 + 4 * f + k


def taxel_evidence(frame, patches):
    """Unknown is distinct from a valid sensor frame with no active taxels."""
    contact = frame.get('source_contact') or {}
    stamp, tt = frame.get('timestamp_ms'), frame.get('tactile_timestamp_ms')
    detector = contact.get('timestamp_ms', tt)
    clocks = [stamp, tt, detector]
    valid = bool(contact.get('ready') is True and
                 all(isinstance(t, (int, float)) and np.isfinite(t) for t in clocks) and
                 abs(stamp - tt) <= 25 and abs(stamp - detector) <= 25)
    cells = contact.get('cell_contact') or []
    values = np.asarray(contact.get('normalized') or [], float)
    valid = valid and len(cells) == 460 and values.shape == (460,) and np.isfinite(values).all()
    mask = np.array([c is True for c in cells], bool) if valid else np.zeros(460, bool)
    mask &= (np.asarray(patches) >= 0) & (np.asarray(patches) < 14)
    return valid, mask, np.clip(values, 0, 1) if valid else np.zeros(460)


def patch_cloud(points, normals, radius=.003):
    """Deterministic small tangent discs; visualization only."""
    cloud = []
    for point, normal in zip(points, normals):
        p, n = np.asarray(point), np.asarray(normal)
        n = n / max(np.linalg.norm(n), 1e-12)
        u = np.cross(n, np.eye(3)[np.argmin(abs(n))]); u /= max(np.linalg.norm(u), 1e-12)
        v = np.cross(n, u)
        for r, count in ((.4 * radius, 8), (radius, 16)):
            for a in np.linspace(0, 2*np.pi, count, endpoint=False):
                cloud.append((p + r*(np.cos(a)*u + np.sin(a)*v)).tolist())
    return cloud


@wp.kernel(enable_backward=False)
def _source_depth(vertices: wp.array(dtype=wp.vec3), source: wp.array(dtype=wp.vec3i),
                  bary: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int),
                  skins: wp.array(dtype=wp.uint64), solids: wp.array(dtype=wp.uint64),
                  allowed: wp.array2d(dtype=int), result: wp.array2d(dtype=float),
                  hit_faces: wp.array2d(dtype=int), hit_bary: wp.array2d(dtype=wp.vec2),
                  contact_faces: wp.array2d(dtype=int), contact_bary: wp.array2d(dtype=wp.vec2),
                  contact_valid: wp.array2d(dtype=int), retracted: wp.array2d(dtype=int), threshold: float):
    s, t = wp.tid(); a = owner[s]
    if allowed[a,t] == 0: return
    tri = source[s]; bc = bary[s]
    p = bc[0]*vertices[tri[0]] + bc[1]*vertices[tri[1]] + bc[2]*vertices[tri[2]]
    skin = wp.mesh_query_point_no_sign(skins[t], p, .05)
    if not skin.result: return
    for i in range(3):
        for j in range(3):
            if tri[i] == wp.mesh_get_index(skins[t], 3*skin.face+j): return
    inside = wp.mesh_query_point_sign_winding_number(solids[t], p, .05)
    if inside.result and inside.sign < 0.:
        hit = wp.mesh_eval_position(skins[t], skin.face, skin.u, skin.v)
        cap = wp.mesh_eval_position(solids[t], inside.face, inside.u, inside.v)
        distance = wp.length(p-hit)
        if distance <= wp.length(p-cap)+.0005:
            result[s,t] = distance
            hit_faces[s,t] = skin.face
            hit_bary[s,t] = wp.vec2(skin.u,skin.v)
            x = vertices[tri[0]]; y = vertices[tri[1]]; z = vertices[tri[2]]
            # Push the penetrating material back towards its own interior,
            # opposite its outward normal; never follow the wrong nearest face.
            normal = wp.normalize(wp.cross(y-x,z-x))
            witness = contact_witness(skins[t],solids[t],p,-normal,threshold)
            contact_valid[s,t] = witness.valid
            retracted[s,t] = witness.retracted
            if witness.valid == 1:
                contact_faces[s,t] = witness.face
                contact_bary[s,t] = wp.vec2(witness.u,witness.v)


class SourceContacts:
    def __init__(self, topology, model, device='cpu'):
        self.topology = topology; self.device = str(device)
        self.faces = np.asarray(topology['faces'], np.int32)
        self.taxels = CurvedTaxels(topology)
        self.patches = np.asarray(topology['sid_patch_indices'])
        self.links = []
        for i, bone in enumerate(model.bone_names):
            name = bone.split('_', 1)[1].replace('little', 'pinky')
            if name == 'wrist' or name.endswith('metacarpal') and not name.startswith('thumb_'): continue
            self.links.append((i, name))
        weights = model.weights.detach().cpu().numpy()
        labels = weights.argmax(1)
        groups = np.array([np.bincount(labels[f], minlength=len(model.bone_names)).argmax() for f in self.faces])
        with torch.no_grad(): rest = model(model.q0[None])[0][0].cpu().numpy()
        closed = closed_surface_parts(rest, self.faces, groups, len(model.bone_names))
        self.groups = groups; self.skins = []; self.solids = []; self.buffers = []; self.loops = []
        self.vertex = wp.array(rest.astype(np.float32), dtype=wp.vec3, device=self.device)
        sources = []; bcs = []; owners = []; self.link_faces = []
        for k, (bone, _) in enumerate(self.links):
            f = self.faces[groups == bone]; self.link_faces.append(f)
            self.skins.append(wp.Mesh(points=self.vertex, indices=wp.array(f.ravel(), dtype=int, device=self.device)))
            v, cf, loops = closed[bone]
            buf = wp.array(v.astype(np.float32), dtype=wp.vec3, device=self.device)
            self.buffers.append(buf); self.loops.append(loops)
            self.solids.append(wp.Mesh(points=buf, indices=wp.array(cf.ravel(), dtype=int, device=self.device), support_winding_number=True))
            sources.extend(np.repeat(f, 4, axis=0)); bcs.extend(np.tile(np.vstack((np.eye(3), np.ones((1,3))/3)), (len(f),1)))
            owners.extend([k]*len(f)*4)
        self.source_np = np.asarray(sources, np.int32); self.bc_np = np.asarray(bcs, np.float32); self.owner_np = np.asarray(owners)
        self.source = wp.array(self.source_np, dtype=wp.vec3i, device=self.device)
        self.bary = wp.array(self.bc_np, dtype=wp.vec3, device=self.device)
        self.owner = wp.array(owners, dtype=int, device=self.device)
        self.skin_ids = wp.array([m.id for m in self.skins], dtype=wp.uint64, device=self.device)
        self.solid_ids = wp.array([m.id for m in self.solids], dtype=wp.uint64, device=self.device)
        n = len(self.links); self.allowed_np = np.zeros((n,n), np.int32)
        for a, b in combinations(range(n), 2):
            fa, fb = [self.links[k][1].split('_')[0] for k in (a,b)]
            if fa != fb or abs(self.links[a][0]-self.links[b][0]) > 1:
                self.allowed_np[a,b] = self.allowed_np[b,a] = 1
        self.allowed = wp.array(self.allowed_np, dtype=int, device=self.device)
        self.result = wp.zeros((len(sources), n), dtype=float, device=self.device)
        self.hit_faces=wp.zeros(self.result.shape,dtype=int,device=self.device)
        self.hit_bary=wp.zeros(self.result.shape,dtype=wp.vec2,device=self.device)
        self.contact_faces=wp.zeros(self.result.shape,dtype=int,device=self.device)
        self.contact_bary=wp.zeros(self.result.shape,dtype=wp.vec2,device=self.device)
        self.contact_valid=wp.zeros(self.result.shape,dtype=int,device=self.device)
        self.retracted=wp.zeros(self.result.shape,dtype=int,device=self.device)

    def detect(self, vertices, joints, frame, *, include_all=False):
        v = np.asarray(vertices, np.float32)
        if v.shape != (len(self.vertex),3) or not np.isfinite(v).all(): raise ValueError('Invalid contact mesh')
        wp.copy(self.vertex, wp.array(v, dtype=wp.vec3, device=self.device))
        boxes = [(v[f].min((0,1)), v[f].max((0,1))) for f in self.link_faces]
        allowed = self.allowed_np.copy()
        for a,b in combinations(range(len(boxes)),2):
            if np.any(boxes[a][0] > boxes[b][1]) or np.any(boxes[b][0] > boxes[a][1]):
                allowed[a,b] = allowed[b,a] = 0
        wp.copy(self.allowed, wp.array(allowed, dtype=int, device=self.device))
        for skin, solid, buf, loops in zip(self.skins, self.solids, self.buffers, self.loops):
            points = np.vstack((v, np.asarray([v[loop].mean(0) for loop in loops]).reshape(-1,3)))
            wp.copy(buf, wp.array(points, dtype=wp.vec3, device=self.device)); skin.refit(); solid.refit()
        self.result.zero_()
        self.contact_valid.zero_(); self.retracted.zero_()
        wp.launch(_source_depth, self.result.shape, inputs=[self.vertex,self.source,self.bary,self.owner,
            self.skin_ids,self.solid_ids,self.allowed,self.result,self.hit_faces,self.hit_bary,
            self.contact_faces,self.contact_bary,self.contact_valid,self.retracted,DEEP_PENETRATION_M], device=self.device)
        depths = self.result.numpy(); pairs = []
        valid=self.contact_valid.numpy().astype(bool); corrected=self.retracted.numpy().astype(bool)
        faces=self.contact_faces.numpy();uv=self.contact_bary.numpy()
        centres=np.asarray([(lo+hi)*.5 for lo,hi in boxes])
        triangles=v[self.source_np]
        source_normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        source_normals/=np.maximum(np.linalg.norm(source_normals,axis=-1,keepdims=True),1.e-12)
        # In a deeply engulfed link, its back-facing samples can also be inside.
        # Those cannot establish the facing side of a contact from this frame.
        facing=(source_normals@centres.T-(source_normals*centres[self.owner_np]).sum(-1)[:,None])>1.e-7
        usable=valid & (~corrected | facing)
        for a,b in combinations(range(len(self.links)),2):
            if not allowed[a,b]: continue
            choices = []
            for src, target in ((a,b),(b,a)):
                ids = np.flatnonzero((self.owner_np == src)&usable[:,target])
                if not len(ids):continue
                idx = ids[depths[ids,target].argmax()]
                choices.append((float(depths[idx,target]), idx, src, target))
            if not choices:continue
            depth, idx, src, target = max(choices)
            if depth <= .0001: continue
            point = (v[self.source_np[idx]] * self.bc_np[idx,:,None]).sum(0)
            face=int(faces[idx,target]);u,w=uv[idx,target];bc=np.array([u,w,1-u-w])
            target_tri=v[self.link_faces[target][face]];hit=(target_tri*bc[:,None]).sum(0)
            normal=source_normals[idx]
            other=np.cross(target_tri[1]-target_tri[0],target_tri[2]-target_tri[0]);other/=max(np.linalg.norm(other),1.e-12)
            names = [self.links[k][1] for k in (src,target)]
            pairs.append(dict(kind='self', regions=names, points_m=[point.tolist(),hit.tolist()],
                normals=[normal.tolist(),other.tolist()], depth_m=depth, target_gap_m=0.,
                witness_resolution='directional_retraction' if corrected[idx,target] else 'nearest_surface',
                separation_m=float(np.linalg.norm(point-hit)),
                material=[dict(vertices=self.source_np[idx].tolist(),barycentric=self.bc_np[idx].tolist()),
                          dict(vertices=self.link_faces[target][face].tolist(),barycentric=bc.tolist())]))
        ready, mask, response = taxel_evidence(frame, self.patches)
        p,n = self.taxels.evaluate(v); objects = []
        from .contact import PATCH_IDS
        for patch in sorted(set(self.patches[mask].tolist())):
            ids = np.flatnonzero(mask & (self.patches == patch)); name = PATCH_IDS[patch]
            values = response[ids]; weights = values if values.sum() > 1e-12 else np.ones(len(ids))
            normal = np.average(n[ids], axis=0, weights=weights); normal /= max(np.linalg.norm(normal),1e-12)
            # A Cartesian mean of curved taxels can lie INSIDE the finger.
            # Restrict the CoP projection to observed material triangles.
            from .wuji_contact_source import closest_triangle
            source_faces=np.unique(np.asarray(self.topology['sid_face_ids'])[ids])
            reference=np.average(p[ids],axis=0,weights=weights)
            point,face,bary=closest_triangle(v[self.faces[source_faces]],reference)
            if len(ids)==1:
                point=p[ids[0]];face=int(np.flatnonzero(source_faces==self.topology['sid_face_ids'][ids[0]])[0])
                bary=np.asarray(self.topology['sid_barycentric'][ids[0]])
            objects.append(dict(kind='object',region=name,position_m=point.tolist(),
                normal=normal.tolist(),force=float(values.sum()),sids=ids.tolist(),responses=values.tolist(),
                observed_points_m=p[ids].tolist(),observed_normals=n[ids].tolist(),
                source_face=int(source_faces[face]),barycentric=bary.tolist(),
                material=[dict(face=int(self.topology['sid_face_ids'][i]),barycentric=self.topology['sid_barycentric'][i]) for i in ids],target_gap_m=0.))
        result=dict(policy=CONTACT_POLICY,tactile_ready=bool(ready),tactile_status='valid' if ready else 'unknown',
                    self_pairs=pairs,object_contacts=objects,timestamp_ms=float(frame['timestamp_ms']),
                    contact_resolution=dict(threshold_m=DEEP_PENETRATION_M,
                        retracted_samples=int((corrected&usable).sum()),
                        unresolved_samples=int(((depths>0)&~usable).sum()),raw_geometry_unchanged=True))
        if include_all:result['self_penetrations']=self.all_penetrations(v,depths)
        return result

    def all_penetrations(self,vertices,depths):
        # Every positive sampled penetration, in BOTH directions. No argmax,
        # top-k, spatial deduplication, or display-cloud substitution.
        samples,targets=np.nonzero(depths>0.)
        source_tri=self.source_np[samples]
        source_bary=self.bc_np[samples]
        source_points=(vertices[source_tri]*source_bary[...,None]).sum(1)
        local_faces=self.hit_faces.numpy()[samples,targets]
        target_tri=np.asarray([self.link_faces[t][f] for t,f in zip(targets,local_faces)],np.int32).reshape(-1,3)
        uv=self.hit_bary.numpy()[samples,targets]
        target_bary=np.c_[uv,1-uv.sum(-1)]
        target_points=(vertices[target_tri]*target_bary[...,None]).sum(1)
        def normals(triangles):
            tri=vertices[triangles];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            return n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1.e-12)
        # Original topology face IDs retain unambiguous material provenance.
        if not hasattr(self,'face_lookup'):
            self.face_lookup={tuple(face):i for i,face in enumerate(self.faces.tolist())}
        return dict(source_sample_indices=samples.tolist(),
            source_regions=[self.links[self.owner_np[i]][1] for i in samples],
            target_regions=[self.links[i][1] for i in targets],
            positions_m=source_points.tolist(),surface_positions_m=target_points.tolist(),
            normals=normals(source_tri).tolist(),surface_normals=normals(target_tri).tolist(),
            depths_m=depths[samples,targets].tolist(),source_vertices=source_tri.tolist(),
            source_barycentric=source_bary.tolist(),
            target_face_indices=[self.face_lookup[tuple(face)] for face in target_tri.tolist()],
            target_barycentric=target_bary.tolist(),
            method='native_link_winding_original_skin_all_positive_samples',
            sampling='original_face_vertices_and_centres_both_directions',
            count=len(samples),truncated=False,continuous_intersection_proof=False)



def transform_contacts(contacts, transform, root, inverse=False):
    """Transform positions and physical normals together (including reflection)."""
    import copy
    result = copy.deepcopy(contacts)
    def point(p):
        return ((np.asarray(p)-root)@transform if not inverse else np.asarray(p)@transform.T+root).tolist()
    def normal(n):
        return (np.asarray(n)@transform if not inverse else np.asarray(n)@transform.T).tolist()
    for pair in result['self_pairs']:
        pair['points_m'] = [point(p) for p in pair['points_m']]
        pair['normals'] = [normal(n) for n in pair['normals']]
    if 'self_penetrations' in result:
        samples=result['self_penetrations']
        for key in ('positions_m','surface_positions_m'):
            p=np.asarray(samples[key]).reshape(-1,3)
            samples[key]=((p-root)@transform if not inverse else p@transform.T+root).tolist()
        for key in ('normals','surface_normals'):
            samples[key]=(np.asarray(samples[key]).reshape(-1,3)@(transform if not inverse else transform.T)).tolist()
    for c in result['object_contacts']:
        c['position_m'] = point(c['position_m']); c['normal'] = normal(c['normal'])
        if 'observed_points_m' in c:c['observed_points_m']=[point(p) for p in c['observed_points_m']]
        if 'observed_normals' in c:c['observed_normals']=[normal(n) for n in c['observed_normals']]
    return result


def overlay(contacts, **metadata):
    points = [p for c in contacts['self_pairs'] for p in c['points_m']] + [p for c in contacts['object_contacts'] for p in c.get('observed_points_m',[c['position_m']])]
    centers = [p for c in contacts['self_pairs'] for p in c['points_m']] + [c['position_m'] for c in contacts['object_contacts']]
    normals = [n for c in contacts['self_pairs'] for n in c['normals']] + [c['normal'] for c in contacts['object_contacts']]
    return dict(ready=True,contact_mode='observed',contact_points_m=points,patch_points_m=patch_cloud(centers,normals),
                self_contacts=len(contacts['self_pairs']),object_contacts=len(contacts['object_contacts']),
                contact_resolution=contacts.get('contact_resolution'),
                tactile_status=contacts['tactile_status'],visualization_only_patch=True,**metadata)
