"""GPU contact geometry and tangent-history rollout for the current patch CEM."""
import math
import torch
from retarget.solver.cem import DEFAULTS
import warp as wp

from .comfree_rollout import RolloutSettings, mesh_mass_properties
from .mano_contact_collision import ManoContactCollision
from .contact_geometry import penetration_terms
from .graspadmm_scene import required_body_groups
from retarget.task.grasp.dexonomy import wuji_body


class _TorchWarpAllocator:
    """Route temporary Warp buffers through the active Torch graph memory pool."""
    def allocate(self,size):
        return torch.cuda.caching_allocator_alloc(size,device=torch.cuda.current_device(),stream=torch.cuda.current_stream())

    def deallocate(self,ptr,size):
        torch.cuda.caching_allocator_delete(ptr)


class ManoRolloutGeometry:
    def __init__(self, decoder, fixed_contacts, surfaces, chunk, *, rollout=None, cuda_graph=True,
                 spring_settings=None, spring_backend='warp',geometry_backend='reference'):
        if decoder.device.type != 'cuda':
            raise ValueError('MANO patch scoring requires CUDA; no CPU fallback')
        self.decoder, self.fixed, self.surfaces, self.chunk = decoder, fixed_contacts, surfaces, chunk
        if geometry_backend not in ('reference','optimized'):raise ValueError('Unknown geometry backend')
        self.geometry_backend=geometry_backend
        if geometry_backend=='optimized':
            from .triangle_penetration import TrianglePenetration
            self.vertex_penetration=TrianglePenetration(surfaces)
        self.settings = DEFAULTS.contact
        self.rollout_settings = rollout or DEFAULTS.rollout
        self.collision = ManoContactCollision(decoder,surfaces,chunk)
        from .spring_history import SpringHistoryRollout, SpringHistorySettings
        self.rollout = SpringHistoryRollout(chunk,self.collision.capacity,device=str(decoder.device),settings=self.rollout_settings,spring=spring_settings or DEFAULTS.spring,backend=spring_backend)
        names = [wuji_body(r['shadow_body']) for r in fixed_contacts.records]
        self.required_groups = [torch.tensor([i for i,n in enumerate(names) if n in group],device=decoder.device)
                                for group in required_body_groups(names)]
        self.group_names = [fixed_contacts.records[int(ids[0])]['finger'] for ids in self.required_groups]
        self.finger_groups = torch.tensor([i for i, name in enumerate(self.group_names) if name != 'palm'],
                                          device=decoder.device,dtype=torch.long)
        constants = {}
        # Asset preparation only; never performed per candidate or rollout step.
        for key in dict.fromkeys(surfaces.object_keys):
            idx, scale = key
            geo = surfaces.geometries[idx]
            if not geo.get('watertight', True):
                raise ValueError('Mesh inertia requires a watertight object')
            vertices = torch.as_tensor(geo['vertices'], device=decoder.device, dtype=torch.float64) * scale
            faces = torch.as_tensor(geo['faces'], device=decoder.device, dtype=torch.long)
            constants[key] = mesh_mass_properties(vertices, faces, self.rollout_settings.mass)
        self.centers = torch.stack([constants[k][0] for k in surfaces.object_keys])
        self.inertias = torch.stack([constants[k][1] for k in surfaces.object_keys])
        self.last = {}
        self.cuda_graph = cuda_graph
        self.graph = None

    @torch.no_grad()
    def evaluate(self, variables, rows, collect=False):
        output = variables.new_empty(len(variables)); collected = {}
        device = str(self.decoder.device)
        with wp.ScopedDevice(device), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.decoder.device))):
            for start in range(0, len(variables), self.chunk):
                value = variables[start:start+self.chunk]; ids = rows[start:start+self.chunk]; count = len(value)
                if count < self.chunk:
                    value = torch.cat((value, value[:1].expand(self.chunk-count,-1)))
                    ids = torch.cat((ids, ids[:1].expand(self.chunk-count)))
                cost,parts = self._score(value,ids)
                output[start:start+count] = cost[:count]
                if collect:
                    for name,v in parts.items():
                        if name not in collected:
                            collected[name] = v.new_empty((len(variables),*v.shape[1:]))
                        collected[name][start:start+count] = v[:count]
        if collect:
            self.last = collected
        return output

    @torch.no_grad()
    def _compute(self,value,ids,*,decoded=None):
        vertices,joints,frames = self.decoder(value,full=True) if decoded is None else decoded
        hand,hn = self.fixed.evaluate(vertices,full=True)
        mapped,on,gap,face,bary = self.surfaces.query(hand,ids)
        if self.geometry_backend=='optimized':
            triangle_depth,surface_valid=self.vertex_penetration.query(vertices,ids)
        else:
            projected,outward,distance,_,_ = self.surfaces.query(vertices,ids)
            triangle_depth=(-distance).clamp_min(0).amax(-1)
            surface_valid=torch.isfinite(projected).all((-2,-1))
        packet = self.collision.query(joints,frames,ids)
        overflow = packet['overflow']
        depth = torch.stack((torch.maximum(triangle_depth,packet['depth'][:,0]),packet['depth'][:,1]),-1)
        hp,op,normal,signed = packet['hand'],packet['object'],packet['normal'],packet['gap']
        rotation = self.surfaces.rotation[ids]
        origin = self.surfaces.origin[ids]
        center = (rotation @ self.centers[ids,:,None]).squeeze(-1)+origin
        response = self.rollout.solve(hp,op,normal,signed,center,rotation,self.inertias[ids],eligible=packet['eligible'])
        penetration,soft,_ = penetration_terms(depth,self.settings)
        cost=response['cost']
        valid=response['valid']&~overflow&torch.isfinite(value).all(-1)&torch.isfinite(cost)
        valid &= torch.isfinite(gap).all(-1)&surface_valid
        parts=dict(penetration=penetration,soft_penetration=soft,
            penetration_depth_m=depth,valid=valid,
            mapped_hand_points=hand,mapped_hand_normals=hn,mapped_object_points=mapped,
            mapped_object_normals=on,mapped_distance=gap,projection_face_ids=face,
            projection_barycentric=bary,collision_contact_count=packet['count'],
            collision_depth_m=packet['depth'],triangle_object_depth_m=triangle_depth,
            contact_hand_points=hp,contact_object_points=op,contact_normals=normal,contact_gap=signed,
            object_final_origin=response['center_world']-(response['rotation']@self.centers[ids,:,None]).squeeze(-1),
            **{'rollout_'+k:v for k,v in response.items()})
        return cost,parts

    def _score(self,value,ids):
        if not self.cuda_graph:
            return self._compute(value,ids)
        if self.graph is None:
            self.graph_values=value.clone();self.graph_rows=ids.clone()
            stream=torch.cuda.Stream(device=self.decoder.device)
            stream.wait_stream(torch.cuda.current_stream(self.decoder.device))
            with torch.cuda.stream(stream):
                self._compute(self.graph_values,self.graph_rows)
                self._compute(self.graph_values,self.graph_rows)
            torch.cuda.current_stream(self.decoder.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            from .capture import capture_guard
            with capture_guard(), wp.ScopedAllocator(str(self.decoder.device),_TorchWarpAllocator()), torch.cuda.graph(graph,stream=stream):
                with wp.ScopedStream(wp.stream_from_torch(stream)):
                    self.graph_output=self._compute(self.graph_values,self.graph_rows)
            self.graph=graph
        self.graph_values.copy_(value);self.graph_rows.copy_(ids)
        self.graph.replay()
        return self.graph_output
