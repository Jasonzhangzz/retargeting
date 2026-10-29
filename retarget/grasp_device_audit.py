"""Per-node geometry and closure audits without host decisions during replay."""
from retarget.grasp_capture import capture_guard
import numpy as np
import torch
import warp as wp
from forceaware import mpc_loss as fa
from retarget.grasp_device_closure import closure_tensors, export_closure


@wp.kernel(enable_backward=False)
def capsule_depth_selected(job: fa.LossJob, selected: wp.array(dtype=int),
                           depth: wp.array(dtype=float)):
    i = wp.tid()
    world = wp.max(selected[0], 0)
    left = job.model.finger_joint_pair_left[i]
    right = job.model.finger_joint_pair_right[i]
    a = fa._finger_joint_world(job, world, left, job.model.finger_joint_start[left])
    b = fa._finger_joint_world(job, world, left, job.model.finger_joint_end[left])
    c = fa._finger_joint_world(job, world, right, job.model.finger_joint_start[right])
    d = fa._finger_joint_world(job, world, right, job.model.finger_joint_end[right])
    gap = wp.sqrt(fa._segment_distance_squared(a, b, c, d)+1.e-12)
    depth[i] = wp.max(job.model.finger_joint_radius[left]+job.model.finger_joint_radius[right]-gap, 0.)


class GraspDeviceAudit:
    def __init__(self, runtime, selected, stream, *, capture_standalone=True):
        self.auditor, hand, self.args = runtime.audit_context
        self.selected, self.stream = selected, stream
        self.torch_stream = torch.cuda.ExternalStream(stream.cuda_stream, device=str(stream.device))
        b = runtime.optimizer.backend
        self.job = (runtime.output_workspace.job if runtime.grasp_optimizer == 'cem' else
                    (b.loss.immediate[1] if 1 < len(b.steps) else b.loss.terminal).job)
        device = str(stream.device)
        tensor = lambda value, dtype: torch.as_tensor(value, device=device, dtype=dtype)
        self.ids = tensor(hand.bodies, torch.long)
        self.local = (wp.to_torch(runtime.card_contact_local) if runtime.grasp_optimizer == 'cem'
                      else tensor(hand.local, torch.float32))
        self.normals = (wp.to_torch(runtime.card_contact_normal) if runtime.grasp_optimizer == 'cem'
                        else tensor(hand.normals, torch.float32))
        self.all_ids = tensor(hand.all_bodies, torch.long)
        self.all_local = tensor(hand.all_local, torch.float32)
        groups = {}
        for i, body in enumerate(hand.bodies):
            name = hand.model.body(int(body)).name
            key = name.split('_link')[0] if name.startswith('right_finger') else 'palm'
            groups.setdefault(key, []).append(i)
        mask = np.zeros((len(groups), len(hand.bodies)), bool)
        for i, ids in enumerate(groups.values()):
            mask[i, ids] = True
        self.groups = tensor(mask, torch.bool)
        self.depth = wp.zeros(self.job.model.finger_joint_pair_count, dtype=float, device=stream.device)
        self.query_buffers = [tuple(wp.empty(count, dtype=dtype, device=stream.device)
            for dtype in (wp.vec3, wp.vec3, float)) for count in (len(hand.local), len(hand.all_local))]
        with torch.no_grad(), wp.ScopedStream(stream), torch.cuda.stream(self.torch_stream):
            self.compute()
        wp.synchronize_stream(stream)
        self.graph = None
        if not capture_standalone:
            return
        # Retain the graph so the scene scheduler can add it as a child of
        # one device DAG, with only the matching solve node as a dependency.
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.no_grad(), wp.ScopedStream(stream):
            with capture_guard(), torch.cuda.graph(self.graph, stream=self.torch_stream):
                self.compute()
        self.graph.instantiate()
        with torch.cuda.stream(self.torch_stream):
            self.graph.replay()
        wp.synchronize_stream(stream)

    def compute(self):
        job, args = self.job, self.args
        index = wp.to_torch(self.selected).long().clamp(min=0)
        position = wp.to_torch(job.state.body_position).index_select(0, index)[0]
        rotation = wp.to_torch(job.state.body_matrix).index_select(0, index)[0]
        obj = job.model.object_body

        def points(ids, local):
            world = torch.bmm(rotation[ids], local[..., None])[..., 0]+position[ids]
            return (world-position[obj])@rotation[obj]

        hand_points = points(self.ids, self.local)
        projected, normals, distance = self.query(hand_points, 0)
        hn = torch.bmm(rotation[self.ids], self.normals[..., None])[..., 0]@rotation[obj]
        cosine = -(hn*normals).sum(-1).clamp(-1., 1.)
        residual = 2*(1-cosine)
        angle = torch.rad2deg(torch.acos(cosine))
        active = (distance.abs() <= args.contact_tolerance_mm/1000)&(cosine > .25)
        dense = points(self.all_ids, self.all_local)
        _, _, dense_distance = self.query(dense, 1)
        penetration = (-dense_distance.min()).clamp(min=0)*1000
        wp.launch(capsule_depth_selected, self.depth.shape, inputs=[job, self.selected, self.depth], device=self.stream.device)
        self_depth = wp.to_torch(self.depth).max()*1000
        required = (self.groups&active[None]).any(dim=1).all()
        geometry = required&(torch.maximum(penetration, self_depth) <= args.penetration_tolerance_mm)
        self.closure = closure_tensors(projected, normals, active, friction=args.friction,
                                      radius=self.auditor.radius, edges=args.cone_edges)
        self.active = active
        self.metrics = dict(active_contacts=active.sum(), required_groups_satisfied=required,
            geometry_feasible=geometry, feasible=geometry&self.closure['force_closure'],
            gap_mean_mm=distance.abs().mean()*1000, gap_max_mm=distance.abs().max()*1000,
            max_object_penetration_mm=penetration, max_self_penetration_mm=self_depth,
            normal_squared_residual_max=residual.max())
        self.arrays = dict(hand_points=hand_points, realized_object_points=projected,
            realized_hand_normals=hn, realized_normal_squared_residual=residual,
            realized_normal_angle_deg=angle, realized_object_normals=normals,
            realized_signed_distance=distance, realized_contact_mask=active)

    def query(self, points, slot):
        from retarget.grasp_audit import query_surface
        source = wp.from_torch(points.contiguous(), dtype=wp.vec3)
        projected, normals, distance = self.query_buffers[slot]
        wp.launch(query_surface, len(points), inputs=[self.auditor.mesh.id, source],
                  outputs=[projected, normals, distance], device=self.stream.device)
        return wp.to_torch(projected), wp.to_torch(normals), wp.to_torch(distance)

    def launch(self):
        if self.graph is None:
            raise RuntimeError('Audit is captured inside the combined node graph')
        with torch.cuda.stream(self.torch_stream):
            self.graph.replay()

    def export(self):
        report = {key: value.detach().cpu().item() for key, value in self.metrics.items()}
        arrays = {key: value.detach().cpu().numpy().copy() for key, value in self.arrays.items()}
        report.update(force_closure=export_closure(self.closure, self.active, self.args.cone_edges),
            normal_squared_residual=arrays['realized_normal_squared_residual'].tolist(),
            normal_angle_deg=arrays['realized_normal_angle_deg'].tolist(),
            template_normal_alignment_in_shared_solver=False,
            self_collision_measurement='same inter-finger capsules as shared GPU objective',
            audit_backend='cuda_graph_mesh_BVH_and_device_primal_dual_LP')
        return report, arrays
