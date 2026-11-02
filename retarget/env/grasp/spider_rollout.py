"""CUDA-graph batched native MJWarp rollouts for SPIDER MPPI."""

import mujoco_warp as mjw
import torch
import warp as wp

from retarget.task.grasp.spider_cost import make_hand_scale, trajectory_cost
from .capture import capture_guard
from .spider_physics import BatchedPhysics


class SamplingRollout:
    def __init__(self, cpu, reference, warm, *, worlds, horizon_steps, cost_settings,
                 device="cuda:0", nconmax=256, njmax=2048):
        self.device, self.worlds, self.horizon = device, worlds, horizon_steps
        self.physics = BatchedPhysics(cpu, worlds, device, nconmax=nconmax,
                                      njmax=njmax, capture=False)
        self.settings = cost_settings
        tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, device=device, dtype=dtype)
        self.bodies = tensor(reference.contact_bodies, torch.long)
        self.local = tensor(reference.contact_local)
        self.qref = tensor(reference.qpos).expand(horizon_steps, -1).clone()
        self.vref = tensor(reference.qvel).expand(horizon_steps, -1).clone()
        qw, qv = self.qref[:, -4:-3], self.qref[:, -3:]
        anchors = tensor(reference.object_local)[None].expand(horizon_steps, -1, -1)
        # Rotate object-frame anchors by the reference WXYZ orientation.
        uv = torch.cross(qv[:, None].expand_as(anchors), anchors, dim=-1)
        rotated = anchors+2.*(qw[:, None]*uv+torch.cross(qv[:, None].expand_as(anchors), uv, dim=-1))
        self.targets = rotated+self.qref[:, None, 26:29]
        self.mask = torch.ones((horizon_steps, len(reference.contact_bodies)), device=device)
        self.hand_scale = make_hand_scale(cost_settings, device=device)
        self.temporal = torch.ones(horizon_steps, device=device)
        self.temporal[-1] = cost_settings.terminal_rew_scale
        self.controls = tensor(warm)[None, None].expand(worlds, horizon_steps, -1).clone()
        self.qtrace = torch.empty((worlds, horizon_steps, 33), device=device)
        self.vtrace = torch.empty((worlds, horizon_steps, 32), device=device)
        self.ctrace = torch.empty((worlds, horizon_steps, len(reference.contact_bodies), 3), device=device)
        self.contact_loads, self.ftrace = None, None
        self.group_geometry=None;self.ptrace=None;self.wtrace=None
        self.required_fingers=None
        if reference.required_groups is not None:
            from .spider_geometry import NativeContactGeometry
            from .spider_seed import load_seed
            self.group_geometry=NativeContactGeometry(self.physics,load_seed(reference.seed_source)[2])
            self.required_fingers=tensor(reference.required_groups[:5],torch.bool)
            self.ftrace=torch.empty((worlds,horizon_steps,5),device=device)
            if cost_settings.palm_keep_rew_scale>0:self.ptrace=torch.empty((worlds,horizon_steps),device=device)
            if cost_settings.wrench_rew_scale>0:self.wtrace=torch.empty((worlds,horizon_steps),device=device)
        elif cost_settings.contact_keep_rew_scale > 0:
            from .spider_contact import ContactLoads
            self.contact_loads = ContactLoads(self.physics, reference.contact_bodies)
            self.ftrace = torch.empty((worlds, horizon_steps, 5), device=device)
            self.required_fingers=self.contact_loads.required_fingers
        self.valid = torch.ones(worlds, dtype=torch.bool, device=device)
        self.qinitial, self.vinitial = tensor(reference.qpos), tensor(reference.qvel)
        self.physics.reset(self.qinitial, self.vinitial, control=tensor(warm))
        self.state = self.physics.snapshot()
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream), wp.ScopedStream(wp.stream_from_torch(self.stream)):
            self.physics.warmup()
            self.physics.restore(self.state)
            self._run()
        torch.cuda.current_stream(device).wait_stream(self.stream)
        torch.cuda.synchronize(device)
        self.physics.restore(self.state)
        self.graph = torch.cuda.CUDAGraph()
        # Select both streams before capture: Warp's stream-entry event sync
        # would invalidate an active global capture for nondefault callers.
        with torch.cuda.stream(self.stream), wp.ScopedStream(wp.stream_from_torch(self.stream)):
            with capture_guard(), torch.cuda.graph(self.graph, stream=self.stream):
                # Retain Warp temporaries for Torch's external CUDA graph.
                with wp.ScopedCapture(stream=wp.stream_from_torch(self.stream), external=True,
                                      force_module_load=False,
                                      capture_mode=wp.CaptureMode.GLOBAL) as external:
                    self._run()
                self._external_capture = external
        torch.cuda.synchronize(device)

    def _run(self):
        self.valid.fill_(True)
        for step in range(self.horizon):
            self.physics.step(self.controls[:, step], use_graph=False)
            if self.group_geometry is not None:
                loads=self.group_geometry.sample()
                self.ftrace[:,step].copy_(loads[:,:5])
                if self.ptrace is not None:self.ptrace[:,step].copy_(loads[:,5])
                if self.wtrace is not None:
                    # Check wrench support every 20 ms and at the horizon end;
                    # real contact-load costs still cover every physics step.
                    stride=10
                    if (step+1)%stride==0 or step==self.horizon-1:
                        weight=stride if (step+1)%stride==0 else (step+1)%stride
                        self.wtrace[:,step].copy_(self.group_geometry.wrench_cost()*weight)
                    else:self.wtrace[:,step].zero_()
            if self.contact_loads is not None:
                # Actual normal forces belong to these solver contact rows;
                # copy before the following native step overwrites the buffer.
                self.ftrace[:, step].copy_(self.contact_loads.sample())
            self.qtrace[:, step].copy_(self.physics.qpos)
            self.vtrace[:, step].copy_(self.physics.qvel)
            # Preserve SPIDER's native observation staging: post-step qpos/qvel,
            # pre-integration body/site transforms from that same engine step.
            positions = self.physics.xpos[:, self.bodies]
            matrices = self.physics.xmat[:, self.bodies]
            self.ctrace[:, step].copy_(positions+torch.einsum("bcij,cj->bci", matrices, self.local))
            self.valid.logical_and_(wp.to_torch(self.physics.data.overflow) == 0)
        self.parts = trajectory_cost(self.qtrace, self.vtrace, self.qref, self.vref,
            contact_positions=self.ctrace, contact_target_world=self.targets,
            contact_mask=self.mask, settings=self.settings, hand_scale=self.hand_scale,
            temporal_weights=self.temporal, finger_normal_forces=self.ftrace,
            required_finger_mask=self.required_fingers,palm_normal_forces=self.ptrace,wrench_support_cost=self.wtrace)
        self.cost = torch.where(self.valid, self.parts["total"], torch.inf)

    def prepare(self, state):
        self.state = state

    def evaluate(self, controls):
        values = []
        for start in range(0, len(controls), self.worlds):
            chunk = controls[start:start+self.worlds]
            self.controls[:len(chunk)].copy_(chunk)
            if len(chunk) < self.worlds:
                self.controls[len(chunk):].copy_(chunk[:1].expand(self.worlds-len(chunk), -1, -1))
            # Outside the capture: every candidate starts from the current
            # executed integration state, never a stale captured qpos pointer.
            self.physics.restore(self.state)
            self.graph.replay()
            values.append(self.cost[:len(chunk)].clone())
        return torch.cat(values)
