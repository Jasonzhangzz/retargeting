"""Independent grasp nodes with device-resident, fixed-budget CUDA graphs.

Mesh/model construction and graph compilation are preparation. No Python callback,
host readback or synchronization runs between optimization iterations on replay.
Each heterogeneous scene owns its model, buffers, graph, and non-default stream.
"""
from time import perf_counter
from retarget.grasp_capture import capture_guard
import numpy as np
import warp as wp


@wp.struct
class AdamState:
    raw: wp.array2d(dtype=float)
    gradient: wp.array2d(dtype=float)
    loss: wp.array(dtype=float)
    valid: wp.array(dtype=int)
    scale: wp.array(dtype=wp.float64)
    x: wp.array2d(dtype=wp.float64)
    m: wp.array2d(dtype=wp.float64)
    v: wp.array2d(dtype=wp.float64)
    best: wp.array2d(dtype=wp.float64)
    best_m: wp.array2d(dtype=wp.float64)
    best_v: wp.array2d(dtype=wp.float64)
    count: wp.array(dtype=int)
    best_count: wp.array(dtype=int)
    best_value: wp.array(dtype=wp.float64)
    offset: int
    width: int
    lr: wp.float64
    decay: wp.float64
    beta1: wp.float64
    beta2: wp.float64
    epsilon: wp.float64


@wp.kernel(enable_backward=False)
def begin_block(s: AdamState, first: int):
    world = wp.tid()
    s.best_value[world] = wp.float64(wp.inf)
    s.count[world] = s.best_count[world]
    if first == 1:
        s.count[world] = 0
    for j in range(s.width):
        s.x[world, j] = wp.float64(s.raw[world, s.offset+j])/s.scale[j]
        s.best[world, j] = s.x[world, j]
        s.m[world, j] = s.best_m[world, j]
        s.v[world, j] = s.best_v[world, j]
        if first == 1:
            s.m[world, j] = wp.float64(0.)
            s.v[world, j] = wp.float64(0.)
        s.best_m[world, j] = s.m[world, j]
        s.best_v[world, j] = s.v[world, j]
    s.best_count[world] = s.count[world]


@wp.kernel(enable_backward=False)
def adam_step(s: AdamState, update: int):
    world = wp.tid()
    value = wp.float64(s.loss[world])*wp.float64(1.e-6)
    eligible = s.valid[world] != 0 and wp.isfinite(value)
    if update == 1:
        for j in range(s.width):
            if not wp.isfinite(s.gradient[world, s.offset+j]):
                eligible = False
    if eligible and value < s.best_value[world]:
        s.best_value[world] = value
        s.best_count[world] = s.count[world]
        for j in range(s.width):
            s.best[world, j] = s.x[world, j]
            s.best_m[world, j] = s.m[world, j]
            s.best_v[world, j] = s.v[world, j]
    if eligible and update == 1:
        s.count[world] += 1
        correction1 = wp.float64(1.)-wp.pow(s.beta1, wp.float64(s.count[world]))
        correction2 = wp.float64(1.)-wp.pow(s.beta2, wp.float64(s.count[world]))
        for j in range(s.width):
            g = wp.float64(s.gradient[world, s.offset+j])*s.scale[j]
            m = s.beta1*s.m[world, j]+(wp.float64(1.)-s.beta1)*g
            v = s.beta2*s.v[world, j]+(wp.float64(1.)-s.beta2)*g*g
            s.m[world, j] = m
            s.v[world, j] = v
            x = s.x[world, j]*(wp.float64(1.)-s.lr*s.decay)
            x -= s.lr*(m/correction1)/(wp.sqrt(v/correction2)+s.epsilon)
            s.x[world, j] = x
            s.raw[world, s.offset+j] = float(x*s.scale[j])


@wp.kernel(enable_backward=False)
def restore_best(s: AdamState):
    world, j = wp.tid()
    s.x[world, j] = s.best[world, j]
    s.raw[world, s.offset+j] = float(s.best[world, j]*s.scale[j])


@wp.kernel(enable_backward=False)
def finish_candidates(s: AdamState, scores: wp.array(dtype=float),
                      norms: wp.array(dtype=float), available: wp.array(dtype=int),
                      selected: wp.array(dtype=int)):
    # One small reduction per node; all scenes execute independently on the GPU.
    chosen = int(-1)
    best = float(wp.inf)
    for world in range(s.raw.shape[0]):
        norm2 = float(0.)
        finite = s.valid[world] != 0 and wp.isfinite(s.loss[world])
        for j in range(s.raw.shape[1]):
            finite = finite and wp.isfinite(s.raw[world, j])
            norm2 += s.gradient[world, j]*s.gradient[world, j]
        value = float(wp.inf)
        if finite:
            value = s.loss[world]
        scores[world] = value
        norms[world] = wp.sqrt(norm2)
        available[world] = int(finite)
        if value < best:
            best, chosen = value, world
    selected[0] = chosen


class GraspDeviceNode:
    """One model's complete contact/AdamW/dual schedule, captured once."""
    def __init__(self, runtime, *, capture_standalone=True):
        from retarget.fa.solver import initialize_points, update_contact_dual
        self.runtime = runtime
        self.solver = opt = runtime.optimizer
        self.backend = b = opt.backend
        self.device = b.device
        settings = runtime.settings
        self.method = getattr(runtime, 'grasp_optimizer', 'adamw')
        if self.method != 'adamw':
            raise ValueError('Use CardCemNode for static CARD CEM, or solve_nodes for automatic dispatch')
        if settings.hand_solver != 'adamw' or settings.contact_solver != 'sampled-lm':
            raise ValueError('GPU nodes require AdamW and sampled-lm')
        if settings.adamw_persist:
            raise ValueError('Static GPU nodes have independent moments; --adamw-persist is not applicable')
        self.stream = wp.Stream(self.device)
        self.done = wp.Event(self.device, enable_timing=True)
        self.start_event = wp.Event(self.device, enable_timing=True)
        self.end_event = wp.Event(self.device, enable_timing=True)
        self.started = perf_counter()
        b._prepare_window(runtime.qpos, runtime.qvel, runtime.targets,
            warm=runtime.initial_warm, previous_action=runtime.initial_action,
            seed=runtime.cfg.optimizer.multistart_seed, start_time=0.)
        raw = b.control.raw.numpy()
        raw[:, :b.index.action_dim] = 0.
        b.control.raw.assign(raw)
        self.seed = wp.array(raw, dtype=float, device=self.device)
        self.selected = wp.zeros(1, dtype=int, device=self.device)
        state = self.state = AdamState()
        state.raw, state.gradient = b.control.raw, b.control.gradient
        state.loss, state.valid = b.loss.total, b.control.valid
        state.offset = b.index.action_dim
        state.width = raw.shape[1]-state.offset
        state.scale = wp.array(opt.variable_scale, dtype=wp.float64, device=self.device)
        shape = (b.settings.worlds, state.width)
        for name in ('x', 'm', 'v', 'best', 'best_m', 'best_v'):
            setattr(state, name, wp.zeros(shape, dtype=wp.float64, device=self.device))
        state.count = wp.zeros(b.settings.worlds, dtype=int, device=self.device)
        state.best_count = wp.zeros_like(state.count)
        state.best_value = wp.empty(b.settings.worlds, dtype=wp.float64, device=self.device)
        state.lr, state.decay = settings.adamw_lr, settings.adamw_weight_decay
        state.beta1, state.beta2, state.epsilon = .9, .999, 1.e-8
        self.rounds = settings.contact_rounds if settings.contact_admm else 1
        self.workspaces = (*b.loss.immediate, b.loss.terminal)
        self.scales = []
        # Constants depend on target metadata only; computed once during setup.
        for workspace in self.workspaces:
            target = workspace.job.target
            weight = target.query_mask.numpy()*target.query_weight.numpy()
            self.scales.append(.5*settings.contact_rho*1.e6/
                (settings.contact_radius**2*max(float(weight.sum()), 1.e-12))
                if opt.contact_explorer is not None else 0.)
        # Compile and bind existing production geometry/adjoint kernels before
        # capture. This preparation is explicit and excluded from replay timing.
        opt._current_targets = runtime.targets
        opt._forward_variables = None
        opt._objective_dirty = False
        opt.evaluations = 0
        opt._contact_rollout_seconds = []
        opt._contact_geometry_seconds = []
        opt._contact_device_seconds = []
        variables = raw[:, state.offset:].astype(float)/opt.variable_scale
        if opt.contact_explorer is not None:
            opt._contact_block(variables)
            if hasattr(opt, '_lambda_device'):
                stages, worlds, contacts = len(self.workspaces), b.settings.worlds, b.guidance_count
                self.updated = opt._lambda_device.job.selected.reshape((stages, worlds, contacts))
        opt._evaluate(variables, True)
        wp.synchronize_device(self.device)
        self._initialize_points = initialize_points
        self._update_dual = update_contact_dual
        # Prime lazy buffers and load every kernel, then capture the identical
        # fixed schedule. Replaying starts by resetting all node state.
        self.graph = None
        if capture_standalone:
            with wp.ScopedStream(self.stream):
                self._compute()
            wp.synchronize_stream(self.stream)
            with wp.ScopedStream(self.stream):
                with capture_guard(), wp.ScopedCapture(stream=self.stream) as capture:
                    self._compute()
            self.graph = capture.graph
            wp.capture_launch(self.graph, stream=self.stream)
            wp.synchronize_stream(self.stream)
        else:
            # Kernels/adjoints were primed above. The combined DAG owns capture;
            # avoid constructing and executing a second standalone full solve.
            wp.load_module(module=__name__, device=self.device)
        self.audit = None
        if hasattr(runtime, 'audit_context'):
            from retarget.grasp_device_audit import GraspDeviceAudit
            self.audit = GraspDeviceAudit(runtime, self.selected, self.stream,
                                          capture_standalone=capture_standalone)
        self.prepare_seconds = perf_counter()-self.started

    def _evaluate(self, gradient):
        self.backend.control.valid.fill_(1)
        self.solver._launch(gradient)

    def _compute(self):
        for _ in self._compute_steps():
            pass

    def _compute_steps(self):
        """Capture-time pieces; replay contains kernels, never this iterator.

        Recording nodes breadth first prevents a long first scene from taking
        creation-order scheduling priority over every other ready scene.
        """
        b, opt, state = self.backend, self.solver, self.state
        wp.copy(b.control.raw, self.seed)
        state.count.zero_()
        state.best_count.zero_()
        for workspace, points, scale in zip(self.workspaces, b.objective.points, self.scales):
            wp.launch(self._initialize_points, points.c.shape, inputs=[workspace.job, points], device=self.device)
            points.consensus_scale.fill_(scale)
        yield
        for block in range(self.rounds):
            self._evaluate(False)
            yield
            if opt.contact_explorer is not None:
                if hasattr(opt, '_lambda_device'):
                    self._contact()
                else:
                    from retarget.fa.solver import freeze_contact_points
                    for workspace, points in zip(self.workspaces, b.objective.points):
                        wp.launch(freeze_contact_points, points.c.shape, inputs=[workspace.job, points], device=self.device)
            yield
            wp.launch(begin_block, b.settings.worlds, inputs=[state, int(block == 0)], device=self.device)
            for _ in range(b.settings.iterations):
                self._evaluate(True)
                wp.launch(adam_step, b.settings.worlds, inputs=[state, 1], device=self.device)
                yield
            self._evaluate(False)
            wp.launch(adam_step, b.settings.worlds, inputs=[state, 0], device=self.device)
            wp.launch(restore_best, state.x.shape, inputs=[state], device=self.device)
            self._evaluate(False)
            yield
            if opt.contact_explorer is not None:
                opt._contact_reduction.zero_()
                for workspace, points in zip(self.workspaces, b.objective.points):
                    wp.launch(self._update_dual, points.c.shape, inputs=[workspace.job, points,
                        4*self.runtime.settings.contact_radius, opt._contact_reduction], device=self.device)
            yield
        for points in b.objective.points:
            points.consensus_scale.zero_()
        self._evaluate(False)
        wp.copy(b.control.best_raw, b.control.raw)
        wp.launch(finish_candidates, 1, inputs=[state, b.control.best_loss,
            b.control.best_grad_norm, b.control.best_available, self.selected], device=self.device)

    def _contact(self):
        from retarget.fa.solver import gather_contact_batch, scatter_contact_batch
        b, opt = self.backend, self.solver
        dimensions = (b.settings.worlds, b.guidance_count)
        for stage, (workspace, points) in enumerate(zip(self.workspaces, b.objective.points)):
            velocity = (b.steps[stage].inputs.dynamics.qvel if stage < len(self.workspaces)-1
                        else b.steps[-1].recorded.result.qvel)
            wp.launch(gather_contact_batch, dimensions, inputs=[workspace.job, points, stage,
                velocity, opt.contact_object_dof, opt._contact_batch], device=self.device)
        opt._lambda_device._launch()
        for stage, points in enumerate(b.objective.points):
            wp.launch(scatter_contact_batch, dimensions, inputs=[self.updated, stage, points], device=self.device)

    def launch(self):
        if self.graph is None:
            raise RuntimeError('This node belongs to a combined graph; launch its CudaNodeGraph')
        with wp.ScopedStream(self.stream, sync_enter=False):
            wp.record_event(self.start_event)
            wp.capture_launch(self.graph)
            wp.record_event(self.end_event)
            if self.audit is not None:
                self.audit.launch()
            wp.record_event(self.done)

    def export(self):
        """Called only after the group completion barrier; host I/O starts here."""
        from forceaware.optimizer_contract import finalize_candidate_evaluation
        b, runtime = self.backend, self.runtime
        selected = int(self.selected.numpy()[0])
        if selected < 0:
            raise FloatingPointError('All GPU node candidates are invalid')
        scores = b.control.best_loss.numpy()
        valid = b.control.best_available.numpy().astype(bool)
        evaluation = finalize_candidate_evaluation(scores, b._candidate_drifts(runtime.targets, 1),
            has_finite_best=valid, hard_simulator_valid=valid,
            training_failure_iteration=np.full(len(valid), -1, np.int32),
            training_failure_code=np.zeros(len(valid), np.int32))
        plan = b._plan(selected, self.started, evaluation=evaluation, execution_steps=1)
        gpu_seconds = wp.get_event_elapsed_time(self.start_event, self.end_event, synchronize=False)/1000.
        from dataclasses import replace
        plan = replace(plan, seconds=gpu_seconds)
        self.solver.last_solver_diag = dict(solver='adamw', evaluation_backend='native',
            gradient_source='analytic_adjoint', finite_difference_gradients=False,
            scheduling='independent_cuda_graph_nodes', device_resident_iterations=True,
            host_readbacks_during_replay=0, host_syncs_during_replay=0,
            cpu_trigrasp_seed=False, search_starts=b.settings.worlds,
            evaluation_worlds=b.settings.worlds, hand_iteration_budget=self.rounds*b.settings.iterations,
            contact_admm_enabled=self.solver.contact_explorer is not None,
            contact_surface_optimizer='sampled_comfree_batched_lm',
            graph_prepare_seconds=self.prepare_seconds, gpu_seconds=gpu_seconds)
        if hasattr(self, 'dispatch_timing'):
            self.solver.last_solver_diag['node_dispatch'] = self.dispatch_timing
        b._commit_regularization_history()
        runtime.output_step = 1
        runtime.output_workspace = b.loss.immediate[1] if 1 < len(b.steps) else b.loss.terminal
        runtime.predicted_terminal_qpos = b.steps[-1].recorded.result.qpos.numpy()[selected].copy()
        snapshot = b.objective.snapshot(selected)
        if self.audit is not None:
            runtime.device_audit_result = self.audit.export()
        return plan, snapshot, plan.next_qpos.copy()


class CudaNodeGraph:
    """One flat CUDA DAG with independent solve -> audit branches.

    Fork/join events capture all scene streams into a single graph. Keeping
    kernels in the flat graph avoids child-graph scheduling serialization.
    """
    def __init__(self, nodes):
        import torch
        self.nodes = nodes  # Own the captured buffers throughout replay.
        self.stream = wp.Stream(nodes[0].device)
        self.torch_stream = torch.cuda.ExternalStream(self.stream.cuda_stream, device=str(nodes[0].device))
        self.origin = wp.Event(nodes[0].device, enable_timing=True)
        self.fork = wp.Event(nodes[0].device)
        self.joins = [wp.Event(node.device) for node in nodes]
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), wp.ScopedStream(self.stream, sync_enter=False):
            with capture_guard(), torch.cuda.graph(self.graph, stream=self.torch_stream):
                # Register this externally owned capture with Warp, including
                # explicit event nodes so CUDA timing survives capture/replay.
                with capture_guard(), wp.ScopedCapture(stream=self.stream, external=True):
                    self._record()

    def _record(self):
        import torch
        wp.record_event(self.origin, external=True)
        wp.record_event(self.fork)
        streams = [torch.cuda.ExternalStream(node.stream.cuda_stream, device=str(node.device)) for node in self.nodes]
        for node, stream in zip(self.nodes, streams):
            with wp.ScopedStream(node.stream, sync_enter=False), torch.cuda.stream(stream):
                wp.wait_event(self.fork)
                wp.record_event(node.start_event, external=True)
        active = [(node, stream, iter(node._compute_steps())) for node, stream in zip(self.nodes, streams)]
        while active:
            remaining = []
            for node, stream, operations in active:
                with wp.ScopedStream(node.stream, sync_enter=False), torch.cuda.stream(stream):
                    try:
                        next(operations)
                        remaining.append((node, stream, operations))
                    except StopIteration:
                        wp.record_event(node.end_event, external=True)
            active = remaining
        for node, stream, join in zip(self.nodes, streams, self.joins):
            with wp.ScopedStream(node.stream, sync_enter=False), torch.cuda.stream(stream):
                if node.audit is not None:
                    node.audit.compute()
                wp.record_event(node.done, external=True)
                wp.record_event(join)
        for join in self.joins:
            wp.wait_event(join)

    def launch(self):
        import torch
        # Caller-side Warp/Torch writes may use different producer streams.
        # Order them before replay with device events, without a host barrier.
        self.stream.wait_stream(wp.get_stream(self.stream.device))
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.torch_stream.device))
        with torch.cuda.stream(self.torch_stream):
            self.graph.replay()

    def close(self):
        self.graph.reset()


def solve_nodes(runtimes):
    """Submit one DAG containing every independent node, then wait once."""
    from retarget.grasp_cem import CardCemNode
    nodes = [(CardCemNode if runtime.grasp_optimizer == 'cem' else GraspDeviceNode)(
        runtime, capture_standalone=False) for runtime in runtimes]
    if not nodes:
        return []
    dag = CudaNodeGraph(nodes)
    origin = dag.origin
    try:
        dag.launch()
        wp.synchronize_stream(dag.stream)
    finally:
        dag.close()
    intervals = [(wp.get_event_elapsed_time(origin, node.start_event, synchronize=False)/1000.,
                  wp.get_event_elapsed_time(origin, node.end_event, synchronize=False)/1000.) for node in nodes]
    events = sorted([(start, 1) for start, _ in intervals]+[(end, -1) for _, end in intervals])
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    for node, (start, end) in zip(nodes, intervals):
        node.dispatch_timing = dict(nodes=len(nodes), observed_overlapping_compute_graphs=peak,
            start_seconds=start, end_seconds=end,
            completed_seconds=wp.get_event_elapsed_time(origin, node.done, synchronize=False)/1000.)
    return [node.export() for node in nodes]
