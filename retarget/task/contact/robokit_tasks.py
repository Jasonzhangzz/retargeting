"""Thin execution bridge around the migrated original Robokit task factories."""
from dataclasses import asdict
import importlib
import inspect
from types import SimpleNamespace
import mujoco
import numpy as np
from .tasks import Profile
from retarget.env.contact.native import snapshot, restore


class NamedArray:
    def __init__(self, value: np.ndarray):
        self.value = np.array(value, copy=True)

    def numpy(self) -> np.ndarray:
        return self.value

    def assign(self, value: np.ndarray) -> None:
        self.value = np.array(value, copy=True)


class ExecutionMirror:
    """Original callbacks read this named-array view of measured CPU4 state."""
    def __init__(self, model: mujoco.MjModel, state: mujoco.MjData):
        self.model, self.probe, self.data = model, mujoco.MjData(model), SimpleNamespace()
        self.update(snapshot(state))

    def update(self, state: dict, contacts: list | None = None) -> None:
        restore(self.model, self.probe, state)
        for name in ('qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart'):
            value = state.get(name, np.zeros_like(getattr(self.probe, name)))
            getattr(self.probe, name)[:] = value
            setattr(self.data, name, NamedArray(np.asarray(value)[None]))
        self.data.time = NamedArray(np.array([state.get('time', 0.)]))
        mujoco.mj_forward(self.model, self.probe)
        self.data.sensordata = NamedArray(self.probe.sensordata[None])
        if contacts is None:
            contacts = [dict(geom1=int(c.geom1), geom2=int(c.geom2), geometric_distance=float(c.dist))
                        for c in self.probe.contact]
        self.data.nacon = NamedArray(np.array([len(contacts)], np.int32))
        self.data.contact = SimpleNamespace(
            worldid=NamedArray(np.zeros(len(contacts), np.int32)),
            geom=NamedArray(np.array([[c['geom1'], c['geom2']] for c in contacts], np.int32).reshape(-1, 2)),
            dist=NamedArray(np.array([c['geometric_distance'] for c in contacts])))


class Task:
    def __init__(self, name: str, seed: int = 0, device: str = 'cuda:0'):
        module_name = {'sharpa_pen': 'pen', 'sharpa_coin': 'coin', 'h1_jog': 'h1'}[name]
        module = importlib.import_module(f'retarget.task.contact.source.{module_name}')
        self.name, self.seed, self.source = name, seed, module
        self.context = c = module.prepare(seed, device, ExecutionMirror)
        self.config = cfg = c['config']
        self.scene, self.model, self.mirror = c['scene'], c['model'], c['sim']
        self.plan_scene = c.get('plan', self.scene).copy()
        self.state = c['states'][0] if name == 'sharpa_coin' else c['state']
        self.initial = snapshot(self.state)
        self.adr = {self.model.sensor(i).name: int(self.model.sensor_adr[i]) for i in range(self.model.nsensor)}
        self.ctrl_std = np.broadcast_to(cfg.ctrl_std, self.model.nu).copy()
        self.ctrl_range = self.model.actuator_ctrlrange.copy() if cfg.ctrl_range is None else cfg.ctrl_range.copy()
        stage = cfg.stages[0]
        schedule = stage.dt if isinstance(stage.dt, list) else [(1., stage.dt)]
        self.profile = Profile(module.STEPS, cfg.horizon_steps, cfg.node_steps, cfg.ctrl_steps,
            cfg.num_samples, cfg.max_iterations, cfg.init_iterations,
            cfg.interpolation if isinstance(cfg.interpolation, str) else 'quadratic', cfg.noise_decay,
            float(schedule[0][0]), round((schedule[-1][1] or self.model.opt.timestep) / self.model.opt.timestep))
        self.owner = np.array([next((i for i, finger in enumerate(('thumb', 'index', 'middle', 'ring', 'pinky'))
            if f'_{finger}_' in self.model.body(int(self.model.geom_bodyid[g])).name), 5)
            for g in range(self.model.ngeom)])
        self.grasp = np.asarray(module.GRASP if name != 'h1_jog' else self.model.actuator_ctrlrange.mean(1))
        self.pen_first = float(c.get('first', 0.))
        self.pen_heading, self.goal, self.phase, self.clock = self.pen_first, 0, 0., 0
        self.reference = None
        self.attempts = int(c.get('attempt', -1)) + 1
        if name == 'sharpa_coin':
            self.lines, self.lanes, self.tops = [c[key][0] for key in ('lines', 'lanes', 'tops')]
        self.ctrl_weight = np.zeros(self.model.nu)
        for term, _, _ in cfg.terms:
            if type(term).__name__ == 'ControlTask':
                self.ctrl_weight = np.broadcast_to(term.weight, self.model.nu).copy()

    def bind_reference(self) -> None:
        """Bind the frozen artifact to original SensorTask arrays before graph capture."""
        if self.reference is None:
            return
        target = self.context['sensor_ref']
        ids = np.minimum(np.arange(len(target)), len(self.reference['qpos']) - 1)
        for sensor in ('torso_vel', 'torso_gyro', 'torso_pos', 'pelvis_zaxis', 'heading', 'left_foot', 'right_foot'):
            a, width = self.adr[sensor], int(self.model.sensor(sensor).dim[0])
            target[:, a:a + width] = self.reference['sensordata'][ids, a:a + width]
        control = self.context['ctrl_ref']
        control[:] = self.reference['ctrl'][np.minimum(np.arange(len(control)), len(self.reference['ctrl']) - 1)]

    def packet(self, state: dict, clock: int, contacts: list = ()) -> dict:
        c, cfg, m, H = self.context, self.config, self.model, self.profile.horizon
        self.mirror.update(state, list(contacts))
        mujoco.mj_copyData(self.state, m, self.mirror.probe)
        if clock != self.clock and 'lead' in c:
            c['lead']([dict(t0=self.clock)])
        self.clock = clock
        if 'lead' in c:
            variables = inspect.getclosurevars(c['lead']).nonlocals
            self.goal = int(np.asarray(variables.get('goal', 0)).ravel()[0])
            self.phase = float(np.asarray(variables.get('s', 0.)).ravel()[0])
            self.pen_heading = float(variables.get('heading', self.pen_first))
        source_ref = c['sensor_ref'] if self.name == 'h1_jog' else c['ref']
        ref = source_ref[clock:clock + H + 1] if source_ref.ndim == 2 else source_ref[0, clock:clock + H + 1]
        controls = c['ctrl_ref']
        controls = controls if controls.ndim == 2 else controls[0]
        ctrl_ref = controls[clock:clock + H + 1].copy()
        weight, terminal = np.zeros(m.nsensordata), np.zeros(m.nsensordata)
        heading, heading_weight, heading_terminal, support_obj = -1, 0., 1., -1
        descriptors = []
        for term, running, final in cfg.terms:
            final = cfg.terminal_scale if final is None else final
            kind = type(term).__name__
            if kind == 'SensorTask':
                if term.residual == 5:
                    sensor, value = next(iter(term.weight.items()))
                    heading, heading_weight, heading_terminal = self.adr[sensor], value * running, final / running
                else:
                    for sensor, value in term.weight.items():
                        a = self.adr[sensor]; width = int(m.sensor(sensor).dim[0])
                        weight[a:a + width] += np.asarray(value) * running
                        terminal[a:a + width] += np.asarray(value) * final
                descriptors.append(dict(type=kind, weight=term.weight, residual=term.residual,
                    norm=term.norm, running_scale=running, terminal_scale=final))
            elif kind == 'ControlTask':
                descriptors.append(dict(type=kind, weight=np.asarray(term.weight), norm=term.norm,
                    running_scale=running, terminal_scale=final))
            elif kind == 'SupportTask':
                support_obj = m.geom(term.obj).id
                descriptors.append(dict(type=kind, obj=term.obj, groups=term.groups, need=term.need,
                    weight=term.weight, margin=term.margin, free=term.free,
                    running_scale=running, terminal_scale=final))
            else:
                raise TypeError(f'Unsupported original task term: {kind}')
        packet = dict(ref=ref.copy(), weight=weight, terminal=terminal, ctrl_ref=ctrl_ref,
            ctrl_weight=self.ctrl_weight.copy(), heading=heading, heading_weight=heading_weight,
            heading_terminal=heading_terminal, support_obj=support_obj, support_margin=.0025,
            owner=self.owner.copy(), dt=m.opt.timestep, robokit_terms=descriptors,
            initial_ctrl=np.asarray(state.get('ctrl', self.mirror.probe.ctrl)).copy(),
            target_geoms=[m.geom('floor' if self.name == 'h1_jog' else 'pen' if self.name == 'sharpa_pen' else 'coin').id])
        if self.reference is not None:
            ids = np.minimum(clock + np.arange(H + 1), len(self.reference['qpos']) - 1)
            packet['qpos_ref'], packet['qvel_ref'] = self.reference['qpos'][ids], self.reference['qvel'][ids]
            packet['ctrl_ref'] = self.reference['ctrl'][np.minimum(ids, len(self.reference['ctrl']) - 1)]
            for sensor in ('torso_vel', 'torso_gyro', 'torso_pos', 'pelvis_zaxis', 'heading', 'left_foot', 'right_foot'):
                a, width = self.adr[sensor], int(m.sensor(sensor).dim[0])
                packet['ref'][:, a:a + width] = self.reference['sensordata'][ids, a:a + width]
        return packet

    def metadata(self) -> dict:
        from dataclasses import fields
        config = {field.name: getattr(self.config, field.name) for field in fields(self.config)
                  if field.name != 'terms'}
        config['stages'] = [asdict(stage) for stage in self.config.stages]
        if not np.isfinite(config['min_improvement']):
            config['min_improvement'] = str(config['min_improvement'])
        return dict(name=self.name, seed=self.seed, profile=asdict(self.profile),
            reset_attempts=self.attempts, robokit_config=config,
            task_source=self.source.__file__, implementation='original_robokit_callbacks')
