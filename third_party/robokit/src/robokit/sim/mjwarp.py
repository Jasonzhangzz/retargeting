# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
"""`MjWarpSim`: MuJoCo Warp worlds of one model or several scenes stepped together, their fields as Warp arrays."""

import dataclasses
import gc
import re
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src import math as mjmath  # wxyz quaternions, as MuJoCo Warp's


if TYPE_CHECKING:
    from robokit.sim.mjentity import Entity
    from robokit.sim.mjsensor import ContactSensor


# --- fields ----------------------------------------------------------------
# The data fields `reset` restores.
_STATE = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "time")
# The model fields `mjw.set_const` derives from masses and inertias; randomized masses need them per world too.
_DERIVED = (
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
)
# The model fields with a world axis; the others are shared by every world.
_BATCHED = {f.name for f in dataclasses.fields(mjw.Model) if getattr(f.type, "shape", ())[:1] == ("*",)}


class _Fields:
    """Cached Warp arrays of the fields of a `mjw.Data`, `mjw.Model` or `mjw.Option`."""

    def __init__(self, struct: Any, sim: "MjWarpSim"):
        self.__dict__.update(_struct=struct, _sim=sim)

    def __getattr__(self, field: str) -> Any:
        """The field; a model field with a world axis gets a row per world on first access."""
        value, nworld = getattr(self._struct, field), self._sim.nworld
        if field in _BATCHED and value.shape[0] < nworld:  # copy the shared row(s) out to every world
            value = wp.array(value.numpy()[np.arange(nworld) % value.shape[0]], dtype=value.dtype, device=value.device)
            setattr(self, field, value)
        self.__dict__[field] = value
        return value

    def __setattr__(self, field: str, value: Any):
        """Replace the field and drop `sim`'s graphs, which hold the old array."""
        setattr(self._struct, field, value)
        self.__dict__[field] = value
        self.__dict__.pop("_torch", None)  # its torch views hold the old
        self._sim._graphs.clear()


class _DataFields(_Fields):
    """`_Fields` of a `mjw.Data` plus `substep_sensordata`; `torch`: zero-copy torch views, made on first access."""

    @property
    def torch(self) -> types.SimpleNamespace:
        if "_torch" not in self.__dict__:
            views = {}
            for field in dataclasses.fields(self._struct):
                value = getattr(self._struct, field.name)
                if isinstance(value, wp.array):
                    views[field.name] = wp.to_torch(value)
            views["substep_sensordata"] = wp.to_torch(self.substep_sensordata)
            self.__dict__["_torch"] = types.SimpleNamespace(**views)
        return self.__dict__["_torch"]


# m: a geom's or pair's margin in the worlds whose scene lacks it: it never collides there, but rays and renders see it.
_NO_CONTACT_MARGIN = -1e3


# --- simulator -------------------------------------------------------------
@wp.kernel
def _mask_kernel(mask: wp.array(dtype=Any), out: wp.array(dtype=int)):
    w = wp.tid()
    out[w] = 0
    if mask[w]:
        out[w] = 1


@wp.kernel
def _reset_kernel(mask: wp.array(dtype=int), value: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
    w, i = wp.tid()  # pyright: ignore[reportAssignmentType]
    if mask[w]:
        out[w, i] = value[w, i]


@wp.kernel
def _scene_kernel(
    mask: wp.array(dtype=int), scene: wp.array(dtype=int), rows: wp.array2d(dtype=Any), out: wp.array2d(dtype=Any)
):
    w, i = wp.tid()  # pyright: ignore[reportAssignmentType]
    if mask[w]:
        out[w, i] = rows[scene[w], i]


@wp.kernel
def _static_kernel(
    mask: wp.array(dtype=int),
    geoms: wp.array(dtype=int),
    geom_bodyid: wp.array(dtype=int),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
):
    # MuJoCo Warp places the geoms of static bodies once, in `put_data`; kinematics still computes the bodies' frames
    w, i = wp.tid()  # pyright: ignore[reportAssignmentType]
    g = geoms[i]
    b = geom_bodyid[g]
    if mask[w]:
        geom_xpos[w, g] = xpos[w, b] + mjmath.rot_vec_quat(geom_pos[w % geom_pos.shape[0], g], xquat[w, b])
        geom_xmat[w, g] = mjmath.quat_to_mat(mjmath.mul_quat(xquat[w, b], geom_quat[w % geom_quat.shape[0], g]))


@dataclasses.dataclass(frozen=True)
class CoarseConfig:
    """`MjWarpSim.build_coarse_spec`'s cheaper geometry and physics; a field at its default changes nothing.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> spec = mujoco.MjSpec.from_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
        >>> MjWarpSim.build_coarse_spec(spec, CoarseConfig()).to_xml() == spec.to_xml()
        True
        >>> MjWarpSim.build_coarse_spec(spec, CoarseConfig(timestep=0.01)).option.timestep
        0.01
    """

    max_hull_vertices: int = -1  # each mesh's convex hull's most vertices
    qpos_ref: Optional[np.ndarray] = None  # (T, nq): only the `<pair>`s that come within `reach` m along it remain
    reach: float = 0.02  # m
    collide: str = ""  # only the geoms whose name (else their body's) fully matches this regex collide
    condim: int = 0  # contacts of at most this many dimensions
    condim_geoms: str = ""  # `condim` only for the geoms whose name (else their body's) fully matches this regex
    timestep: float = 0.0  # s
    min_timeconst: float = 0.0  # contacts' and equalities' solref time constants at least this, s
    min_impedance_width: float = 0.0  # contacts' and equalities' solimp widths at least this, m
    pyramidal: bool = False  # pyramidal friction cones with impratio 1, as mjbatch's
    iterations: int = 0  # the most solver iterations
    ls_iterations: int = 0  # the most line search iterations, else `iterations`
    integrator: str = ""  # e.g. "implicitfast", stable at larger steps with joint damping


class MjWarpSim:
    """Step `nworld` worlds of `model`, or `nworld[i]` of each scene `model[i]`, `substeps` physics steps per `step()`.

    Args:
        model: A `mujoco.MjModel`, or `mujoco.MjSpec` scenes that may differ in geoms, meshes, inertias, sites, pairs.
        nworld: The worlds, or each scene's; each world gets its scene's values.
        data: The state `reset()` restores, else keyframe `key`'s, else the defaults; several scenes: their defaults.
        device: The Warp device; on a GPU each call runs as one cached CUDA graph.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import mujoco, numpy as np, warp as wp
        >>> from robokit.sim import MjWarpSim
        >>> xml = '<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
        >>> xml += '<body pos="0 0 .5"><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        >>> sim = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=8, substeps=4)
        >>> sim.model.geom_friction.shape  # one row per world, ready to randomize
        (8, 2)
        >>> sim.opt.iterations = 5  # from the next step on
        >>> sim.step()
        >>> bool((sim.data.qpos.numpy()[:, 2] < 0.5).all()), sim.data.substep_sensordata.shape
        (True, (4, 8, 0))
        >>> sim.reset(wp.array(np.arange(8) < 4))  # the first four worlds back to the start
        >>> z = sim.data.qpos.numpy()[:, 2]
        >>> float(z[0]), bool(z[7] < 0.5)
        (0.5, True)
        >>> big = mujoco.MjSpec.from_string(xml.replace('size=".1"', 'size=".2"'))
        >>> mixed = MjWarpSim([mujoco.MjSpec.from_string(xml), big], nworld=[3, 5])  # worlds 3 to 7: the big ball
        >>> (mixed.model.geom_size.numpy()[:, 1, 0] > 0.15).tolist()
        [False, False, False, True, True, True, True, True]
        >>> pos = mixed.model.geom_pos.numpy()
        >>> pos[0, 0, 2] = -1.0  # world 0's floor, a static geom, a meter lower ...
        >>> mixed.model.geom_pos.assign(pos)
        >>> mixed.reset(wp.array(np.arange(8) == 0))  # ... from its next reset
        >>> mixed.data.geom_xpos.numpy()[:2, 0, 2].tolist()
        [-1.0, 0.0]
    """

    def __init__(
        self,
        model: Union[mujoco.MjModel, Sequence[mujoco.MjSpec]],
        nworld: Union[int, Sequence[int]],
        data: Optional[mujoco.MjData] = None,
        substeps: int = 1,
        njmax: Optional[int] = None,
        nconmax: Optional[int] = None,
        key: Optional[str] = None,
        device=None,
    ):
        self._args = {k: v for k, v in locals().items() if k not in ("self", "nworld")}  # for `resized`
        mixed = not isinstance(model, mujoco.MjModel)
        if mixed:
            model, scenes, values = self._merge_scenes(model)
        else:
            scenes, values = [model], {}
        scene = np.repeat(np.arange(len(scenes)), nworld)  # each world's scene
        if data is None:
            data = mujoco.MjData(model)
            if key is not None:
                mujoco.mj_resetDataKeyframe(model, data, model.key(key).id)
            mujoco.mj_forward(model, data)
        initial = [mujoco.MjData(m) for m in scenes] if mixed else [data]  # each scene's initial state
        self.mjm, self.nworld, self.substeps, self.device = model, len(scene), substeps, wp.get_device(device)
        self.stream = wp.get_stream(self.device) if self.device.is_cuda else None
        self._graphs: Dict[str, wp.Graph] = {}
        with wp.ScopedDevice(self.device):
            self._m = mjw.put_model(model)
            if hasattr(mjw, "OverflowType"):  # newer MuJoCo Warp warns on each capped solve; scenes cap on purpose
                self._m.opt.warn_overflow &= ~int(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
            self._d = mjw.put_data(model, data, nworld=self.nworld, nconmax=nconmax, njmax=njmax)
            self.data, self.model, self.opt = (
                _DataFields(self._d, self),
                _Fields(self._m, self),
                _Fields(self._m.opt, self),  # a change takes effect from the next call
            )
            sensordata = wp.zeros((substeps, self.nworld, model.nsensordata), dtype=float)
            self.data.__dict__["substep_sensordata"] = sensordata  # not a `mjw.Data` field
            self._mask = wp.zeros(self.nworld, dtype=int)  # `reset`'s, at a fixed address its cached graph reads
            self._static = wp.array(np.flatnonzero(model.body_weldid[model.geom_bodyid] == 0), dtype=int)
            self._scene_rows: List[Tuple[wp.array, wp.array]] = []  # for `set_scene`: each world's field, every scene's
            for f, x in values.items():  # each world its scene's values where the scenes differ
                if not (x == x[:1]).all():
                    field = getattr(self.model, f)
                    shape = field.numpy().shape
                    field.assign(x[scene].reshape(shape))
                    rows = wp.array(x.reshape((len(scenes),) + shape[1:]), dtype=field.dtype)
                    self._scene_rows.append((field.reshape((self.nworld, -1)), rows.reshape((len(scenes), -1))))
            self._initial: List[Tuple[wp.array, wp.array]] = []  # for `reset`: each `_STATE` field, its initial rows
            for f in _STATE:
                value = np.stack([np.atleast_1d(getattr(d, f)) for d in initial])[scene]  # its starting scene's
                self._initial.append((getattr(self._d, f).reshape((self.nworld, -1)), wp.array(value, dtype=float)))
            if mixed:  # each world from its scene's initial state, its static geoms where its scene has them
                self.reset(wp.ones(self.nworld, dtype=int))
                mjw.forward(self._m, self._d)

    def resized(self, nworld: int) -> "MjWarpSim":
        """A new simulator of this one's scenes and settings with `nworld` worlds, split evenly over several scenes."""
        scenes = self._args["model"]
        one = isinstance(scenes, mujoco.MjModel)
        return MjWarpSim(nworld=nworld if one else [nworld // len(scenes)] * len(scenes), **self._args)

    @staticmethod
    def build_coarse_spec(spec: mujoco.MjSpec, config: CoarseConfig) -> mujoco.MjSpec:
        """A copy of `spec` with `config`'s cheaper geometry and physics, e.g. for planning or early RL training.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> spec = mujoco.MjSpec.from_string(
            ...     '<mujoco><worldbody><body name="d"><freejoint/><geom name="a" size="0.1"/><geom size="0.1"/></body>'
            ...     '<geom name="b" size="0.1" pos="1 0 0"/><geom name="c" size="0.1" pos="0 0 -0.21"/></worldbody>'
            ...     '<contact><pair geom1="a" geom2="b"/><pair geom1="a" geom2="c"/></contact></mujoco>'
            ... )
            >>> coarse = MjWarpSim.build_coarse_spec(spec, CoarseConfig(qpos_ref=np.array([[0, 0, 0, 1, 0, 0, 0]])))
            >>> coarse.compile().npair  # only a-c comes near
            1
            >>> config = CoarseConfig(collide="a|c|d", condim=1, condim_geoms="d", timestep=0.01, pyramidal=True)
            >>> model = MjWarpSim.build_coarse_spec(spec, config).compile()
            >>> model.geom_contype.tolist(), model.geom_condim.tolist()  # b, c, a, d's unnamed: b no longer collides
            ([0, 1, 1, 1], [3, 3, 3, 1])
            >>> model.opt.timestep, model.opt.cone
            (0.01, 0)
            >>> config = CoarseConfig(min_timeconst=0.01, min_impedance_width=0.002)
            >>> model = MjWarpSim.build_coarse_spec(spec, config).compile()
            >>> model.geom_solref[:, 0].tolist(), model.geom_solimp[:, 2].tolist()  # from 0.02 and 0.001 by default
            ([0.02, 0.02, 0.02, 0.02], [0.002, 0.002, 0.002, 0.002])
            >>> config = CoarseConfig(iterations=2, ls_iterations=4, integrator="implicitfast")
            >>> opt = MjWarpSim.build_coarse_spec(spec, config).option
            >>> opt.iterations, opt.ls_iterations, opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            (2, 4, True)
        """
        coarse = spec.copy()
        for mesh in coarse.meshes if config.max_hull_vertices > 0 else []:
            mesh.maxhullvert = config.max_hull_vertices
        if config.qpos_ref is not None:
            wide = coarse.copy()
            for pair in wide.pairs:
                pair.margin = config.reach
            model, near = wide.compile(), set()
            data = mujoco.MjData(model)
            for q in config.qpos_ref:
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_collision(model, data)
                near |= {tuple(sorted(map(int, c.geom))) for c in data.contact[: data.ncon]}
            for pair in list(coarse.pairs):
                if tuple(sorted((model.geom(pair.geomname1).id, model.geom(pair.geomname2).id))) not in near:
                    coarse.delete(pair)
        for geom in coarse.geoms:
            name = geom.name or geom.parent.name
            if config.collide and not re.fullmatch(config.collide, name):
                geom.contype = geom.conaffinity = 0
            elif config.condim and re.fullmatch(config.condim_geoms or ".*", name):
                geom.condim = min(geom.condim, config.condim)
        for pair in coarse.pairs if config.condim and not config.condim_geoms else []:
            pair.condim = min(pair.condim, config.condim)
        for element in [*coarse.geoms, *coarse.pairs, *coarse.equalities]:
            if element.solref[0] > 0:  # time constants, not direct stiffness and damping
                element.solref[0] = max(element.solref[0], config.min_timeconst)
            element.solimp[2] = max(element.solimp[2], config.min_impedance_width)
        coarse.option.timestep = config.timestep or coarse.option.timestep
        if config.pyramidal:
            coarse.option.cone, coarse.option.impratio = mujoco.mjtCone.mjCONE_PYRAMIDAL, 1.0
        iterations = config.iterations or coarse.option.iterations
        coarse.option.iterations = min(iterations, coarse.option.iterations)
        ls_iterations = config.ls_iterations or config.iterations or coarse.option.ls_iterations
        coarse.option.ls_iterations = min(ls_iterations, coarse.option.ls_iterations)
        if config.integrator:
            coarse.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{config.integrator.upper()}")
        return coarse

    def set_const(self):
        """Recompute constants derived from masses and inertias, per world, after randomizing them or `body_ipos`."""
        for field in _DERIVED:
            getattr(self.model, field)
        with wp.ScopedDevice(self.device):
            mjw.set_const(self._m, self._d)

    def build_entity(self, prefix: str, soft_joint_pos_limit_factor: float = 1.0) -> "Entity":
        """The `robokit.sim.mjentity.Entity` named `prefix*`, its default pose the initial `qpos`.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> xml = '<mujoco><worldbody><body name="r/base" pos="0 0 .5"><freejoint name="r/root"/><geom size=".1"/>'
            >>> xml += '<body><joint name="r/a"/><geom size=".1"/></body></body></worldbody></mujoco>'
            >>> robot = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2).build_entity("r/")
            >>> robot.joint_names, robot.data.default_root_state[:3].tolist()
            (['a'], [0.0, 0.0, 0.5])
        """
        from robokit.sim.mjentity import Entity

        default_qpos = self._initial[0][1].numpy()[0]  # `qpos`, world 0's
        return Entity(self, prefix, default_qpos, soft_joint_pos_limit_factor)

    def build_contact_sensor(self, name: str, track_air_time: bool = False) -> "ContactSensor":
        """A `robokit.sim.mjsensor.ContactSensor` of the contact sensors named `{name}_{primary}_{found or force}`.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> xml = '<mujoco><worldbody><body><freejoint/><geom name="ball" size=".1"/></body></worldbody><sensor>'
            >>> xml += '<contact name="touch_ball_found" geom1="ball" data="found"/></sensor></mujoco>'
            >>> MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2).build_contact_sensor("touch").primary_names
            ['ball']
        """
        from robokit.sim.mjsensor import ContactSensor

        return ContactSensor(self, name, track_air_time)

    def get_sensor(self, name: str) -> Any:
        """(nworld, dim): sensor `name`'s readings, a zero-copy torch view of `data.sensordata`.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> xml = '<mujoco><worldbody><body name="b" pos="0 0 .5"><freejoint/><geom size=".1"/></body></worldbody>'
            >>> xml += '<sensor><framepos name="p" objtype="body" objname="b"/></sensor></mujoco>'
            >>> MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2).get_sensor("p").tolist()
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]
        """
        sensor = self.mjm.sensor(name)
        adr = int(self.mjm.sensor_adr[sensor.id])
        return self.data.torch.sensordata[:, adr : adr + int(self.mjm.sensor_dim[sensor.id])]

    def compute_rays(self, render_context: Any, **rays: Any):
        """Refit `render_context`'s BVH to the current poses, then cast `rays` (`mjw.rays` arguments) in every world.

        Example:
            >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
            >>> mjm = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" pos="0 0 -.5" size="1 1 .5"/>'
            ...     '</worldbody></mujoco>')
            >>> rc = mjw.create_render_context(mjm, 2, cam_res=[], render_rgb=[], render_depth=[], render_seg=[])
            >>> dist, down = wp.zeros((2, 1), dtype=float), wp.array([[[0, 0, -1]]] * 2, dtype=wp.vec3)  # from z = 1 m
            >>> MjWarpSim(mjm, nworld=2).compute_rays(rc, pnt=wp.array([[[0, 0, 1]]] * 2, dtype=wp.vec3), vec=down,
            ...     geomgroup=wp.types.vector(6, wp.float32)(-1, -1, -1, -1, -1, -1), flg_static=True,
            ...     bodyexclude=wp.array([-1], dtype=int), dist=dist, geomid=wp.zeros((2, 1), dtype=int),
            ...     normal=wp.zeros((2, 1), dtype=wp.vec3))
            >>> dist.numpy().tolist()
            [[1.0], [1.0]]
        """
        with wp.ScopedDevice(self.device):
            mjw.refit_bvh(self._m, self._d, render_context)
            mjw.rays(self._m, self._d, rc=render_context, **rays)

    def set_torch_stream(self):
        """Run torch on Warp's `stream`, so torch and the simulator stay ordered without host syncs.

        Example:
            >>> # xdoctest: +REQUIRES(env:ROBOKIT_CUDA_DOCS)
            >>> import torch
            >>> xml = '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
            >>> (sim := MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=2)).set_torch_stream()
            >>> torch.cuda.current_stream().cuda_stream == sim.stream.cuda_stream
            True
        """
        import torch

        torch.cuda.set_stream(wp.stream_to_torch(self.stream))

    def set_random(
        self,
        field: str,
        index: Tuple[Any, ...],
        low: Union[float, Sequence[float]],
        high: Union[float, Sequence[float]],
        log: bool = False,
        add: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        """Set (`add`: offset) `model.<field>[:, *index]` to one uniform draw per world from `low` to `high`.

        Args:
            field: After `body_mass`, `body_inertia`, `body_ipos` or `body_iquat`, it runs `set_const()`.
            low: Lower bound, broadcast to the trailing axes.
            high: Upper bound, broadcast to the trailing axes.
            log: Draw log-uniformly.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> xml = '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
            >>> sim = MjWarpSim(mujoco.MjModel.from_xml_string(xml), nworld=8)
            >>> sim.set_random("geom_friction", (0, 0), 0.5, 1.0)  # sliding friction
            >>> friction = sim.model.geom_friction.numpy()[:, 0, 0]
            >>> bool(((friction >= 0.5) & (friction <= 1.0)).all()), len(set(friction.tolist()))
            (True, 8)
            >>> sim.set_random("body_ipos", (1,), (-0.01, -0.01, 0.0), (0.01, 0.01, 0.0), add=True)  # center of mass
            >>> float(abs(sim.model.body_ipos.numpy()[:, 1, 2]).max())
            0.0
        """
        rng = rng if rng is not None else np.random.default_rng()
        array = getattr(self.model, field)
        values = array.numpy()
        lo = np.asarray(low, dtype=float)
        hi = np.asarray(high, dtype=float)
        target = values[(slice(None), *index)]
        shape = (self.nworld,) + (1,) * (target.ndim - 1 - lo.ndim) + lo.shape
        u = rng.random(shape)
        draw = lo * (hi / lo) ** u if log else lo + (hi - lo) * u
        values[(slice(None), *index)] = target + draw if add else draw
        array.assign(values)
        if field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            self.set_const()

    def step(self):
        """Advance `substeps` physics steps into `substep_sensordata`; derived fields then lag `qpos` by a substep."""

        def body():
            for k in range(self.substeps):
                mjw.step(self._m, self._d)
                wp.copy(self.data.substep_sensordata, self._d.sensordata, dest_offset=k * self._d.sensordata.size)

        self._launch("step", body)

    def kinematics(self):
        """Recompute `xpos`, `xmat`, `site_xpos`, ... from `qpos`, e.g. after writing it, as `mj_kinematics`."""
        self._launch("kinematics", lambda: mjw.kinematics(self._m, self._d))

    def com_pos(self):
        """Recompute `subtree_com` (subtree centers of mass), `cinert` and `cdof` from the positions, as `mj_comPos`."""
        self._launch("com_pos", lambda: mjw.com_pos(self._m, self._d))

    def forward(self):
        """Recompute every derived quantity, accelerations and `sensordata` included, as `mj_forward`."""
        self._launch("forward", lambda: mjw.forward(self._m, self._d))

    def reset(self, mask: wp.array):
        """Return the worlds where `mask` is set to their initial state, static geoms to their current rows.

        Args:
            mask: (nworld,), bool or int; read on the device, with no host syncs.
        """

        def body():
            m, d = self._m, self._d
            for out, value in self._initial:
                wp.launch(_reset_kernel, dim=out.shape, inputs=[self._mask, value], outputs=[out])
            mjw.kinematics(m, d)
            # static geoms from the current `body_pos`, `body_quat`, `geom_pos`, `geom_quat`
            inputs = [self._mask, self._static, m.geom_bodyid, m.geom_pos, m.geom_quat, d.xpos, d.xquat]
            wp.launch(_static_kernel, (self.nworld, len(self._static)), inputs, [d.geom_xpos, d.geom_xmat])

        wp.launch(_mask_kernel, dim=self.nworld, inputs=[mask], outputs=[self._mask], device=self.device)
        self._launch("reset", body)

    def set_scene(self, mask: wp.array, scene: wp.array):
        """Give the worlds where `mask` is set scene `scene[w]`'s model rows; their static geoms move at `reset()`.

        Args:
            mask: (nworld,), read on the device, with no host syncs.
            scene: (nworld,), int.

        Example:
            >>> # xdoctest: +REQUIRES(module:mujoco_warp)
            >>> xml = '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
            >>> scenes = [mujoco.MjSpec.from_string(xml), mujoco.MjSpec.from_string(xml.replace(".1", ".2"))]
            >>> sim = MjWarpSim(scenes, nworld=[2, 2])
            >>> sim.set_scene(wp.array([True, False, False, False]), wp.array([1, 0, 0, 0], dtype=int))
            >>> (sim.model.geom_size.numpy()[:, 0, 0] > 0.15).tolist()  # world 0: the big ball
            [True, False, True, True]
        """
        wp.launch(_mask_kernel, dim=self.nworld, inputs=[mask], outputs=[self._mask], device=self.device)
        for out, rows in self._scene_rows:
            wp.launch(_scene_kernel, dim=out.shape, inputs=[self._mask, scene, rows], outputs=[out], device=self.device)

    def _launch(self, name: Optional[str], body: Callable[[], None]):
        """Run `body` into an enclosing capture, directly on the CPU or without a `name`, else as a cached graph."""
        with wp.ScopedDevice(self.device):
            stream = wp.get_stream() if self.device.is_cuda else None
            if stream is not None and stream.is_capturing:  # inside an enclosing capture: record the kernels into it
                external = stream not in self.device.captures  # not started by Warp, e.g. torch's
                conditional = self._m.opt.graph_conditional
                # In a capture Warp did not start, the solver runs all iterations: an early stop would pause it.
                self._m.opt.graph_conditional = conditional and not external
                if external:  # tell Warp about it, as its own launches and allocations need
                    wp.capture_begin(stream=stream, external=True, capture_mode=wp.CaptureMode.GLOBAL)
                body()
                if external:
                    wp.capture_end(stream=stream)
                self._m.opt.graph_conditional = conditional
            elif stream is None or name is None:
                body()
            else:
                if name not in self._graphs:
                    wp.synchronize_device()
                    gc.collect()  # as torch.cuda.graph: a collection during the capture may free CUDA modules
                    gc.disable()
                    try:
                        with wp.ScopedCapture() as capture:
                            body()
                    finally:
                        gc.enable()
                    self._graphs[name] = capture.graph
                wp.capture_launch(self._graphs[name])

    # --- mixed scenes ---------------------------------------------------------
    @staticmethod
    def _merge_scenes(
        specs: Sequence[mujoco.MjSpec],
    ) -> Tuple[mujoco.MjModel, List[mujoco.MjModel], Dict[str, np.ndarray]]:
        """Merge `specs` into the one with the most geoms; others' geoms match its own by name (unnamed: by place).

        Returns:
            One model of every scene's geoms, meshes, pairs; the compiled scenes; each scene's values `(nscene, ...)`.
        """
        scenes = [s.compile() for s in specs]
        t = int(np.argmax([m.ngeom for m in scenes]))  # the template
        T = scenes[t]
        sizes = ("nq", "nv", "nu", "na", "nbody", "njnt", "nsite", "nsensordata")
        for m in scenes:
            if [getattr(m, k) for k in sizes] != [getattr(T, k) for k in sizes] or (
                m.body_geomnum > T.body_geomnum
            ).any():
                raise ValueError(f"scenes must match in {', '.join(sizes)}, and one have the most geoms on every body")

        # A scene's mesh the template lacks joins its spec as `<name>__<scene>`, with a geom so MuJoCo builds the hull.
        spec = specs[t].copy()
        used = np.bitwise_or.reduce(np.r_[T.geom_contype, T.geom_conaffinity])
        unused = int(~used & (used + 1))  # a collision bit no geom uses: those geoms collide with nothing
        # MjSpec copies no element across specs, so these attributes carry a mesh over (its material may not exist).
        copied = "content_type scale refpos refquat inertia maxhullvert smoothnormal uservert userface".split()
        meshes = [[m.mesh(k).name for k in range(m.nmesh)] for m in scenes]  # each scene's meshes' names in the model
        verts = [[m.mesh_vert[a : a + n] for a, n in zip(m.mesh_vertadr, m.mesh_vertnum)] for m in scenes]
        for i, s in enumerate(specs):
            for k, name in enumerate(meshes[i]):
                if i == t or name in meshes[t] and np.array_equal(verts[i][k], verts[t][meshes[t].index(name)]):
                    continue
                mesh = s.mesh(name)
                meshes[i][k] = f"{name}__{i}"
                file = str(Path(s.modelfiledir, s.meshdir, mesh.file).resolve()) if mesh.file else ""
                spec.add_mesh(name=meshes[i][k], file=file, **{f: getattr(mesh, f) for f in copied})
                spec.worldbody.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH, meshname=meshes[i][k], contype=unused, conaffinity=0, group=5
                )
        model = spec.compile()

        # Each scene's values: the per-world geom, pair, body, site fields but `geom_matid` (materials do not join).
        fields = [
            f for f in sorted(_BATCHED) if f.split("_")[0] in ("geom", "pair", "body", "site") and f != "geom_matid"
        ]
        values = {f: np.repeat(getattr(model, f)[None], len(scenes), 0) for f in fields + ["dof_invweight0", "qpos0"]}
        geoms = {model.geom(k).name: k for k in range(model.ngeom)}
        pairs = {(g1, g2): p for p, (g1, g2) in enumerate(zip(model.pair_geom1, model.pair_geom2))}
        for i, m in enumerate(
            scenes
        ):  # `geom_` and `pair_` values at its geoms' (by name, else place) and pairs' places
            place = model.body_geomadr[m.geom_bodyid] + np.arange(m.ngeom) - m.body_geomadr[m.geom_bodyid]
            g = np.array([geoms[m.geom(k).name] if m.geom(k).name else place[k] for k in range(m.ngeom)], int)
            p = np.array([pairs[g[a], g[b]] for a, b in zip(m.pair_geom1, m.pair_geom2)], int)
            for f in values:
                index = g if f.startswith("geom_") else p if f.startswith("pair_") else slice(None)
                values[f][i][index] = getattr(m, f)
            mesh = m.geom_type == mujoco.mjtGeom.mjGEOM_MESH
            values["geom_dataid"][i][g[mesh]] = [model.mesh(meshes[i][k]).id for k in m.geom_dataid[mesh]]
            lacks = np.setdiff1d(np.arange(model.ngeom), g)
            values["geom_margin"][i][lacks] = _NO_CONTACT_MARGIN
            values["geom_gap"][i][lacks] = 0
            lacks = np.setdiff1d(np.arange(model.npair), p)
            values["pair_margin"][i][lacks] = _NO_CONTACT_MARGIN
            values["pair_gap"][i][lacks] = 0
        return model, scenes, values
