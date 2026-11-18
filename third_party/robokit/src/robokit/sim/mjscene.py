"""MuJoCo scenes from robokit robot files: robots at given transforms, a light, a floor or a curriculum terrain."""

from pathlib import Path
from typing import Sequence, Tuple, Union

import mujoco  # pyright: ignore[reportMissingImports]
import numpy as np


def build_robot_spec(path: Union[str, Path], prefix: str = "", body: str = "") -> mujoco.MjSpec:
    """Load an MJCF or URDF robot into a spec with prefixed names; `body` keeps only that subtree, e.g. one ALOHA arm.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "arm.xml"
        >>> _ = path.write_text('<mujoco><worldbody><body><joint name="joint1"/><geom size=".1"/></body></worldbody></mujoco>')
        >>> build_robot_spec(path, prefix="left/").compile().joint(0).name
        'left/joint1'
    """
    source = mujoco.MjSpec.from_file(str(path))
    robot = mujoco.MjSpec()
    source.option = robot.option
    root = source.body(body) if body else source.worldbody.first_body()
    robot.worldbody.add_frame().attach_body(root, prefix, "")
    return robot


def build_scene_spec(
    robots: Sequence[Tuple[mujoco.MjSpec, Sequence[float]]] = (), floor: bool = True, timestep: float = 0.005
) -> mujoco.MjSpec:
    """Build a scene spec: implicitfast, a light, a floor if `floor`, each `(robot, T_world_base)` at its 7D transform.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "arm.xml"
        >>> _ = path.write_text('<mujoco><worldbody><body><joint name="joint1"/><geom size=".1"/></body></worldbody></mujoco>')
        >>> model = build_scene_spec([(build_robot_spec(path), (0, 0, 0, 1, 0, 0, 0))]).compile()
        >>> model.joint(0).name, model.geom("floor").type[0] == mujoco.mjtGeom.mjGEOM_PLANE
        ('joint1', True)
    """
    spec = mujoco.MjSpec()
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    if floor:
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    for robot, T_world_base in robots:
        robot.option = spec.option
        frame = spec.worldbody.add_frame(pos=T_world_base[:3], quat=T_world_base[3:])
        frame.attach_body(robot.worldbody.first_body(), "", "")
    return spec


CONTACT_DATA = dict(found=1, force=2, torque=4, dist=8, pos=16, normal=32, tangent=64)  # contact sensor fields
CONTACT_REDUCE = dict(none=0, mindist=1, maxforce=2, netforce=3)  # how a contact sensor combines its matches


def build_contact_sensor(
    spec: mujoco.MjSpec,
    name: str,
    obj: Tuple[int, str],
    ref: Tuple[int, str] = (mujoco.mjtObj.mjOBJ_UNKNOWN, ""),
    data: Sequence[str] = ("found",),
    reduce: str = "none",
    num: int = 1,
) -> mujoco.MjsSensor:
    """Add a contact sensor `name` on `obj` against `ref`, reporting `data` for `num` contacts combined by `reduce`.

    Args:
        obj: `(mujoco.mjtObj, name)`.
        ref: `(mujoco.mjtObj, name)`; without it, contacts with anything count.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> spec = build_scene_spec()
        >>> ball = spec.worldbody.add_body(name="ball", pos=[0, 0, 0.1])
        >>> _ = ball.add_geom(name="ball", size=[0.1, 0, 0])
        >>> _ = build_contact_sensor(spec, "touch", (mujoco.mjtObj.mjOBJ_GEOM, "ball"), data=("found", "force"))
        >>> int(spec.compile().sensor_dim[0])
        4
    """
    return spec.add_sensor(
        name=name,
        type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=obj[0],
        objname=obj[1],
        reftype=ref[0],
        refname=ref[1],
        intprm=[sum(CONTACT_DATA[field] for field in data), CONTACT_REDUCE[reduce], num],
    )


def _box(body: mujoco.MjsBody, pos: Sequence[float], dims: Sequence[float]):
    """Add a box geom at `pos` with side lengths `dims`, each at least 2e-6 m."""
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=np.maximum(1e-6, np.array(dims) / 2))


def _ring(body: mujoco.MjsBody, center: Sequence[float], size: Sequence[float], hole: Sequence[float], height: float):
    """Add four boxes `height` high framing a `hole` (x, y) at `center` within a `size` (x, y) square."""
    x, y, z = center
    tx = (size[0] - hole[0]) / 2
    ty = (size[1] - hole[1]) / 2
    _box(body, (x, y + (hole[1] + ty) / 2, z), (size[0], ty, height))
    _box(body, (x, y - (hole[1] + ty) / 2, z), (size[0], ty, height))
    _box(body, (x - (hole[0] + tx) / 2, y, z), (tx, hole[1], height))
    _box(body, (x + (hole[0] + tx) / 2, y, z), (tx, hole[1], height))


def build_terrain(
    spec: mujoco.MjSpec,
    kinds: Sequence[str] = ("flat", "stairs", "stairs_inv", "slope", "slope_inv", "random_rough", "wave"),
    rows: int = 10,
    tile: float = 8.0,
    border: float = 20.0,
    seed: int = 0,
) -> np.ndarray:
    """Add a body "terrain" of `tile` m square tiles; return the tiles' spawn origins (rows, len(kinds), 3).

    Args:
        kinds: A column of tiles per kind.
        rows: Rows of tiles, of difficulty 0 to 1.
        border: Width of the flat ring around the tiles, m.

    Example:
        >>> # xdoctest: +REQUIRES(module:mujoco_warp)
        >>> spec = build_scene_spec(floor=False)
        >>> origins = build_terrain(spec, kinds=("flat", "stairs", "wave"), rows=2)
        >>> origins.shape, spec.compile().nhfield
        ((2, 3, 3), 2)
    """
    body = spec.worldbody.add_body(name="terrain")
    rng = np.random.default_rng(seed)  # random_rough heights
    origins = np.zeros((rows, len(kinds), 3))
    for col, kind in enumerate(kinds):
        for row in range(rows):
            difficulty = row / max(rows - 1, 1)
            cx = -rows * tile / 2 + row * tile + tile / 2
            cy = -len(kinds) * tile / 2 + col * tile + tile / 2
            if kind == "flat":
                _box(body, (cx, cy, -0.5), (tile, tile, 1.0))
                spawn_z = 0.0
            elif kind.startswith("stairs"):  # steps 0 to 0.1 m high, 0.3 m deep, to a 3 m platform; a 1 m flat edge
                rise = 0.1 * difficulty
                run = 0.3
                inner = tile - 2 * 1.0
                steps = max(0, int((inner - 3.0) / (2 * run)))
                top = (steps + 1) * rise
                edge_height = max(rise, 0.05)
                _ring(body, (cx, cy, -edge_height / 2), (tile, tile), (inner, inner), edge_height)
                for k in range(steps):  # step k: a square ring, its top at (k + 1) * rise up or down
                    if kind == "stairs_inv":
                        step_z = -top / 2 - (k + 1) * rise / 2
                        height = top - (k + 1) * rise
                    else:
                        step_z = k * rise / 2
                        height = (k + 2) * rise
                    size = inner - 2 * k * run
                    offset = inner / 2 - (k + 0.5) * run
                    _box(body, (cx, cy + offset, step_z), (size, run, height))
                    _box(body, (cx, cy - offset, step_z), (size, run, height))
                    _box(body, (cx + offset, cy, step_z), (run, size - 2 * run, height))
                    _box(body, (cx - offset, cy, step_z), (run, size - 2 * run, height))
                rest = inner - 2 * steps * run
                if kind == "stairs_inv":
                    _box(body, (cx, cy, -top - rise / 2), (rest, rest, rise))
                    spawn_z = -top
                else:
                    _box(body, (cx, cy, steps * rise / 2), (rest, rest, top + rise))
                    spawn_z = top
            else:  # a heightfield: 0.1 m cells, 0.005 m height units, its outer 2 cells flat
                cells = int(tile / 0.1)
                n = cells - 4  # the cells inside the flat edge
                width = n * 0.1
                unit = 0.005
                i = np.arange(n)
                if kind.startswith("slope"):  # a pyramid of slope 0 to 1, up or down to a 2 m platform
                    slope = -difficulty if kind == "slope_inv" else difficulty
                    peak = int(slope * width / 2 / unit)
                    center = n // 2
                    ramp = (center - np.abs(center - i)) / center
                    raw = peak * ramp[:, None] * ramp[None, :]
                    plateau = raw[center - 10, center - 10]  # the platform's edge, 1 m from the center
                    raw = np.clip(raw, min(0, plateau), max(0, plateau))
                elif kind == "random_rough":  # 0.02 to 0.1 m in 0.02 m steps, a random height per cell
                    raw = rng.choice(np.arange(4, 21, 4), (n, n))
                else:  # wave: 4 waves along x plus 4 along y, 0 to 0.2 m high
                    wavenumber = 2 * np.pi / (n / 4)
                    amplitude = int(0.1 * difficulty / unit)
                    raw = amplitude * (np.cos(i[None, :] * wavenumber) + np.sin(i[:, None] * wavenumber))
                heights = np.zeros((cells, cells), dtype=np.int16)
                heights[2:-2, 2:-2] = np.rint(raw)
                heights = heights.T
                lo = heights.min()
                span = max(heights.max() - lo, 1)
                top = span * unit
                base = top / 4 if kind == "wave" else top  # the depth below the lowest cell
                field = spec.add_hfield(
                    name=f"hfield_{row}_{col}",
                    size=[tile / 2, tile / 2, top, base],
                    nrow=cells,
                    ncol=cells,
                    userdata=((heights - lo) / span).flatten().astype(np.float32).tolist(),
                )
                geom_z = {"slope_inv": -top, "wave": -top / 2}.get(kind, 0.0)  # the flat edge at about z 0
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name, pos=(cx, cy, geom_z))
                spawn_z = {"slope": top, "slope_inv": -top, "random_rough": 0.06, "wave": 0.0}[kind]
            origins[row, col] = cx, cy, spawn_z
    grid = (rows * tile, len(kinds) * tile)
    _ring(body, (0.0, 0.0, -0.5), (grid[0] + 2 * border, grid[1] + 2 * border), grid, 1.0)
    for index, geom in enumerate(body.geoms):
        geom.name = f"terrain_{index}"
    return origins
