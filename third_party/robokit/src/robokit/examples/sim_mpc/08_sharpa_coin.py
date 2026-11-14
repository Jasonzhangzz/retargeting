# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a palm-down Sharpa Wave hand walks a 2-euro coin across the backs of its fingers, flipping it at each one. Each seed starts from reset noise and passes with one round trip (index to pinky and back, every crossing a flip of at least 150 deg), no drop before it, the coin never 2 mm into the hand, jitter at most 5 rad/s^2 and no two hand shapes overlapping by over 1 mm. Seed 10000 plans 12 s in 93 s on an RTX 4090."""

import time

import mujoco
import numpy as np
import tyro
import viser
import warp as wp

from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import ControlTask, SensorTask
from robokit.utils.visualize_utils import ViserMjcf


STEPS = 2400  # 12 s at the 5 ms step
FLIP, LEAD, DWELL = 0.6, 0.5, 0.1  # s per flip; the plan's phase leads the coin's by at most LEAD; s on a finger
ORDER = [0, 1, 2, 3, 2, 1]  # goal k rests the coin on finger ORDER[k % 6]: index (0) to pinky (3) and back
GRASP = np.array([0.6, 0.0, 0.3, 0.0, 0.3, 0.35, -0.1, 0.3, 0.2, 0.35, -0.04, 0.3, 0.2, 0.35, 0.04, 0.3, 0.2, 0.0, 0.35,
                  0.1, 0.3, 0.2])  # fingers slightly curled and spread, 1.5-3 mm apart  # fmt: skip
# a flip against its phase s: lateral progress to the next finger, lift (m) and roll (rad), from earlier plans' flips
KEYS = np.array([
    [0.0, 0.15, 0.35, 0.55, 0.8, 1.0],  # s
    [0.0, 0.1, 0.25, 0.55, 0.9, 1.0],  # progress
    [0.0, 0.0, 0.002, 0.003, 0.002, 0.0],  # lift
    [0.0, 0.43, 0.85, 1.45, 2.3, np.pi],  # roll
])  # fmt: skip


def main(seeds: int = 1, seed0: int = 0):
    """Plan `seeds` seeds from `seed0` at once, score each, and replay the first."""
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()
    joints = np.array([[5, 6], [9, 10], [13, 14], [18, 19]])  # each finger's flexion and side swing
    half, skin = 0.0011, 0.001  # the coin's half thickness; contact starts 1 mm out, so it rests that far off

    # --- scene: the hand palm down 0.5 m high (fingers along +x, the index at +y), friction 1, stiff against itself; the coin flat on the back of the index's proximal phalanx ---
    scene = mujoco.MjSpec.from_file(str(sharpa_hand.MJCF_PATH_RIGHT))
    for mesh in scene.meshes:  # absolute mesh paths, so the viewer's copy finds them
        mesh.file = str(sharpa_hand.MJCF_PATH_RIGHT.parent / scene.meshdir / mesh.file)
    scene.meshdir, scene.option.timestep = "", 0.005
    root = scene.body("right_hand_C_MC")
    root.pos, root.quat = [0, 0, 0.5], [np.sqrt(0.5), 0, np.sqrt(0.5), 0]
    for geom in scene.geoms:
        if geom.contype or geom.conaffinity:
            geom.friction, geom.solmix = [1.0, 0.005, 1e-4], 0.0
            geom.solref, geom.solimp[:3] = [0.01, 1.0], [0.99, 0.999, 0.001]
    coin = scene.worldbody.add_body(name="coin")  # a 2-euro coin: 25.75 mm across, 2.2 mm thick, 8.5 g
    coin.add_freejoint()
    coin.add_geom(
        name="coin", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012875, half, 0], solref=[0.01, 1.0],
        solimp=[0.99, 0.999, 0.001, 0.5, 2], margin=skin, mass=0.0085, friction=[1.0, 0.005, 1e-4], condim=4,
        rgba=[0.85, 0.68, 0.2, 1],
    )  # fmt: skip
    body = mujoco.mjtObj.mjOBJ_XBODY
    scene.add_sensor(name="pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=body, objname="coin")
    scene.add_sensor(name="normal", type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS, objtype=body, objname="coin")
    for j in scene.joints[:22]:  # the hand joints' speeds, for a smoothing cost
        scene.add_sensor(
            name=j.name, type=mujoco.mjtSensor.mjSENS_JOINTVEL, objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=j.name
        )
    model = scene.compile()
    plan = scene.copy()  # planned on the capsules and the coin, with a capped solver; executed on every shape
    plan.option.iterations, plan.option.ls_iterations = 8, 8
    for geom in plan.geoms:
        if geom.type != mujoco.mjtGeom.mjGEOM_CAPSULE and geom.name != "coin":
            geom.contype = geom.conaffinity = 0
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    coin_geom, names = model.geom("coin").id, [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
    owner = np.array([next((i for i, f in enumerate(fingers) if f"_{f}_" in n), 5) for n in names])  # 5: the palm

    # --- starts: 0.03 rad on the joints, 2 mm and 8 deg of tilt on the coin, a 1 s settle; kept if flat, fingers apart
    probe, index = mujoco.MjData(model), model.geom("right_index_PP_fit").id
    probe.qpos[:22] = GRASP
    mujoco.mj_forward(model, probe)
    start = probe.geom_xpos[index] + [0, -0.003, model.geom_size[index, 0] + half + skin + 0.002]
    B, states = seeds, []
    for b in range(B):
        rng = np.random.default_rng(seed0 + b)
        while len(states) == b:
            state, q = mujoco.MjData(model), GRASP + rng.uniform(-0.03, 0.03, 22)
            tilt = rng.uniform(-np.radians(8), np.radians(8))
            state.qpos[:22], state.ctrl[:] = q, q
            state.qpos[22:25] = start + [*rng.uniform(-0.002, 0.002, 2), 0]
            state.qpos[25:29] = [np.cos(tilt / 2), np.sin(tilt / 2), 0, 0]
            for _ in range(200):
                mujoco.mj_step(model, state)
            g = owner[state.contact.geom[: state.ncon]]
            kiss = np.any((g[:, 0] != g[:, 1]) & np.isin(g, [1, 2, 3, 4]).all(1))  # two fingers touch
            flat = state.xmat[model.body("coin").id][8] > np.cos(np.radians(30))
            if flat and not kiss and np.linalg.norm(state.qvel[22:25]) < 0.02:
                states.append(state)

    # --- each finger's centreline and the coin's resting height on it, at the coin's x ---
    pp = [model.geom(f"right_{f}_PP_fit").id for f in fingers[1:]]
    x0, lines = np.array([s.qpos[22] for s in states]), np.array([s.geom_xpos[pp, 1] for s in states])
    axis = np.array([s.geom_xmat[pp].reshape(-1, 3, 3)[:, :, 2] for s in states])  # the capsules' axes
    centre = np.array([s.geom_xpos[pp] for s in states])
    tops = centre[..., 2] + (x0[:, None] - centre[..., 0]) * axis[..., 2] / axis[..., 0] + model.geom_size[pp, 0]
    tops, slope = tops + half + skin, np.arctan2(-axis[..., 2], axis[..., 0]).mean(1)
    lanes = lines + [-0.004, 0, 0, 0.004]  # the coin rests 4 mm inside the index and pinky, else it falls off

    def flip_ref(k: np.ndarray, u: np.ndarray) -> tuple:
        """The coin's centre and face normal in crossing `k` (goal k to k + 1) at phase `u` of the flip shape."""
        a, b = np.take(ORDER, k % 6)[:, None], np.take(ORDER, (k + 1) % 6)[:, None]
        go, lift, roll = (np.interp(np.clip(u, 0, 1), KEYS[0], KEYS[i]) for i in (1, 2, 3))
        y = lanes[rows, a] + (lanes[rows, b] - lanes[rows, a]) * go
        z = tops[rows, a] + (tops[rows, b] - tops[rows, a]) * go + lift
        roll, sl = np.pi * a + np.sign(b - a) * roll, slope[:, None]  # the face normal tips the way the coin goes
        normal = np.stack([np.cos(roll) * np.sin(sl), -np.sin(roll), np.cos(roll) * np.cos(sl)], -1)
        return np.stack(np.broadcast_arrays(x0[:, None], y, z), -1), normal

    # --- SimMPC: track the coin through the flip to the next goal, the phase at most LEAD ahead of the coin's; the fingers get a feed-forward (the finger the coin leaves rises) and their side swing stays within 0.06 rad; the plan looks 0.88 s ahead, over a whole flip; three iterations a plan with shrinking noise cut stalls, all at the sim's step (coarser steps lose the coin) ---
    H, pos, nrm, dt = 176, model.sensor("pos").adr[0], model.sensor("normal").adr[0], model.opt.timestep
    rows = np.arange(B)[:, None]
    ref = np.zeros((B, STEPS + H + 1, model.nsensordata), np.float32)
    sensor_ref = wp.array(ref, dtype=float)
    ctrl_ref = np.tile(GRASP.astype(np.float32), (B, ref.shape[1], 1))
    ctrl_wp = wp.array(ctrl_ref, dtype=float)
    weight = np.full(model.nu, 0.2)
    weight[joints[:, 1]] = 20.0
    ctrl_std = np.full(model.nu, 0.12)
    ctrl_std[model.actuator_gainprm[:, 0] < 2] *= 2  # the weak distal joints need twice the noise to move a fingertip
    ctrl_range = model.actuator_ctrlrange.copy()
    side = [*joints[:, 1], 17]  # the side swings and the pinky's cupping stay within 0.06 rad, so fingers rarely touch
    ctrl_range[side] = GRASP[side, None] + [-0.06, 0.06]
    config = SimMPCConfig(
        ctrl_std=ctrl_std, ctrl_range=ctrl_range, batch_size=B, num_samples=128, horizon_steps=H, node_steps=50,
        ctrl_steps=12, stages=[SimMPCStage(3)], std_growth=1.0, noise_decay=0.6, interpolation="cubic", terminal_scale=1.0,
        seed=seed0,
    )  # fmt: skip
    speeds = {j.name: 0.1 for j in scene.joints[:22]}  # a small cost on joint speed keeps the hand smooth
    config.add(SensorTask(sensor_ref, dict(pos=20000.0, normal=20.0) | speeds)).add(ControlTask(weight, ctrl_wp))
    sim = MjWarpSim(model, B, states[0], nconmax=64, njmax=400)
    for f in ("time", "qpos", "qvel", "act", "qacc_warmstart"):  # each seed from its start
        getattr(sim.data, f).assign(np.stack([getattr(s, f) for s in states]))
    clock, goal, s, s_coin, mpc = 0, np.zeros(B, int), np.zeros(B), np.zeros(B), None

    def lead(plan: list):
        nonlocal clock
        if plan:  # the goal is reached when its finger pushes on the coin, within 6 mm of its line, 45 deg of its face
            clock += config.ctrl_steps
            c, n = sim.data.qpos.numpy()[:, 22:25], sim.data.sensordata.numpy()[:, nrm : nrm + 3]
            m = int(sim.data.nacon.numpy()[0])
            pair, world = sim.data.contact.geom.numpy()[:m], sim.data.contact.worldid.numpy()[:m]
            hit = (pair == coin_geom).any(1) & (sim.data.contact.dist.numpy()[:m] < skin)
            touching = np.zeros((B, 6), bool)
            touching[world[hit], owner[np.where(pair[hit, 0] == coin_geom, pair[hit, 1], pair[hit, 0])]] = True
            a, b = np.take(ORDER, goal % 6), np.take(ORDER, (goal + 1) % 6)
            reached = touching[np.arange(B), b + 1] & (np.abs(c[:, 1] - lines[np.arange(B), b]) <= 0.006)
            reached &= np.where(goal % 2, 1.0, -1.0) * n[:, 2] >= np.cos(np.pi / 4)
            # the coin's phase: the lesser of what its progress across and its roll reach on the flip shape
            go = (c[:, 1] - lanes[np.arange(B), a]) / (lanes[np.arange(B), b] - lanes[np.arange(B), a])
            roll = np.arccos(np.clip((n * flip_ref(goal, np.zeros((B, 1)))[1][:, 0]).sum(1), -1, 1))
            s_coin[:] = np.minimum(np.interp(go, KEYS[1], KEYS[0]), np.interp(roll, KEYS[3], KEYS[0]))
            s[:] = np.minimum(s + config.ctrl_steps * dt / FLIP, s_coin + LEAD)
            goal[:] += reached & (goal < 12)
            s[:] = np.where(reached, -DWELL / FLIP, s)
            s[:] = np.where(goal >= 12, -1.0, s)  # after two round trips, hold the coin face up on the index
            s_coin[:] = np.where(reached, s, s_coin)
        u = np.minimum(s[:, None] + dt / FLIP * np.arange(H + 1), (s_coin + LEAD)[:, None])
        ref[:, clock : clock + H + 1, pos : pos + 3], ref[:, clock : clock + H + 1, nrm : nrm + 3] = flip_ref(goal, u)
        sensor_ref.assign(ref)
        # feed-forward over the flip (flexion, side swing): the finger the coin leaves rises, the one it lands on dips
        a, b = np.take(ORDER, goal % 6), np.take(ORDER, (goal + 1) % 6)
        bump = np.sin(np.pi * np.clip(u, 0, 1))[..., None]
        target = np.tile(GRASP.astype(np.float32), (B, H + 1, 1))
        for i in range(B):
            target[i][:, joints[a[i]]] += bump[i] * [-0.3, 0.0]
            target[i][:, joints[b[i]]] += bump[i] * [0.06, 0.0]
        prev = ctrl_ref[:, max(clock - 1, 0)]
        for t in range(H + 1):  # eased in over 50 ms
            prev = ctrl_ref[:, clock + t] = prev + (target[:, t] - prev) * 0.1
        ctrl_wp.assign(ctrl_ref)
        if mpc is not None:
            mpc.ctrl_ref.assign(ctrl_ref)

    lead([])
    print("building SimMPC (the first run compiles kernels) ...")
    mpc = SimMPC(config, MjWarpSim(plan.compile(), 1, states[0], nconmax=64, njmax=400), ctrl_ref)
    print(f"planning {STEPS * dt:.0f} s of motion ...")
    start = time.perf_counter()
    rollouts, _ = mpc.execute(sim, STEPS, callback=lead)
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- score each seed on the executed shapes: a part of the hand pushes on the coin over 2 mN; a drop is 0.3 s with no push or the coin 5 cm below its start; each goal's face is its flattest normal while it holds ---
    hand_only = scene.copy()
    hand_only.geom("coin").contype = hand_only.geom("coin").conaffinity = 0
    check, passed = hand_only.compile(), 0
    cd, forward, force = mujoco.MjData(check), mujoco.MjData(model), np.zeros(6)
    for b, x in enumerate(rollouts):
        q, n, T = x["qpos"], x["sensordata"][:, nrm : nrm + 3], STEPS
        touch, overlap, pierce, free = np.zeros((T, 6), bool), 0.0, 0.0, np.zeros(T, int)
        for t in range(T):
            forward.qpos[:], forward.qvel[:], forward.ctrl[:] = q[t], x["qvel"][t], x["ctrl"][t]
            mujoco.mj_forward(model, forward)
            for c in range(forward.ncon):
                con = forward.contact[c]
                if coin_geom in (con.geom1, con.geom2):
                    mujoco.mj_contactForce(model, forward, c, force)
                    touch[t, owner[con.geom1 + con.geom2 - coin_geom]] |= force[0] > 0.002
                    pierce = max(pierce, -con.dist)
            cd.qpos[:] = q[t]
            mujoco.mj_kinematics(check, cd)
            mujoco.mj_collision(check, cd)
            overlap = max(overlap, -cd.contact.dist[: cd.ncon].min(initial=0.0))
            free[t] = 0 if touch[t].any() else free[t - 1] + 1
        drop = np.flatnonzero((free * dt >= 0.3) | (q[:, 24] - q[0, 24] < -0.05))
        k, reach = 0, [0]
        for t in range(T):
            f = ORDER[(k + 1) % 6]
            if touch[t, f + 1] and abs(q[t, 23] - lines[b, f]) <= 0.006 and (-1) ** (k + 1) * n[t, 2] >= np.sqrt(0.5):
                k, reach = k + 1, reach + [t]
        flat = []
        for j, (t0, t1) in enumerate(zip(reach, reach[1:] + [T])):
            f = ORDER[j % 6]
            ok = touch[t0:t1, f + 1] & (np.abs(q[t0:t1, 23] - lines[b, f]) <= 0.006)
            flat.append(n[t0 + np.where(ok, (-1) ** j * n[t0:t1, 2], -2.0).argmax()])
        turns = [float(np.degrees(np.arccos(np.clip(u @ v, -1, 1)))) for u, v in zip(flat, flat[1:])]
        jitter = np.sqrt(np.mean(np.diff(q[::4, :22], 2, axis=0) ** 2)) / 0.02**2
        trip = k >= 6 and min(turns[:6]) >= 150 and pierce <= 0.002 and (drop[:1] > reach[6]).all()
        ok = bool(trip and jitter <= 5 and overlap <= 0.001)
        passed += ok
        print(f"seed {seed0 + b}: {'pass' if ok else 'fail'}; {k} goals, turns {np.round(turns)}, coin "
              f"{1000 * pierce:.1f} mm in, jitter {jitter:.1f} rad/s^2, overlap {1000 * overlap:.1f} mm")  # fmt: skip
    print(f"{passed}/{B} seeds pass")
    if B > 1:
        return

    # --- viewer: replay the simulated motion ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    viewer = ViserMjcf(server, scene.to_xml())
    while True:
        for k in range(0, STEPS, 4):
            viewer.update_qpos(rollouts[0]["qpos"][k])
            time.sleep(4 * dt)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    tyro.cli(main)
