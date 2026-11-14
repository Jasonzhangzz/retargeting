# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a palm-down Sharpa Wave hand unscrews a bottle cap in 60 deg thumb-and-finger twists, with no training. Each seed starts from reset noise on the joints and the bottle's place and tilt, and passes with two turns in 15 s, no hand shape 2 mm into the cap or bottle, jitter at most 5 rad/s^2, no two hand shapes overlapping by over 1 mm, the cap within 15 deg of its reference (median) and no two of the four fingers touching. Held-out seeds 20000-20023 pass 23/24 in each of two runs; one seed plans 15 s in 59 s on an RTX 3090."""

import time

import mujoco
import numpy as np
import tyro
import viser
import warp as wp

from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import ContactTask, ControlTask, SensorTask, StateTask, SupportTask
from robokit.utils.visualize_utils import ViserMjcf


STEPS = 3000  # 15 s at the 5 ms step
STEP, PERIOD, TURN, COUNT = np.radians(60), 0.8, 0.45, 14  # each twist: 60 deg over 0.45 s of every 0.8 s; 14 in all
GRASP = np.array([0.8408, 0.0874, 0.4856, -0.0684, 0.2538, 0.5119, -0.0032, 0.4689, 0.3745, 0.4867, -0.0119, 0.8655,
                  0.3462, 0.9, 0.1, 1.1, 0.7, 0.1, 0.9, 0.2444, 1.1, 0.7])  # thumb opposite index, middle  # fmt: skip


def reference(phase: np.ndarray) -> np.ndarray:
    """The cap's planned turn, rad, at each phase, s: COUNT minimum-jerk twists of STEP, each held while fingers reset.

    Example:
        >>> np.degrees(reference(np.array([0.0, 0.225, 0.6, 1.025, 50.0]))).round(1).tolist()
        [0.0, 30.0, 60.0, 90.0, 840.0]
    """
    k = np.minimum(phase // PERIOD, COUNT)
    u = np.clip((phase - k * PERIOD) / TURN, 0, 1)
    return STEP * (k + (k < COUNT) * u**3 * (10 - 15 * u + 6 * u**2))


def main(seeds: int = 1, seed0: int = 0):
    """Plan `seeds` seeds from `seed0` at once, score each, and replay the first."""
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()
    centre, skin = np.array([0.0, 0.0, 0.205]), 0.001  # the cap's centre, the bottle below it; contact starts 1 mm out
    fingers = ("thumb", "index", "middle", "ring", "pinky")  # the first three twist the cap

    # --- scene: the hand palm-down over a fixed 500 ml PET bottle, friction 1, stiff against itself; the 30 x 12 mm, 2.5 g cap of a 28 mm neck turns on a hinge with a cracked-open cap's 0.05 N m running torque, and a slide coupled to it by the 3 mm thread pitch lifts it as it turns ---
    scene = mujoco.MjSpec.from_file(str(sharpa_hand.MJCF_PATH_RIGHT))
    for mesh in scene.meshes:  # absolute mesh paths, so the viewer's copy finds them
        mesh.file = str(sharpa_hand.MJCF_PATH_RIGHT.parent / scene.meshdir / mesh.file)
    scene.meshdir, scene.option.timestep = "", 0.005
    root = scene.body("right_hand_C_MC")
    root.pos, root.quat = [-0.0823, -0.1084, 0.2861], [0.6424, -0.2343, 0.6933, 0.2274]
    for geom in scene.geoms:
        if geom.contype or geom.conaffinity:
            geom.friction, geom.solmix = [1.0, 0.005, 1e-4], 0.0
            geom.solref, geom.solimp[:3] = [0.01, 1.0], [0.99, 0.999, 0.001]
    stiff = dict(friction=[1.0, 0.005, 1e-4], solref=[0.01, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2])
    cylinder = mujoco.mjtGeom.mjGEOM_CYLINDER
    bottle = scene.worldbody.add_body(name="bottle", pos=centre)  # its frame is the cap's centre, so tilts pivot there
    for name, radius, half, z in [("body", 0.0325, 0.085, -0.12), ("neck", 0.0125, 0.0145, -0.0205),
                                  ("ring", 0.0165, 0.001, -0.015)]:  # radius, half height, z; m  # fmt: skip
        bottle.add_geom(name=f"bottle_{name}", type=cylinder, size=[radius, half, 0], pos=[0, 0, z],
                        contype=2, conaffinity=1, margin=skin, rgba=[0.75, 0.88, 0.95, 0.5], **stiff)  # fmt: skip
    cap = bottle.add_body(name="cap")
    cap.add_joint(name="lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    firm = [0.9999, 0.9999, 0.001, 0.5, 2]  # MuJoCo's default would let the light cap creep at 26 rad/s at 0.02 N m
    cap.add_joint(name="turn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], frictionloss=0.05, damping=5e-4,
                  solref_friction=[0.01, 1.0], solimp_friction=firm, limited=True, range=[-0.05, 1e3],
                  solref_limit=[0.01, 1.0], solimp_limit=firm)  # seated, it cannot tighten further  # fmt: skip
    cap.add_geom(name="cap", type=cylinder, size=[0.015, 0.006, 0], mass=0.0025, contype=4, conaffinity=1,
                 margin=skin, condim=4, rgba=[0.9, 0.2, 0.15, 1], **stiff)  # fmt: skip
    cap.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.013, 0.002, 5e-4], pos=[0, 0, 0.006], contype=0, conaffinity=0,
                 density=0, rgba=[1, 1, 1, 1])  # a stripe across the top shows the turn  # fmt: skip
    thread = scene.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="turn", name2="lift")  # turn = 2 pi lift / pitch
    thread.data[1], thread.solref, thread.solimp[:3] = 2 * np.pi / 0.003, [0.01, 1.0], [0.99, 0.999, 0.001]
    for actuator in scene.actuators:  # twice the servos' damping, calmer fingers; their speeds, for the plan's cost
        actuator.biasprm[2] *= 2
        scene.add_sensor(name=actuator.name, type=mujoco.mjtSensor.mjSENS_ACTUATORVEL,
                         objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname=actuator.name)  # fmt: skip
    for f in fingers[:3]:  # where the thumb, index and middle tips are, for the plan's cost on their height
        scene.add_sensor(name=f, type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_XBODY,
                         objname=f"right_{f}_DP")  # fmt: skip
    scene.add_sensor(name="turn", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                     objname="turn")  # fmt: skip
    plan = scene.copy()  # planned on the capsules, cap and bottle, with a capped solver; executed on every shape
    plan.option.iterations, plan.option.ls_iterations = 8, 8
    for geom in plan.geoms:
        kept = geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE or geom.name.endswith("VL_fit")  # links and knuckles
        if not kept and not geom.name.startswith(("cap", "bottle")):
            geom.contype = geom.conaffinity = 0

    # --- starts: 0.03 rad on the joints, 2 mm on the bottle's place and 5 deg of tilt about x and y, a 1 s settle ---
    B, specs, plans, models, states, hold = seeds, [], [], [], [], GRASP.copy()
    hold[[0, 5, 9]] += 0.1  # the commands press the thumb, index and middle onto the cap
    hold[[6, 10]] += [-0.05, 0.05]  # and hold the index and middle a little apart
    for b in range(B):
        rng = np.random.default_rng(seed0 + b)
        q, place = GRASP + rng.uniform(-0.03, 0.03, 22), centre + rng.uniform(-0.002, 0.002, 3)
        quat = np.array([1.0, 0, 0, 0])
        mujoco.mju_quatIntegrate(quat, [*rng.uniform(-np.radians(5), np.radians(5), 2), 0.0], 1.0)
        for spec, out in ((scene.copy(), specs), (plan.copy(), plans)):
            spec.body("bottle").pos, spec.body("bottle").quat = place, quat
            out.append(spec)
        models.append(specs[-1].compile())
        states.append(mujoco.MjData(models[-1]))
        states[-1].qpos[:22], states[-1].ctrl[:] = q, hold + q - GRASP
        for _ in range(200):
            mujoco.mj_step(models[-1], states[-1])
    model = models[0]
    print(f"scene: {model.nq} qpos, {model.nu} actuators; cap at {np.degrees(states[0].qpos[23]):.1f} deg")

    # --- SimMPC: track the reference on a phase clock; the twisting tips stay at the rim's height, two on the cap; touches by the lower phalanges, of the bottle and between fingers cost, and the fingertips' a little; the joints are kept slow, and the ring and pinky barely explored ---
    H, turn, dt = 100, model.sensor("turn").adr[0], model.opt.timestep
    tips = [model.sensor(f).adr[0] + 2 for f in fingers[:3]]
    height = np.array([[s.xpos[model.body(f"right_{f}_DP").id, 2] for f in fingers[:3]] for s in states])
    ref = np.zeros((B, STEPS + H + 1, model.nsensordata), np.float32)
    sensor_ref = wp.array(ref, dtype=float)
    ctrl_ref = np.tile(hold, (ref.shape[1], 1))
    ctrl_std = np.full(model.nu, 0.09)
    ctrl_std[13:] *= 0.2  # the ring and pinky stay out of the twist
    config = SimMPCConfig(  # one iteration of 512 draws a plan, weighted by their cheapest 10%
        ctrl_std=ctrl_std, batch_size=B, num_samples=512, horizon_steps=H, node_steps=25, ctrl_steps=8,
        stages=[SimMPCStage(1)], elite_ratio=0.1, interpolation="cubic", std_growth=1.0, terminal_scale=1.0,
        seed=seed0,
    )  # fmt: skip
    links = [geom.name for geom in scene.geoms if geom.name.endswith("_fit") or geom.name.startswith("bottle")]
    tip_geoms = [f"right_{f}_fingertip_collision" for f in fingers[:3]]
    config.add(SensorTask(sensor_ref, dict(turn=60.0)), terminal_scale=10.0)
    config.add(SensorTask(sensor_ref, {actuator.name: 0.1 for actuator in scene.actuators}))  # their references are 0
    config.add(SensorTask(sensor_ref, {f: [0, 0, 1e5] for f in fingers[:3]}, hinge=1e5 * 0.003**2))  # 3 mm of the rim
    config.add(ControlTask(1.0, ctrl_ref))
    config.add(ContactTask(links, np.zeros((1, len(links), 2)), np.zeros((1, len(links))), penalty=20.0, margin=skin))
    config.add(SupportTask("cap", [f"{f}_fingertip" for f in fingers[:3]], 2, 5.0, skin))  # two of them on the cap
    config.add(ContactTask(tip_geoms, np.zeros((1, 3, 2)), np.zeros((1, 3)), penalty=5.0, margin=skin))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "" for g in range(model.ngeom)]
    of = {f: np.array([f"_{f}_" in n for n in names]) for f in fingers}
    rest = of["index"] | of["middle"] | of["ring"] | of["pinky"]
    for a, b in [(of["thumb"], rest), (of["index"] | of["ring"], of["middle"] | of["pinky"])]:  # any touch costs
        config.add(StateTask(np.zeros((1, model.nq)), np.zeros((1, model.nv)), np.zeros(model.nv),
                             penetration_groups=(a, b), penetration_scale=1e7, margin=-0.002))  # fmt: skip
    sim = MjWarpSim(specs, [1] * B, nconmax=48, njmax=300)  # each seed's bottle where its draw put it
    for f in ("time", "qpos", "qvel", "act", "qacc_warmstart"):  # each seed from its start
        getattr(sim.data, f).assign(np.stack([getattr(s, f) for s in states]))
    clock, phase, first = 0, np.zeros(B), np.array([s.qpos[23] for s in states])

    def lead(plan: list):
        nonlocal clock
        angle = sim.data.qpos.numpy()[:, 23] - first
        if plan:  # the phase advances unless the cap lags its reference by over 0.35 rad
            clock += config.ctrl_steps
            phase[reference(phase + config.ctrl_steps * dt) <= angle + 0.35] += config.ctrl_steps * dt
        p = phase[:, None] + dt * np.arange(H + 1)
        ref[:, clock : clock + H + 1, turn] = first[:, None] + reference(p)
        lift = reference(p)[..., None] * 0.003 / (2 * np.pi)  # the cap rises 3 mm a turn; the tips stay at its rim
        ref[:, clock : clock + H + 1, tips] = height[:, None] + lift
        sensor_ref.assign(ref)

    lead([])
    print("building SimMPC (the first run compiles kernels) ...")
    # planned on `plans`
    mpc = SimMPC(config, MjWarpSim(plans, [1] * B, nconmax=48, njmax=300), ctrl_ref)
    print(f"planning {STEPS * dt:.0f} s of motion ...")
    start = time.perf_counter()
    rollouts, _ = mpc.execute(sim, STEPS, callback=lead)
    print(f"planned in {time.perf_counter() - start:.0f} s")

    # --- score each seed on the executed shapes (CPU MuJoCo): depths into the cap and bottle, every second state; overlaps and touches between fingers, every state ---
    finger = [next((i for i, f in enumerate(fingers) if f"_{f}_" in n), -1) for n in names]
    objects = {model.geom(n).id for n in ("cap", "bottle_body", "bottle_neck", "bottle_ring")}
    cap_geom = model.geom("cap").id
    hand_only = scene.copy()
    for geom in hand_only.geoms:
        if geom.name.startswith(("cap", "bottle")):
            geom.contype = geom.conaffinity = 0
    check, passed = hand_only.compile(), 0
    for b, x in enumerate(rollouts):
        q, forward, cd = x["qpos"], mujoco.MjData(models[b]), mujoco.MjData(check)
        depth, overlap, touches = {"cap": 0.0, "bottle": 0.0}, 0.0, 0
        for t in range(STEPS):
            cd.qpos[:] = q[t]
            mujoco.mj_kinematics(check, cd)
            mujoco.mj_collision(check, cd)
            for con in cd.contact[: cd.ncon]:
                overlap, f1, f2 = max(overlap, -con.dist), finger[con.geom1], finger[con.geom2]
                touches += f1 > 0 and f2 > 0 and f1 != f2  # two of the four fingers touch
            if t % 2 == 0:
                forward.qpos[:], forward.qvel[:], forward.ctrl[:] = q[t], x["qvel"][t], x["ctrl"][t]
                mujoco.mj_forward(models[b], forward)
                for con in forward.contact[: forward.ncon]:
                    if con.geom1 in objects or con.geom2 in objects:
                        key = "cap" if cap_geom in (con.geom1, con.geom2) else "bottle"
                        depth[key] = max(depth[key], -con.dist)
        net = q[-1, 23] - q[0, 23]
        jitter = np.sqrt(np.mean(np.diff(q[::4, :22], 2, axis=0) ** 2)) / 0.02**2
        error = np.degrees(np.median(np.abs(q[:, 23] - ref[b, :STEPS, turn])))  # to the planner's reference
        ok = bool(net >= 4 * np.pi and max(depth.values()) <= 0.002 and jitter <= 5 and overlap <= 0.001)
        ok &= bool(error <= 15 and touches == 0)
        passed += ok
        print(f"seed {seed0 + b}: {'pass' if ok else 'fail'}; {net / (2 * np.pi):.2f} turns, {error:.1f} deg off, "
              f"cap {1000 * depth['cap']:.1f} mm and bottle {1000 * depth['bottle']:.1f} mm in, jitter {jitter:.1f} "
              f"rad/s^2, overlap {1000 * overlap:.1f} mm, {touches} finger touches")  # fmt: skip
    print(f"{passed}/{B} seeds pass")
    if B > 1:
        return

    # --- viewer: replay the simulated motion ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    viewer = ViserMjcf(server, specs[0].to_xml())
    while True:
        for k in range(0, STEPS, 4):
            viewer.update_qpos(rollouts[0]["qpos"][k])
            time.sleep(4 * dt)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    tyro.cli(main)
