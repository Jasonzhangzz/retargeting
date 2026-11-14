# /// script
# requires-python = ">=3.10"
# dependencies = ["robokit[mjwarp]"]
#
# [tool.uv.sources]
# robokit = { path = "../../../..", editable = true }
# ///
# pyright: reportMissingImports=false
# pyright: reportArgumentType=false
"""SimMPC: a palm-down Sharpa Wave hand twirls a free precision screwdriver to drive a slotted screw two turns."""

import copy
import time

import mujoco
import numpy as np
import viser
import warp as wp

from robokit.assets.robots.end_effectors import sharpa_hand
from robokit.helpers.sim_mpc import SimMPC, SimMPCConfig, SimMPCStage
from robokit.sim import MjWarpSim
from robokit.terms import ContactTask, ControlTask, SensorTask, StateTask, SupportTask
from robokit.utils.visualize_utils import ViserMjcf


STEPS = 1500  # 15 s at the 10 ms step
QUARTER, GOALS = np.pi / 2, 9  # goals every quarter turn up to 2.25 turns: the one after next past the screw's angle
PERIOD, DUTY, PACE = 1.5, 0.6, 1.2  # thumb, middle push in turn for 60% of every 1.5 s; a quarter per 1.2 s at most
HOLD = 4.0  # while the tool tilts over 4 deg (until back under 3), the twirl stops and the fingers right it
PRESS = 0.001  # the fingertips aim 1 mm below where they first held the handle, pressing the blade into the slot
# a tripod grip, solved at the start: per tip, where its collision capsule touches the driver (geom, height mm, angle
# deg, radius mm, in the driver's frame), at what gap (mm), with which point of its pad (mm, in its body's frame)
GRASP_AT = dict(thumb=("handle", 82.173, -130.83, 8, -4.411, [26.466, 6.138, 6.444]),
                index=("handle", 80.065, 85.14, 8, 0.386, [22.121, -2.476, 5.163]),
                middle=("handle", 79.515, -13.17, 8, 0.097, [21.882, -1.922, -5.702]))  # fmt: skip
CURL = [1.1, 0.15, 1.2, 0.8, 0.15, 1.0, 0.33, 1.1, 0.7]  # the ring and pinky, loosely curled
ROOT_POS, ROOT_QUAT = [-0.0682, -0.0790, 0.2779], [0.6182, -0.2554, 0.7151, 0.2030]
HEAD, SLOT, SKIN = 0.0973, 0.0013, 0.001  # the screw head's top above the floor; the slot's depth; the 1 mm skin; m
TIPS = ("thumb", "index", "middle")  # the fingertips that twist the handle


def main():
    wp.config.log_level = wp.LOG_WARNING  # no per-module load lines
    wp.init()

    # --- scene: the hand palm-down over an M5 x 0.8 slotted cheese-head screw (8.5 mm head, 1.2 x 1.3 mm slot) in a
    # fixed block, turning on a hinge with 0.02 N m of running torque and sinking 0.8 mm a turn on a slide coupled to
    # it; a free 4 mm precision screwdriver (0.8 x 4.0 blade, 4 x 40 mm shaft, 16 x 70 mm handle, 12 x 6 mm swivel cap,
    # 18 g) stands in the slot, and only the blade's contact with the slot turns the screw ---
    scene = mujoco.MjSpec.from_file(str(sharpa_hand.MJCF_PATH_RIGHT_FIT))  # its finger capsules fitted to the meshes
    for mesh in scene.meshes:  # absolute mesh paths, so the viewer's copy finds them
        mesh.file = str(sharpa_hand.MJCF_PATH_RIGHT_FIT.parent / scene.meshdir / mesh.file)
    scene.meshdir, scene.option.timestep = "", 0.01
    root = scene.body("right_hand_C_MC")
    root.pos, root.quat = ROOT_POS, ROOT_QUAT
    for geom in scene.geoms:
        if geom.contype or geom.conaffinity:
            geom.friction, geom.solmix = [1.0, 0.005, 1e-4], 0.0
            geom.solref, geom.solimp[:3] = [0.01, 1.0], [0.99, 0.999, 0.001]
    stiff = dict(solref=[0.01, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2])
    soft = dict(friction=[1.0, 0.005, 1e-4], margin=SKIN, **stiff)  # what the fingers touch
    steel = dict(friction=[0.3, 0.005, 1e-4], **stiff)  # blade and slot
    cyl, box = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX
    block = scene.worldbody.add_body(name="block", pos=[0, 0, HEAD - 0.0173])
    block.add_geom(name="block", type=box, size=[0.02, 0.02, 0.01], contype=2, conaffinity=1, rgba=[0.6, 0.62, 0.66, 1],
                   **soft)  # fmt: skip
    screw = block.add_body(name="screw", pos=[0, 0, 0.0173])  # its frame: the head's top centre, the slot along x
    screw.add_joint(name="sink", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, -1], limited=True, range=[-0.001, 0.004])
    firm = [0.9999, 0.9999, 0.001, 0.5, 2]  # MuJoCo's default would let the light screw creep
    screw.add_joint(name="turn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, -1], frictionloss=0.02, damping=1e-5,
                    solref_friction=[0.01, 1.0], solimp_friction=firm)  # tightens clockwise seen from above  # fmt: skip
    gold = [0.85, 0.7, 0.3, 1]
    screw.add_geom(name="screw_head", type=cyl, size=[0.00425, 0.001, 0], pos=[0, 0, -SLOT - 0.001], mass=0.0012,
                   contype=4, conaffinity=1, rgba=gold, **steel)  # fmt: skip
    for side in (-1, 1):  # the cheeks either side of the slot
        screw.add_geom(name=f"screw_cheek{side + 1}", type=box, size=[0.0035, 0.001825, SLOT / 2],
                       pos=[0, side * 0.002425, -SLOT / 2], mass=0.0003, contype=4, conaffinity=1, rgba=gold,
                       **steel)  # fmt: skip
    thread = scene.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="turn", name2="sink")  # turn = 2 pi sink / pitch
    thread.data[1], thread.solref, thread.solimp[:3] = 2 * np.pi / 0.0008, [0.01, 1.0], [0.99, 0.999, 0.001]
    tool = scene.worldbody.add_body(name="tool", pos=[0, 0, HEAD - SLOT])  # its frame: the blade tip
    tool.add_freejoint(name="tool")
    tool.add_site(name="tip", size=[1e-4, 0, 0], rgba=[0, 0, 0, 0])
    grey, red = [0.75, 0.76, 0.8, 1], [0.85, 0.15, 0.12, 1]
    tool.add_geom(name="blade", type=box, size=[0.002, 0.0004, 0.002], pos=[0, 0, 0.002], mass=0.0001, contype=8,
                  conaffinity=4, rgba=grey, **steel)  # fmt: skip
    tool.add_geom(name="shaft", type=cyl, size=[0.002, 0.02, 0], pos=[0, 0, 0.024], mass=0.0039, contype=8,
                  conaffinity=7, rgba=grey, **soft)  # fmt: skip
    tool.add_geom(name="handle", type=cyl, size=[0.008, 0.035, 0], pos=[0, 0, 0.079], mass=0.013, contype=8,
                  conaffinity=7, rgba=red, **soft)  # fmt: skip
    cap = tool.add_body(name="cap", pos=[0, 0, 0.114])
    cap.add_joint(name="swivel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=1e-5)
    cap.add_geom(name="cap", type=cyl, size=[0.006, 0.003, 0], pos=[0, 0, 0.003], mass=0.0008, contype=8,
                 conaffinity=7, rgba=[0.15, 0.15, 0.17, 1], **soft)  # fmt: skip
    for actuator in scene.actuators:  # twice the servos' damping, calmer fingers; their speeds, for the plan's cost
        actuator.biasprm[2] *= 2
        scene.add_sensor(name=actuator.name, type=mujoco.mjtSensor.mjSENS_ACTUATORVEL,
                         objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname=actuator.name)  # fmt: skip
    for f in TIPS:  # where the thumb, index and middle tips are, for the plan's cost on their height
        scene.add_sensor(name=f, type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_XBODY,
                         objname=f"right_{f}_DP")  # fmt: skip
    for f in TIPS:  # the tips' gaps to the handle
        scene.add_sensor(name=f"gap_{f}", type=mujoco.mjtSensor.mjSENS_GEOMDIST, objtype=mujoco.mjtObj.mjOBJ_GEOM,
                         objname=f"right_{f}_fingertip_collision", reftype=mujoco.mjtObj.mjOBJ_GEOM, refname="handle",
                         cutoff=0.05)  # fmt: skip
    scene.add_sensor(name="turn", type=mujoco.mjtSensor.mjSENS_JOINTPOS, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                     objname="turn")  # fmt: skip
    scene.add_sensor(name="up", type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS, objtype=mujoco.mjtObj.mjOBJ_BODY,
                     objname="tool")  # fmt: skip
    scene.add_sensor(name="seat", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tip",
                     reftype=mujoco.mjtObj.mjOBJ_BODY, refname="screw")  # the blade's tip in the screw's frame  # fmt: skip
    model = scene.compile()
    plan = scene.copy()  # planned on the capsules, the screwdriver, screw and block, with a capped solver
    plan.option.iterations, plan.option.ls_iterations = 8, 8
    plan.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD  # box-box as a primitive, not convex
    plan.geom("screw_head").type, plan.geom("screw_head").size = box, [0.00425, 0.00425, 0.001]  # blade on it: box-box
    for geom in plan.geoms:
        kept = geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE or geom.name.endswith("VL_fit")  # links and knuckles
        if not kept and not geom.name.startswith(("screw", "block", "blade", "shaft", "handle", "cap")):
            geom.contype = geom.conaffinity = 0
    plan_model = plan.compile()

    # --- start: the fingers close on the screwdriver for 1 s while it is held seated (heavy, no gravity), then free ---
    state, heavy = mujoco.MjData(model), copy.copy(model)
    grasp, tool_b, start = np.r_[[0.5] * 13, CURL], model.body("tool").id, time.perf_counter()
    for f, rows in (("thumb", [0, 1, 2, 3, 4]), ("index", [5, 6, 7, 8]), ("middle", [9, 10, 11, 12])):
        geom, z, angle, r, gap, pad = GRASP_AT[f]  # damped least squares on the finger's joints
        tip, a = model.geom(f"right_{f}_fingertip_collision"), np.radians(angle)
        target = np.r_[r * np.cos(a), r * np.sin(a), z, gap, pad] / 1000

        def miss(q: np.ndarray) -> np.ndarray:  # the tip's point on the driver (its frame), gap and pad point, off
            state.qpos[:22], ft = q, np.zeros(6)
            mujoco.mj_kinematics(model, state)
            g = mujoco.mj_geomDistance(model, state, tip.id, model.geom(geom).id, 1.0, ft)
            R, b = state.xmat[tool_b].reshape(3, 3), model.geom_bodyid[tip.id]
            on_pad = state.xmat[b].reshape(3, 3).T @ (ft[:3] - state.xpos[b])
            return np.r_[R.T @ (ft[3:] - state.xpos[tool_b]), g, on_pad] - target

        for _ in range(60):
            e, J = miss(grasp), np.zeros((7, len(rows)))
            for k, j in enumerate(rows):
                J[:, k] = (miss(grasp + 1e-5 * (np.arange(22) == j)) - e) / 1e-5
            dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(len(rows)), J.T @ e)
            grasp[rows] = np.clip(grasp[rows] - dq * min(1, 0.2 / (np.abs(dq).max() + 1e-12)), *model.jnt_range[rows].T)
        print(f"{f} on the handle within {1000 * np.linalg.norm(miss(grasp)):.4f} mm")
    print(f"tripod grip solved in {time.perf_counter() - start:.2f} s")
    hold = grasp.copy()
    hold[[3, 5, 9, 7, 8, 11, 12]] += [0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15]  # the commands press the tips onto it
    hold[[6, 10]] += [-0.05, 0.05]  # and hold the index and middle a little apart
    tb, tool_q, dof = model.body("tool").id, model.joint("tool").qposadr[0], model.joint("tool").dofadr[0]
    heavy.body_mass[tb], heavy.body_inertia[tb], heavy.opt.gravity = (
        1e4 * model.body_mass[tb],
        1e4 * model.body_inertia[tb],
        0,
    )
    state.qpos[:22], state.ctrl[:] = grasp, hold
    placed = state.qpos[tool_q : tool_q + 7].copy()
    for _ in range(100):
        mujoco.mj_step(heavy, state)
        state.qpos[tool_q : tool_q + 7], state.qvel[dof : dof + 6] = placed, 0.0
    mujoco.mj_forward(model, state)

    # --- SimMPC: the screw chases the quarter turn after next; the fingers twirl the handle without opening, the thumb
    # and middle pushing in turn; the tips stay at their height (pressing PRESS lower) and on the handle; the
    # handle stays upright and its blade at the slot's centre; touches by the lower phalanges, of the block, screw and
    # cap, by the ring and pinky tips, and between fingers cost; the joints are kept slow, the ring and pinky still ---
    H, turn_q = 50, model.joint("turn").qposadr[0]
    turn, tips = model.sensor("turn").adr[0], [model.sensor(f).adr[0] + 2 for f in TIPS]
    height = np.array([state.xpos[model.body(f"right_{f}_DP").id, 2] for f in TIPS])
    ref = np.zeros((1, STEPS + H + 1, model.nsensordata), np.float32)
    ref[0, :, model.sensor("up").adr[0] + 2], ref[0, :, model.sensor("seat").adr[0] + 2] = 1.0, -SLOT
    sensor_ref, grip = wp.array(ref, dtype=float), wp.zeros((1, ref.shape[1], 3), dtype=float)
    ctrl_ref = np.tile(hold, (ref.shape[1], 1))
    links = [g.name for g in scene.geoms if g.name.endswith("_fit") or g.name.startswith(("block", "screw", "cap"))]
    links += ["right_ring_fingertip_collision", "right_pinky_fingertip_collision"]
    tip_geoms = [f"right_{f}_fingertip_collision" for f in TIPS]
    ctrl_std = np.full(model.nu, 0.09)
    ctrl_std[13:] *= 0.2  # the ring and pinky stay out of the twist
    config = SimMPCConfig(
        ctrl_std=ctrl_std, ctrl_range=model.actuator_ctrlrange, num_samples=128, horizon_steps=H, node_steps=12,
        ctrl_steps=4, interpolation="cubic", std_growth=1.0, terminal_scale=1.0, stages=[SimMPCStage(2)],  # at the timestep
    )  # fmt: skip
    config.add(SensorTask(sensor_ref, dict(turn=50.0)), terminal_scale=10.0)
    costs = [
        SensorTask(sensor_ref, {actuator.name: 0.1 for actuator in scene.actuators}),  # their references are 0
        SensorTask(sensor_ref, {f: [0, 0, 1e5] for f in TIPS}, hinge=1e5 * 0.0005**2),  # within 0.5 mm of it
        SensorTask(sensor_ref, dict(up=[8000.0] * 3, seat=[1e7] * 3)),
        ControlTask(1.0, ctrl_ref),
        ContactTask(links, np.zeros((1, len(links), 2)), np.zeros((1, len(links))), penalty=20.0, margin=SKIN),
        SupportTask("handle", [f"{f}_fingertip" for f in TIPS], 2, 20.0, SKIN),  # two of them stay on the handle
        ContactTask(["cap"], np.zeros((1, 1, 2)), np.zeros((1, 1)), penalty=200.0, margin=SKIN),  # hands off the cap
        SensorTask(sensor_ref, dict(gap_thumb=1e6, gap_index=1e6), hinge=1e6 * 0.003**2),  # tips within 3 mm of it
        SensorTask(sensor_ref, dict(gap_middle=1e7), hinge=1e7 * 0.003**2),  # the middle, which drifts, held harder
        # a fingertip's touch costs 5, and earns 10 while `grip` is 1
        ContactTask(tip_geoms, np.zeros((1, 3, 2)), grip, weight=20.0, penalty=10.0, margin=SKIN),
    ]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "" for g in range(model.ngeom)]
    of = {f: np.array([f"_{f}_" in n for n in names]) for f in ("thumb", "index", "middle", "ring", "pinky")}
    fingers = of["index"] | of["middle"] | of["ring"] | of["pinky"]
    for a, b in [(of["thumb"], fingers), (of["index"] | of["ring"], of["middle"] | of["pinky"])]:  # any touch costs
        costs.append(StateTask(np.zeros((1, model.nq)), np.zeros((1, model.nv)), np.zeros(model.nv),
                               penetration_groups=(a, b), penetration_scale=1e8, margin=-0.002))  # fmt: skip
    for term in costs:
        config.add(term)
    sim = MjWarpSim(model, 1, state, nconmax=96, njmax=600)  # executes; the plan's rollouts run on `plan_model`
    clock, first, twirl, holding = 0, state.qpos[turn_q], 0.0, False

    def lead(plan: list):
        nonlocal clock, twirl, holding
        q = sim.data.qpos.numpy()[0] if plan else state.qpos
        angle, dt = q[turn_q] - first, model.opt.timestep
        clock += config.ctrl_steps if plan else 0
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[tool_q + 4] ** 2 + q[tool_q + 5] ** 2), -1, 1)))
        holding = tilt > HOLD - 1 if holding else tilt > HOLD  # then all three hold and the screw keeps its angle
        twirl += 0.0 if holding or not plan else config.ctrl_steps * dt  # the twirl's clock, stopped while holding
        t = twirl + np.arange(H + 1) * dt
        goal = min(np.floor(angle / QUARTER) + 2, np.floor(twirl / PACE) + 2, GOALS)
        ref[0, clock : clock + H + 1, turn] = first + (angle if holding else goal * QUARTER)
        ref[0][clock : clock + H + 1, tips] = height - angle * 0.0008 / (2 * np.pi) - PRESS  # sinking 0.8 mm a turn
        sensor_ref.assign(ref)
        g = np.zeros((1, ref.shape[1], 3))
        g[0, clock : clock + H + 1] = holding | ((t[:, None] / PERIOD + [0.0, 0.0, 0.5]) % 1 < [DUTY, 2.0, DUTY])
        grip.assign(g)

    lead([])
    print("building SimMPC (the first run compiles kernels) ...")
    mpc = SimMPC(config, MjWarpSim(plan_model, 1, state, nconmax=96, njmax=600), ctrl_ref)
    print(f"planning {STEPS * model.opt.timestep:.0f} s of motion ...")
    start = time.perf_counter()
    (rollout,), _ = mpc.execute(sim, STEPS, callback=lead)
    planned = time.perf_counter() - start
    q = rollout["qpos"]
    jitter = np.sqrt(np.mean(np.diff(q[::2, :22], 2, axis=0) ** 2)) / 0.02**2  # 20 ms apart
    print(f"planned in {planned:.0f} s; the screw turned {(q[-1, turn_q] - q[0, turn_q]) / (2 * np.pi):.2f} times, "
          f"jitter {jitter:.1f} rad/s^2")  # fmt: skip

    # --- viewer: replay the simulated motion ---
    server = viser.ViserServer()
    print("replaying; Ctrl+C to stop")
    viewer = ViserMjcf(server, scene.to_xml())
    for name in ("visual", "collision"):  # toggles: the drawn meshes, the collision shapes
        box = server.gui.add_checkbox(f"Show {name}", name == "visual")
        box.on_update(lambda _, b=box, n=name: setattr(viewer, f"show_{n}", b.value))
    while True:
        for k in range(0, STEPS, 2):
            viewer.update_qpos(rollout["qpos"][k])
            time.sleep(2 * model.opt.timestep)
        time.sleep(1.0)  # hold the last frame before replaying


if __name__ == "__main__":
    main()
