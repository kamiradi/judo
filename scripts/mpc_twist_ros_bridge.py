# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
"""Bridge judo's `peg_insertion_twist` MPC to the iiwa twist action servers over rosbridge.

Runs INSIDE the judo pixi env (Python 3.13). It never imports ROS -- it talks to a
`rosbridge_server` (websocket) running in the ROS1 container via `roslibpy` (pure Python).

Data flow
---------
    judo MPC (this process)                ROS1 container
    -----------------------                --------------
    PegInsertionTwist controller
        action(t) -> [vx,vy,vz,wx,wy,wz]
        --(geometry_msgs/Twist)----------> /twist_input  --> action_server_cmd_twist
                                                              (optas QP -> joint cmd -> mux)
        <--(sensor_msgs/JointState)-------  /<ns>/joint_states   (feedback, optional)

The twist task's control IS a world-frame end-effector twist, so the MPC action maps
1:1 onto geometry_msgs/Twist -- no FK or differencing needed here.

Two state-feedback modes (see --feedback):
  * deadreckon : advance judo's own carriage state by integrating the commanded twist.
                 Open-loop w.r.t. the real robot, but self-consistent -- good for a first
                 "do they communicate + does the sim move" test.
  * pose       : close the loop on the real peg pose, read from TF via roslibpy's TFClient
                 (which uses the tf2_web_republisher node to compose <fixed_frame> ->
                 <target_frame>). See _qpos_from_pose for the frame caveats.

Start the ROS side first (in the container):
    roslaunch iiwa_peg action_servers.launch
    rosrun rosbridge_server rosbridge_websocket   # or: roslaunch rosbridge_server rosbridge_websocket.launch
    rosrun tf2_web_republisher tf2_web_republisher   # required for --feedback pose

Then run this:
    python scripts/mpc_twist_ros_bridge.py --host <container-host> --port 9090
"""

from __future__ import annotations

import argparse
import base64
import signal
import time

import mujoco
import numpy as np
import roslibpy
from roslibpy.tf import TFClient
from scipy.spatial.transform import Rotation

from judo.app.structs import MujocoState
from judo.controller import make_controller
from judo.tasks.peg_insertion_twist import GOAL_OFFSET_QUAT_PEGLINK, PEG_TIP_SHAFT

# ----------------------------------------------------------------------------------------
# Carriage geometry constants (must match peg_insertion_twist.xml).
# qpos layout = [tx, ty, tz, rx, ry, rz] : 3 slide joints then 3 STACKED hinge joints
# (eef_wx -> eef_wy -> eef_wz), i.e. an intrinsic XYZ-Euler chain. At qpos=0 the carriage
# "eef" frame is world-aligned at EEF_HOME_POS.
EEF_HOME_POS = np.array([0.5, 0.0, 0.45])
EULER_SEQ = "XYZ"  # stacked-hinge order -> intrinsic XYZ

# Fixed transform from the eef carriage frame to the real robot's `peg_link` frame, used to
# recover the eef pose from a measured peg_link pose in --feedback pose. The rotation is
# GOAL_OFFSET_QUAT_PEGLINK (eef -> peg -> peg_link, == 180 deg about Y); the peg_link (tip)
# sits PEG_TIP_SHAFT below the eef origin along the eef -Z axis. Feeding the peg_link pose
# straight into the carriage (skipping this) flips it ~180 deg and the peg renders inverted.
_QPL = np.asarray(GOAL_OFFSET_QUAT_PEGLINK)  # wxyz
R_EEF_TO_PEGLINK = Rotation.from_quat([_QPL[1], _QPL[2], _QPL[3], _QPL[0]])
T_EEF_TO_PEGLINK = np.array([0.0, 0.0, -PEG_TIP_SHAFT])

# State (qpos/qvel) layout of peg_insertion_twist.xml: the movable hole carriage joints
# come FIRST (hole_tx, hole_ty, hole_tz, hole_yaw), then the 6-DOF eef carriage. The goal of
# the task IS the hole pose (the reward sites are rigid children of the hole body), so to
# move the goal we drive these three DOF via the planning state -- NOT the hole_x/y/yaw
# task config, which only takes effect through pre_sim_step and is never called by the
# mujoco.rollout backend used here.
HOLE_QPOS_SLICE = slice(0, 4)  # [hole_tx, hole_ty, hole_tz, hole_yaw]
EEF_QPOS_SLICE = slice(4, 10)  # [eef_tx, eef_ty, eef_tz, eef_wx, eef_wy, eef_wz]
HOLE_HOME_POS = np.array([0.5, 0.0, 0.2])  # nominal hole_base world origin (peg_insertion_twist.xml)


def _world_to_judo_rotation(rz180: bool) -> Rotation:
    """Rotation that maps ASSEMBLY-world vectors into JUDO-world vectors.

    judo's home tip sits at +x while the iiwa workspace is at -x (ee_max.x < 0), i.e. the
    two world frames differ by ~Rz(180). This is applied to the *outgoing twist* (both the
    linear and angular halves) and inversely to the *incoming pose*. VERIFY against the real
    TF between the judo model frame and the robot base before trusting --rz180.
    """
    return Rotation.from_euler("z", 180.0, degrees=True) if rz180 else Rotation.identity()


def _integrate_deadreckon(qpos: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
    """Advance the carriage qpos by integrating the commanded twist (small-dt approx).

    Translation integrates exactly; the stacked-Euler rotation is integrated as
    rx,ry,rz += w*dt, which is only first-order correct (the hinge rates equal true world
    omega solely at qpos_rot = 0). Fine for a feedforward demo; use --feedback pose for
    anything that reorients appreciably.
    """
    out = qpos.copy()
    out[:3] += twist[:3] * dt
    out[3:] += twist[3:] * dt
    return out


def _qpos_from_pose(pos_world: np.ndarray, quat_xyzw: np.ndarray, w2j: Rotation) -> np.ndarray:
    """Map a real peg_link world pose -> judo carriage qpos = [tx,ty,tz, rx,ry,rz].

    The ROS pose is of `peg_link`, but the carriage state is the `eef` frame, which differs
    by the fixed eef->peg_link transform (R_EEF_TO_PEGLINK, T_EEF_TO_PEGLINK). We first undo
    that to recover the eef pose, then bring it into judo-world, offset by the carriage home,
    and decompose the rotation into the stacked XYZ-Euler triplet. Skipping the eef<-peg_link
    step feeds the peg_link orientation straight into the hinges and flips the peg ~180 deg
    (it renders inverted). --rz180 must match the real judo<->robot-base TF.
    """
    # recover the eef pose from the measured peg_link pose (assembly-world)
    R_peglink = Rotation.from_quat(quat_xyzw)
    R_eef = R_peglink * R_EEF_TO_PEGLINK.inv()
    p_eef = pos_world - R_eef.apply(T_EEF_TO_PEGLINK)

    j2w = w2j.inv()  # judo-world <- assembly-world
    pos_judo = j2w.apply(p_eef)
    t = pos_judo - EEF_HOME_POS

    rxyz = (j2w * R_eef).as_euler(EULER_SEQ)  # stacked-hinge angles
    return np.concatenate([t, rxyz])


def _hole_qpos_from_goal(
    pos_world: np.ndarray,
    quat_xyzw: np.ndarray,
    w2j: Rotation,
    tilt_tol_deg: float = 15.0,
) -> np.ndarray:
    """Map a real ``world -> insertion_goal`` TF pose -> hole carriage qpos = [tx, ty, tz, yaw].

    The hole carriage is 4-DOF (x/y/z slide + yaw hinge), so we project the full goal pose
    onto it: take the translation relative to the nominal hole origin and the yaw of the
    goal frame's x-axis. Only roll/pitch (tilt off vertical) is not representable and is
    dropped (with a warning past ``tilt_tol_deg``). ``insertion_goal`` is taken to coincide
    with the hole origin, so its yaw maps straight onto hole_yaw with no GOAL_OFFSET_QUAT
    correction.

    Tilt is measured from the vertical *axis*, not the +z direction: insertion_goal usually
    has its z pointing down into the hole, which is perfectly planar for a yaw-only carriage,
    so a near-180-deg flip is NOT a tilt. Only z-axes that leave the vertical line warn.
    """
    j2w = w2j.inv()  # judo-world <- assembly-world
    pos_judo = j2w.apply(pos_world)
    t = pos_judo - HOLE_HOME_POS

    R = j2w * Rotation.from_quat(quat_xyzw)
    xaxis = R.apply([1.0, 0.0, 0.0])
    yaw = float(np.arctan2(xaxis[1], xaxis[0]))  # in-plane yaw, robust to small tilt

    zaxis = R.apply([0.0, 0.0, 1.0])
    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(zaxis[2]), 0.0, 1.0))))  # from vertical axis
    if tilt_deg > tilt_tol_deg and time.monotonic() - _hole_qpos_from_goal._last_warn > 1.0:  # type: ignore[attr-defined]
        print(f"[goal] WARN goal tilts {tilt_deg:.1f} deg off vertical; ignored (carriage is x,y,z,yaw only)")
        _hole_qpos_from_goal._last_warn = time.monotonic()  # type: ignore[attr-defined]

    return np.array([t[0], t[1], t[2], yaw])


_hole_qpos_from_goal._last_warn = 0.0  # type: ignore[attr-defined]  # throttle the tilt warning to 1/s


def _contact_pt_from_cloud(
    msg: dict,
    w2j: Rotation,
    min_spread: float = 0.05,
    frac: float = 0.5,
) -> np.ndarray | None:
    """Reduce a ``/contact_residuals`` PointCloud2 (x,y,z,intensity) to one judo-world hotspot.

    ``intensity`` is the per-point contact likelihood exp(-residual). When there is no real
    contact the publisher emits a (near) uniform field, so a flat intensity spread means "no
    hotspot" -> None. Otherwise we take the likelihood-weighted centroid of the brightest
    points (>= ``frac`` * max) as the contact location and rotate it into judo-world.
    """
    n = int(msg["width"]) * int(msg["height"])
    if n == 0:
        return None
    offs = {f["name"]: int(f["offset"]) for f in msg["fields"]}
    if not {"x", "y", "z", "intensity"} <= offs.keys():
        return None
    end = ">" if msg.get("is_bigendian") else "<"
    dt = np.dtype(
        {
            "names": ["x", "y", "z", "i"],
            "formats": [end + "f4"] * 4,
            "offsets": [offs["x"], offs["y"], offs["z"], offs["intensity"]],
            "itemsize": int(msg["point_step"]),
        }
    )
    pc = np.frombuffer(base64.b64decode(msg["data"]), dtype=dt, count=n)
    xyz = np.stack([pc["x"], pc["y"], pc["z"]], axis=1).astype(float)
    w = np.nan_to_num(pc["i"].astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    if float(w.max() - w.min()) < min_spread:  # uniform -> no meaningful contact
        return None
    mask = w >= frac * float(w.max())
    wm = w[mask]
    if wm.sum() <= 0.0:
        return None
    c_world = (wm[:, None] * xyz[mask]).sum(0) / wm.sum()  # assembly-world hotspot
    return w2j.inv().apply(c_world)  # -> judo-world


def main() -> None:
    """Run the MPC->twist bridge until interrupted."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="rosbridge_server host")
    p.add_argument("--port", type=int, default=9090, help="rosbridge_server websocket port")
    p.add_argument("--task", default="peg_insertion_twist", help="judo task name")
    p.add_argument("--optimizer", default="ps", choices=["ps", "cem", "mppi"], help="judo optimizer")
    p.add_argument("--ns", default="/yang", help="robot namespace used by the action servers")
    p.add_argument("--twist-topic", default="/twist_input", help="geometry_msgs/Twist sink the server reads")
    p.add_argument("--joint-topic", default=None, help="JointState topic (default: <ns>/joint_states)")
    p.add_argument("--fixed-frame", default="world", help="TF reference frame (for --feedback pose)")
    p.add_argument("--target-frame", default="peg_link", help="TF frame to track (for --feedback pose)")
    p.add_argument("--goal-frame", default="insertion_goal", help="TF frame giving the hole/insertion goal pose")
    p.add_argument("--goal-fixed-frame", default="world", help="TF reference frame for --goal-frame")
    p.add_argument(
        "--contact-topic",
        default="/contact_residuals",
        help="PointCloud2 (x,y,z,intensity) of contact likelihood; '' to disable contact avoidance",
    )
    p.add_argument(
        "--action-server",
        default="linear_planner",
        help="cmd_twist_action_server_name (relative to ns); resolves to <ns>/<name>",
    )
    p.add_argument("--action-type", default="iiwa_optas/TriggerCmdAction", help="actionlib action type")
    p.add_argument("--control-freq", type=float, default=25.0, help="MPC replan / publish rate [Hz]")
    p.add_argument("--feedback", default="deadreckon", choices=["deadreckon", "pose"], help="state source")
    p.add_argument("--rz180", action="store_true", help="apply Rz(180) judo<->assembly world transform")
    p.add_argument("--viz", action="store_true", help="launch a viser visualizer of the planning state + MPC traces")
    p.add_argument("--no-trigger", action="store_true", help="skip sending the action goal (publish twist only)")
    p.add_argument(
        "--no-contact-cost",
        action="store_true",
        help="disable the reward's contact-repulsion term (never pass the hotspot to the planner)",
    )
    args = p.parse_args()

    joint_topic = args.joint_topic or f"{args.ns}/joint_states"
    action_name = f"{args.ns}/{args.action_server}"
    w2j = _world_to_judo_rotation(args.rz180)
    dt = 1.0 / args.control_freq

    # ---- judo controller (CPU mujoco rollout backend per the task registration) ----------
    print(f"[judo] building controller: task={args.task} optimizer={args.optimizer}")
    ctrl = make_controller(args.task, args.optimizer)
    assert ctrl.nu == 6, f"expected a 6-DOF twist task, got nu={ctrl.nu}"
    # Tighten the predictive-sampling exploration noise for this bridge: the linear twist
    # ctrlrange is only +-0.01 m/s and the action normalizer is identity, so sample in raw
    # m/s with a small sigma to avoid saturating the clip (overrides the task default 0.03).
    if args.optimizer == "ps":
        ctrl.optimizer.config.sigma = 0.01
    # Tighten the lateral centering gate to a 2 mm permission zone (default 5 mm) so the
    # peg must align more precisely over the hole axis before the descent target opens up.
    ctrl.task.config.approach_tol = 0.002
    ctrl.reset()

    # ---- optional viser visualizer of the planning state + elite MPC traces --------------
    # Renders exactly what the planner sees: the carriage/hole at the qpos we feed it and the
    # best rollout traces. Useful for debugging "weird" motion (e.g. a mis-mapped goal yaw or
    # an rz180 mismatch) without standing up the full judo app.
    viz_model = None
    if args.viz:
        import viser  # noqa: PLC0415  # optional dep; only needed with --viz

        from judo.visualizers.model import ViserMjModel  # noqa: PLC0415

        viz_server = viser.ViserServer()
        viz_model = ViserMjModel(viz_server, ctrl.task.spec, geom_exclude_substring="collision")
        print("[viz] viser server up -> open the printed http URL in a browser")

    # ---- roslibpy connection -------------------------------------------------------------
    ros = roslibpy.Ros(host=args.host, port=args.port)
    ros.run()
    print(f"[ros] connected={ros.is_connected} -> {args.host}:{args.port}")

    twist_pub = roslibpy.Topic(ros, args.twist_topic, "geometry_msgs/Twist", queue_size=1)
    twist_pub.advertise()

    # latest robot feedback. roslibpy delivers callbacks on its own (reactor) thread, but we
    # only ever swap a single tuple reference here -- under CPython's GIL that write and the
    # loop's read are atomic, so no lock is needed.
    fb: dict[str, object] = {"pose": None, "goal": None, "contact": None}
    # pose/goal: (pos_world[3], quat_xyzw[4]) or None; contact: judo-world xy(z)[3] or None

    # ---- goal feedback: stream world -> insertion_goal so the planner tracks the real hole --
    goal_tf = TFClient(
        ros,
        fixed_frame=args.goal_fixed_frame,
        rate=args.control_freq,
        translation_threshold=0.0005,  # m
        angular_threshold=0.01,  # rad
    )

    def _on_goal(transform: dict) -> None:
        tr = transform["translation"]
        rot = transform["rotation"]
        quat = np.array([rot["x"], rot["y"], rot["z"], rot["w"]])
        # tf2_web_republisher can emit an uninitialized (zero-norm) transform before the
        # real one; drop it so the wait gate releases only on a valid goal.
        if np.linalg.norm(quat) < 1e-6:
            print(f"[goal] invalid pose returned (zero-norm quaternion {quat.tolist()}); ignoring")
            return
        fb["goal"] = (np.array([tr["x"], tr["y"], tr["z"]]), quat)

    goal_tf.subscribe(args.goal_frame, _on_goal)
    print(f"[ros] goal: hole pose <- TF {args.goal_fixed_frame} -> {args.goal_frame}")

    # ---- contact-avoidance feedback: reduce /contact_residuals to a single judo-world hotspot --
    if args.contact_topic:
        contact_sub = roslibpy.Topic(
            ros, args.contact_topic, "sensor_msgs/PointCloud2", queue_size=1, throttle_rate=int(1000 / args.control_freq)
        )

        def _on_contact(msg: dict) -> None:
            fb["contact"] = _contact_pt_from_cloud(msg, w2j)  # None when no hotspot

        contact_sub.subscribe(_on_contact)
        print(f"[ros] contact: hotspot <- {args.contact_topic} (lateral tip repulsion)")
    else:
        print("[ros] contact: disabled")

    if args.feedback == "pose":
        # tf2_web_republisher composes <fixed_frame> -> <target_frame> and streams it here.
        # Low thresholds so small insertion motions still trigger republished updates.
        tf_client = TFClient(
            ros,
            fixed_frame=args.fixed_frame,
            rate=args.control_freq,
            translation_threshold=0.0005,  # m
            angular_threshold=0.01,  # rad
        )

        def _on_tf(transform: dict) -> None:
            tr = transform["translation"]
            rot = transform["rotation"]
            fb["pose"] = (
                np.array([tr["x"], tr["y"], tr["z"]]),
                np.array([rot["x"], rot["y"], rot["z"], rot["w"]]),
            )

        tf_client.subscribe(args.target_frame, _on_tf)
        print(f"[ros] feedback: pose <- TF {args.fixed_frame} -> {args.target_frame} (closed loop)")
    else:
        print("[ros] feedback: deadreckon (open loop)")

    running = {"go": True}
    signal.signal(signal.SIGINT, lambda *_: running.__setitem__("go", False))

    # ---- wait for the goal before doing anything that commands the robot -----------------
    # The hole pose IS the task goal, so we block here -- not triggering the action server
    # and not streaming twists -- until insertion_goal is on TF. The robot stays idle until
    # a real goal exists.
    print(f"[ros] waiting for goal TF {args.goal_fixed_frame} -> {args.goal_frame} ...")
    while running["go"] and ros.is_connected and fb["goal"] is None:
        time.sleep(0.05)
    if not (running["go"] and ros.is_connected):
        print("[ros] interrupted before goal arrived; exiting")
        ros.terminate()
        return
    gpos, _ = fb["goal"]  # type: ignore[misc]
    print(f"[ros] goal acquired: insertion_goal pos(world)={np.array2string(gpos, precision=3)}")

    # ---- trigger the twist action server (selects the mux, starts its timer) -------------
    # The installed roslibpy build has no `actionlib` submodule, so instead of an
    # ActionClient we publish straight to the SimpleActionServer's `<server>/goal` topic.
    # The ActionGoal wrapper type is the action type + "Goal"; rosbridge fills the header
    # and (empty) goal_id defaults, and we stamp a unique goal_id.id so actionlib tracks it.
    if not args.no_trigger:
        goal_topic = roslibpy.Topic(ros, f"{action_name}/goal", f"{args.action_type}Goal", queue_size=1)
        goal_topic.advertise()
        time.sleep(0.3)  # let the advertise register before publishing
        # NB: in TriggerCmd.action `bool trigger_on = True` is a CONSTANT, not a field, so
        # the goal message is empty -- sending it is itself the trigger (trigger_on is
        # always True by definition, which is what the server reads).
        goal_topic.publish(
            roslibpy.Message(
                {
                    "goal_id": {"stamp": {"secs": 0, "nsecs": 0}, "id": f"mpc_bridge_{int(time.time())}"},
                    "goal": {},
                }
            )
        )
        print(f"[ros] published TriggerCmd(trigger_on=True) -> {action_name}/goal")
        time.sleep(0.5)  # let the mux switch settle before streaming

    # ---- control loop --------------------------------------------------------------------
    eef_qpos = np.zeros(6)  # [tx,ty,tz, rx,ry,rz] of the peg carriage
    eef_qvel = np.zeros(6)
    prev_eef_qpos = None  # for finite-difference velocity in pose mode
    # goal (hole) pose: seeded from the goal TF acquired above, refreshed every loop
    hole_qpos = _hole_qpos_from_goal(gpos, fb["goal"][1], w2j)  # type: ignore[index]
    t = 0.0
    n = 0
    print("[loop] streaming twists (Ctrl-C to stop)")
    while running["go"] and ros.is_connected:
        loop_start = time.perf_counter()

        # 0) update the goal: drive the hole carriage from the real insertion_goal TF
        goal = fb["goal"]  # atomic single-reference read
        if goal is not None:
            hole_qpos = _hole_qpos_from_goal(goal[0], goal[1], w2j)

        # 1) set judo's eef planning state from the chosen feedback source
        if args.feedback == "pose":
            pose = fb["pose"]  # atomic single-reference read
            if pose is not None:
                new_eef_qpos = _qpos_from_pose(pose[0], pose[1], w2j)
                # rough body velocity as a planning prior: exact for translation, first-order
                # for the stacked-Euler angles; zero on the first sample to avoid a spike.
                eef_qvel = np.zeros(6) if prev_eef_qpos is None else (new_eef_qpos - prev_eef_qpos) / dt
                eef_qpos = new_eef_qpos
                prev_eef_qpos = new_eef_qpos

        # assemble the full model state: hole carriage (the goal) then the eef carriage. The
        # hole has no actuator and zero velocity, so it stays fixed across each rollout's
        # horizon -- the planner sees the reward sites at the commanded goal pose.
        full_qpos = np.concatenate([hole_qpos, eef_qpos])
        full_qvel = np.concatenate([np.zeros(4), eef_qvel])
        # pass the latest contact hotspot (judo-world) to the reward's lateral repulsion term
        # (--no-contact-cost disables the term by never forwarding the hotspot)
        contact_pt = None if args.no_contact_cost else fb["contact"]  # atomic single-reference read; None when no hotspot
        sim_metadata = {} if contact_pt is None else {"contact_pt": contact_pt}
        state = MujocoState(
            time=t,
            qpos=full_qpos,
            qvel=full_qvel,
            mocap_pos=np.zeros((0, 3)),
            mocap_quat=np.zeros((0, 4)),
            sim_metadata=sim_metadata,
        )
        ctrl.update_states(state)

        # 2) replan and read the current best twist
        ctrl.update_action()
        twist = np.asarray(ctrl.action(t), dtype=float)  # [vx,vy,vz,wx,wy,wz] in judo-world

        # 2b) mirror the planning state + elite traces into the visualizer (if enabled)
        if viz_model is not None:
            ctrl.task.data.qpos[:] = full_qpos
            ctrl.task.data.qvel[:] = full_qvel
            mujoco.mj_forward(ctrl.task.model, ctrl.task.data)
            viz_model.set_data(ctrl.task.data)
            viz_model.set_traces(ctrl.traces, ctrl.all_traces_rollout_size)

        # 3) judo-world -> assembly-world, then publish
        v = w2j.apply(twist[:3])
        w = w2j.apply(twist[3:])
        twist_pub.publish(
            roslibpy.Message(
                {
                    "linear": {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])},
                    "angular": {"x": float(w[0]), "y": float(w[1]), "z": float(w[2])},
                }
            )
        )

        # 4) advance judo's internal eef state (open-loop) when dead-reckoning
        if args.feedback == "deadreckon":
            eef_qpos = _integrate_deadreckon(eef_qpos, twist, dt)
            eef_qvel = twist
        t += dt

        if n % int(max(args.control_freq, 1)) == 0:
            cp = "none" if contact_pt is None else np.array2string(np.asarray(contact_pt)[:2], precision=3)
            print(f"[loop] t={t:6.2f}s  twist(judo)={np.array2string(twist, precision=3)}  contact(xy)={cp}")
        n += 1

        # 5) hold the loop rate
        sleep = dt - (time.perf_counter() - loop_start)
        if sleep > 0:
            time.sleep(sleep)

    print("[loop] stopping; publishing zero twist and closing")
    twist_pub.publish(
        roslibpy.Message({"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}})
    )
    time.sleep(0.1)
    twist_pub.unadvertise()
    ros.terminate()


if __name__ == "__main__":
    main()
