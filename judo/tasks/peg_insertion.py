# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
"""Sampling-based MPC peg-insertion task (FR3 + arch peg + arch hole).

The arch peg (``assembly_description/urdf/arch_peg.urdf``) is rigidly welded to the
FR3 hand (mirroring ``iiwa_arch_peg.dmd.yaml``, which welds the peg to the EE), and
the arch hole is placed at a *known, fixed pose* on a pedestal. MuJoCo collides
meshes as their convex hull, so the hole uses the 21 COACD convex pieces shipped in
``assembly_description`` while the peg's convex hull is its own collision geom.

The hole pose is "known" to the robot because the reward reads world-frame target
sites that are rigidly attached to the hole body -- move the hole body and the whole
task tracks it automatically.
"""

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig

XML_PATH = str(MODEL_PATH / "xml/peg_insertion.xml")

# arch peg meshes (assembly_description). Long shaft is along the peg's local -Y;
# tip is at y = -0.15, top of shaft near y = +0.03.
# The full mesh is used for VISUALS only; MuJoCo would collide it as its convex hull
# (~1.9x the true volume), which fills the arch's concavity and blocks insertion.
# For COLLISION we use the COACD convex decomposition (18 pieces), each collided as
# its own convex hull -- their union faithfully approximates the concave peg.
PEG_VIS_MESH = "/home/slmc/workspace/assembly_description/urdf/meshes/arch_peg.obj"
PEG_COLL_DIR = "/home/slmc/workspace/assembly_description/urdf/meshes/arch_peg"
PEG_NUM_COLL = 18  # arch_peg_convex_coll_000.obj ... _017.obj

# iiwa14 home configuration: flange points straight down with the peg tip hovering
# at world ~(0.5, 0, 0.30), just above the hole entrance on the (short) pedestal.
QPOS_HOME = np.array([0.0, 0.34579, 0.0, -1.68936, 0.0, 1.10644, 0.0])  # arm (7)

# Rigid transform of the peg in the wrist (iiwa link7) frame. link7 +z is the approach
# axis (points "out" of the flange); R_x(-90deg) maps the peg's local -Y onto link7 +z
# so the peg shaft points along the approach direction. Offset to the flange face
# (z=0.045, the iiwa14 attachment_site, where a tool would mount).
PEG_POS_IN_WRIST = [0.0, 0.0, 0.045]
PEG_QUAT_IN_WRIST = [0.70710678, -0.70710678, 0.0, 0.0]  # wxyz, R_x(-90deg)

# size of the orientation-triad axes drawn on the peg tip and hole frames
TRIAD_LEN = 0.06
TRIAD_RAD = 0.0035


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (possibly batched) wxyz quaternions. Shapes broadcast."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


# Goal orientation for the peg tip, expressed as an offset applied in the hole's local
# frame: q_goal = q_hole (x) GOAL_OFFSET_QUAT. The goal frame is the hole frame rotated
# +90 deg about its (local) X axis and then +90 deg about the resulting Y axis -- this
# maps the hole bore (+Z) onto the peg shaft (-Y) with the arch correctly clocked.
# Intrinsic sequence => composed on the right: R_x(90) then R_y(90).
_R_X90 = np.array([0.70710678, 0.70710678, 0.0, 0.0])  # wxyz, R_x(+90deg)
_R_Y90 = np.array([0.70710678, 0.0, 0.70710678, 0.0])  # wxyz, R_y(+90deg)
GOAL_OFFSET_QUAT = _quat_mul(_R_X90, _R_Y90)  # == (0.5, 0.5, 0.5, 0.5)


def _add_frame_triad(body: Any, prefix: str, origin: list[float], length: float = TRIAD_LEN) -> None:
    """Add three visual-only cylinders forming an RGB axis triad at ``origin``.

    x->red, y->green, z->blue, expressed in ``body``'s local frame, so the triad
    rotates with the body and shows its orientation in the viewer (which renders
    geoms, not site frames).
    """
    ox, oy, oz = origin
    axes = [
        ("x", [length, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]),
        ("y", [0.0, length, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ("z", [0.0, 0.0, length], [0.0, 0.0, 1.0, 1.0]),
    ]
    for name, d, rgba in axes:
        body.add_geom(
            name=f"{prefix}_{name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=[ox, oy, oz, ox + d[0], oy + d[1], oz + d[2]],
            size=[TRIAD_RAD, 0.0, 0.0],
            contype=0,
            conaffinity=0,
            rgba=rgba,
        )


@slider("w_lateral", 0.0, 50.0, 0.5)
@slider("w_axial", 0.0, 50.0, 0.5)
@slider("w_align", 0.0, 50.0, 0.5)
@slider("w_qvel", 0.0, 1.0, 0.005)
@slider("w_nominal", 0.0, 10.0, 0.05)
@slider("approach_tol", 0.001, 0.05, 0.001)
@dataclass
class PegInsertionConfig(TaskConfig):
    """Reward configuration for the peg-insertion task."""

    w_lateral: float = 20.0  # align the peg over the hole (lateral error to hole axis)
    w_axial: float = 8.0  # drive the tip toward the (gated) axial target
    w_align: float = 10.0  # full SO(3) alignment of the peg-tip frame with the goal frame
    w_qvel: float = 0.02  # damp joint velocities for smoother motions
    w_nominal: float = 8.0  # keep the arm near a nominal posture (null-space regularizer)
    # Lateral centering tolerance (m) that gates the axial descent: the peg only starts
    # descending once it is centered over the hole axis to within ~this distance.
    # Smaller => the peg must line up more precisely at the entrance before going down.
    approach_tol: float = 0.01


class PegInsertion(Task[PegInsertionConfig]):
    """Peg-in-hole insertion with a rigidly-held arch peg."""

    name: str = "peg_insertion"
    config_t: type[PegInsertionConfig] = PegInsertionConfig

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Build the model (peg is welded onto the hand in ``_process_spec``)."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)

        # position-actuator targets for the home pose (7 arm joints)
        self.reset_command = QPOS_HOME.copy()

        # nominal posture for the null-space regularizer (default: the home config)
        self.q_nominal = QPOS_HOME.copy()

        # sensor start indices (each framepos sensor is 3-dim, each framequat is 4-dim)
        self.tip_adr = self.get_sensor_start_index("tip_pos")
        self.base_adr = self.get_sensor_start_index("base_pos")
        self.entrance_adr = self.get_sensor_start_index("entrance_pos")
        self.bottom_adr = self.get_sensor_start_index("bottom_pos")
        self.tip_quat_adr = self.get_sensor_start_index("tip_quat")
        self.hole_quat_adr = self.get_sensor_start_index("hole_quat")

        self.reset()

    def _process_spec(self) -> None:
        """Weld the arch peg onto the FR3 wrist (fr3_link7) and add task sensors."""
        spec = self.spec

        # full mesh asset for the peg visual; convex pieces for collision
        spec.add_mesh(name="arch_peg_vis", file=PEG_VIS_MESH)
        for i in range(PEG_NUM_COLL):
            spec.add_mesh(name=f"peg_coll_{i}", file=f"{PEG_COLL_DIR}/arch_peg_convex_coll_{i:03d}.obj")

        # rigid child body of the wrist => kinematic weld, no extra DOF
        wrist = spec.body("link7")
        peg = wrist.add_body(name="peg", pos=PEG_POS_IN_WRIST, quat=PEG_QUAT_IN_WRIST)

        # visual geom: full arch mesh, no collisions (contype/conaffinity = 0)
        peg.add_geom(
            name="peg_vis",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="arch_peg_vis",
            contype=0,
            conaffinity=0,
            rgba=[0.49, 0.27, 0.27, 1.0],
        )
        # collision geoms: COACD convex pieces. The peg is a "robot" collider
        # (contype=1, conaffinity=2), so it collides with the "obstacle" geoms
        # (hole + table = contype=2, conaffinity=1) but not with the arm it is
        # welded to, since (1 & 2) == 0.
        for i in range(PEG_NUM_COLL):
            peg.add_geom(
                name=f"peg_collision_{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"peg_coll_{i}",
                group=3,  # collision group (hidden by default, like the hole pieces)
                contype=1,
                conaffinity=2,
                rgba=[0.49, 0.27, 0.27, 1.0],
            )
        # two sites along the shaft define the insertion axis (orientation-free)
        peg.add_site(name="peg_tip", pos=[0.0, -0.15, 0.0], size=[0.004, 0.004, 0.004], rgba=[1, 1, 0, 0.9])
        peg.add_site(name="peg_base", pos=[0.0, 0.03, 0.0], size=[0.004, 0.004, 0.004], rgba=[1, 0.5, 0, 0.9])

        # visual-only marker (the judo viewer renders geoms, not sites): RED sphere at
        # the peg_tip -- this is the robot point the cost drives toward the (green) goal.
        peg.add_geom(
            name="peg_tip_marker",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[0.0, -0.15, 0.0],
            size=[0.01, 0.0, 0.0],
            contype=0,
            conaffinity=0,
            rgba=[1, 0, 0, 1.0],
        )
        # RGB orientation triad at the peg_tip frame (x=red, y=green, z=blue). Visual
        # only. The peg shaft / insertion axis runs along the peg's local -Y (toward
        # the tip), so the green axis marks the shaft direction.
        _add_frame_triad(peg, "peg_tip_axis", origin=[0.0, -0.15, 0.0])

        # Goal-orientation frame: a static child of the hole, offset by GOAL_OFFSET_QUAT.
        # The peg-tip frame is driven to match this frame's orientation. The triad makes
        # the target orientation visible in the viewer (compare against the peg-tip triad).
        hole = spec.body("hole")
        goal = hole.add_body(name="goal_frame", pos=[0.0, 0.0, 0.05], quat=list(GOAL_OFFSET_QUAT))
        goal.add_site(name="goal_orient", pos=[0.0, 0.0, 0.0], size=[0.004, 0.004, 0.004], rgba=[1, 1, 1, 0.9])
        _add_frame_triad(goal, "goal_axis", origin=[0.0, 0.0, 0.0])

        # world-frame position sensors for the reward. The "trace_" sensor is picked up
        # by the visualizer, which draws the peg tip's path for each elite rollout.
        for sname, objname in [
            ("tip_pos", "peg_tip"),
            ("base_pos", "peg_base"),
            ("entrance_pos", "hole_entrance"),
            ("bottom_pos", "hole_bottom"),
            ("trace_peg_tip", "peg_tip"),
        ]:
            spec.add_sensor(
                name=sname,
                type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                objname=objname,
            )

        # orientation (quaternion) sensors for the full SO(3) alignment cost
        for sname, objname in [("tip_quat", "peg_tip"), ("hole_quat", "hole_entrance")]:
            spec.add_sensor(
                name=sname,
                type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                objname=objname,
            )

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward: staged peg insertion -- approach the entrance, then descend.

        Terms: lateral alignment to the hole axis, a *gated* axial-depth target (hover at
        the entrance until centered, then descend to the bottom), full SO(3) orientation
        alignment, velocity damping, and a nominal-posture regularizer. All quantities are
        derived from world-frame sites, so the (known) hole pose is encoded directly by the
        sensor placement on the hole body.
        """
        tip = sensors[..., self.tip_adr : self.tip_adr + 3]  # (R, T, 3)
        entrance = sensors[..., self.entrance_adr : self.entrance_adr + 3]
        bottom = sensors[..., self.bottom_adr : self.bottom_adr + 3]

        # hole axis (points up/out of the hole)
        hole_axis = entrance - bottom
        hole_axis = hole_axis / (np.linalg.norm(hole_axis, axis=-1, keepdims=True) + 1e-9)

        # decompose tip->bottom error into lateral (off-axis) and axial (depth) parts
        vec = tip - bottom  # (R, T, 3)
        axial = np.sum(vec * hole_axis, axis=-1)  # (R, T) signed depth above bottom
        lateral = vec - axial[..., None] * hole_axis
        lateral_dist = np.linalg.norm(lateral, axis=-1)  # (R, T)

        # Staged approach: first reach the entrance (centered over the axis), then
        # descend. The axial *target* is gated by lateral alignment -- when off-axis the
        # target is the entrance height (hover at the top); once centered it drops to the
        # bottom (full insertion). descend_gate -> 1 when centered, -> 0 when off-axis.
        axial_entrance = np.sum((entrance - bottom) * hole_axis, axis=-1)  # (R, T) entrance height above bottom
        approach_tol = max(self.config.approach_tol, 1e-4)
        descend_gate = np.exp(-((lateral_dist / approach_tol) ** 2))  # (R, T) in [0, 1]
        axial_target = (1.0 - descend_gate) * axial_entrance  # entrance height -> 0 (bottom) as it centers
        axial_err = np.abs(axial - axial_target)  # (R, T)

        # full SO(3) orientation error: geodesic angle between the peg-tip frame and the
        # goal frame (hole frame (x) GOAL_OFFSET_QUAT). In [0, pi].
        tip_quat = sensors[..., self.tip_quat_adr : self.tip_quat_adr + 4]  # (R, T, 4)
        hole_quat = sensors[..., self.hole_quat_adr : self.hole_quat_adr + 4]
        goal_quat = _quat_mul(hole_quat, GOAL_OFFSET_QUAT)  # (R, T, 4)
        dot = np.abs(np.sum(tip_quat * goal_quat, axis=-1))  # (R, T), |<q_tip, q_goal>|
        align_err = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))  # (R, T) geodesic angle

        qvel = states[..., self.model.nq : self.model.nq + self.model.nv]
        qvel_norm = np.linalg.norm(qvel, axis=-1)  # (R, T)

        # keep the arm near a nominal posture (null-space regularizer)
        qpos = states[..., : self.model.nq]  # (R, T, nq); nq == 7 arm joints
        nominal_err = np.linalg.norm(qpos - self.q_nominal, axis=-1)  # (R, T)

        c = self.config
        cost = (
            c.w_lateral * lateral_dist
            + c.w_axial * axial_err
            + c.w_align * align_err
            + c.w_qvel * qvel_norm
            + c.w_nominal * nominal_err
        )
        return -cost.sum(axis=-1)  # (R,)

    def optimizer_warm_start(self) -> np.ndarray:
        """Initial control for the MPC plan.

        These are position-actuator targets, so the default base implementation
        (zeros) commands every joint to 0 rad -- the iiwa's "straight up" pose -- which
        makes the peg shoot to the top before MPC optimizes. Warm-start at the home
        configuration so the plan holds the home pose until insertion is planned.
        """
        return QPOS_HOME.copy()

    def reset(self) -> None:
        """Reset arm to the home configuration above the hole."""
        self.data.qpos[:] = QPOS_HOME
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.reset_command
        mujoco.mj_forward(self.model, self.data)
