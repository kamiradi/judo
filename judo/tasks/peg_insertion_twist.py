# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
"""Sampling-based MPC peg-insertion task with END-EFFECTOR TWIST control.

This is the twist-controlled twin of ``peg_insertion.py``. There is NO robot arm:
the peg is mounted on a 6-DOF Cartesian carriage (``peg_insertion_twist.xml``) and the
control inputs are the peg's end-effector twist -- a world-frame linear and angular
velocity ``[vx, vy, vz, wx, wy, wz]`` applied by velocity actuators. The reward is
identical to the arm task (it reads world-frame sites), so the only differences are the
model, where the peg is attached, and the control format (6-dim twist vs. 7 arm joint
targets). The nominal-posture regularizer is dropped because here ``qpos`` *is* the
end-effector pose and penalizing it would fight the insertion.

The peg/hole *shape* (arch, rectangle, ...) is selected per task via ``GEOMETRY``: the
model XML is shape-agnostic (it only defines the carriage, table, and the hole body's
reward sites/markers), and ``_process_spec`` loads the chosen visual + convex
collision meshes for both the peg and the hole. All shapes in ``assembly_description``
share the same frames/extents, so the site offsets and the goal orientation are common.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.descriptions import get_description_path
from judo.tasks.peg_insertion import GOAL_OFFSET_QUAT, _add_frame_triad, _quat_mul

XML_PATH = str(MODEL_PATH / "xml/peg_insertion_twist.xml")
# Peg/hole meshes live in the standalone assembly_description repo, in a shared local cache.
# This is a cache LOOKUP, not a fetch -- warm the cache once with `python -m judo.descriptions`
# (importing this module raises if it is cold), so a sweep's N workers don't each re-clone.
# Tracks the latest commit unless ROBOT_DESCRIPTION_COMMIT pins one.
MESH_ROOT = f"{get_description_path('assembly_description')}/urdf/meshes"

# Home twist: zero velocity (the peg holds its starting pose hovering over the hole).
NU = 6  # [vx, vy, vz, wx, wy, wz]
TWIST_HOME = np.zeros(NU)

# Rigid transform of the peg in the carriage's "eef" frame. At qpos=0 the eef frame is
# world-aligned at (0.5, 0, 0.45); GOAL_OFFSET_QUAT (== (0.5, 0.5, 0.5, 0.5)) maps the
# peg's local -Y (shaft/tip) onto world -Z, so the peg hangs straight down with its tip
# at world (0.5, 0, 0.30) -- the same home pose the arm produced.
PEG_POS_IN_EEF = [0.0, 0.0, 0.0]
PEG_QUAT_IN_EEF = list(GOAL_OFFSET_QUAT)  # wxyz

# Peg-tip offset along the shaft (peg-local -Y), to match the REAL robot's `peg_link` frame.
# From rectangle_peg.xacro (the peg assembly.urdf actually loads): peg_link sits at
# adapter_height + tool_height = 0.005 + 0.175 = 0.18 m from peg_link_base along its z, and
# that point is the peg tip. judo mounts the visual mesh 0.0275 m up that same axis, so in
# the peg body's local frame the tip is 0.18 - 0.0275 = 0.1525 m out (vs the old 0.15).
PEG_TIP_SHAFT = 0.1525

# Orientation of the REAL robot's `peg_link` frame in the judo peg-body (mesh) frame. From
# rectangle_peg.xacro the visual mesh is mounted on peg_link_base with rpy(-90,0,90); judo
# attaches that same mesh with no rotation and calls its frame the peg body, so peg_link is
# the peg body rotated by rpy(90,90,0). We then clock it an extra 180 deg about the peg_link
# z-axis (the insertion axis) to match the robot's reported peg_link orientation. Placing the
# peg_tip frame on a child body with this quat makes judo's tip frame coincide with peg_link.
_PEG_BODY_TO_PEG_LINK_BASE = np.array([0.5, 0.5, 0.5, -0.5])  # wxyz, from the xacro mesh mount
_RZ180 = np.array([0.0, 0.0, 0.0, 1.0])  # wxyz, 180 deg about z
PEG_LINK_QUAT_IN_PEG = list(_quat_mul(_PEG_BODY_TO_PEG_LINK_BASE, _RZ180))  # -> [0.5, 0.5, -0.5, 0.5]
# Goal orientation re-expressed in the peg_link convention (== GOAL_OFFSET_QUAT (x) the above)
# so the SO(3) alignment cost is unchanged in behavior now that the tip lives in peg_link.
GOAL_OFFSET_QUAT_PEGLINK = _quat_mul(np.asarray(GOAL_OFFSET_QUAT), np.asarray(PEG_LINK_QUAT_IN_PEG))

# Success thresholds (see ``PegInsertionTwist.success``). The peg counts as inserted once
# its tip is at least this fraction of the way from the hole entrance down to the bottom,
# AND it is laterally centered on the hole axis to within this tolerance (so a tip that
# merely brushes past the rim at depth does not register as success).
SUCCESS_DEPTH_FRAC = 0.9  # fraction of the entrance->bottom span the tip must pass
SUCCESS_LATERAL_TOL = 0.004  # m; max off-axis distance of the tip while inserted
# The toothed geometries mesh their teeth well before the tip nears the bottom, so a 0.9
# depth gate reports failure for insertions that are mechanically seated. They override the
# depth fraction (via ``PegInsertionTwist.success_depth_frac``); the lateral gate is unchanged.
TEETH_SUCCESS_DEPTH_FRAC = 0.4


@dataclass(frozen=True)
class PegHoleGeometry:
    """Mesh set for one peg/hole shape.

    The visual fields are absolute paths to the full (concave) meshes used for rendering
    only; the collision fields describe the COACD convex decomposition (collided as the
    union of convex hulls). All shapes share frames/extents (peg tip at local y=-0.15,
    hole entrance at z=0.05), so no per-shape offsets are needed.
    """

    name: str
    peg_vis: str  # absolute path to the visual peg mesh
    peg_coll_dir: str  # directory holding the peg convex pieces
    peg_coll_prefix: str  # filename stem, e.g. "arch_peg_convex_coll"
    peg_num_coll: int  # number of peg convex pieces
    hole_vis: str  # absolute path to the visual hole mesh
    hole_coll_dir: str  # directory holding the hole convex pieces
    hole_coll_prefix: str  # filename stem, e.g. "arch_hole_convex_coll"
    hole_num_coll: int  # number of hole convex pieces
    # Per-axis scale applied to the peg COLLISION meshes only (the visual mesh and the
    # reward sites are untouched). The peg shaft runs along peg-local Y, so a value like
    # (s, 1, s) thins the cross-section without shortening the shaft. The COACD convex
    # hulls inflate the peg ~1 mm past its true surface; for a near-zero-clearance fit
    # this makes the collider wider than the hole opening and the peg jams at the
    # entrance. Shrinking the cross-section restores the clearance needed to insert.
    peg_coll_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


ARCH_GEOMETRY = PegHoleGeometry(
    name="arch",
    peg_vis=f"{MESH_ROOT}/arch_peg.obj",
    peg_coll_dir=f"{MESH_ROOT}/arch_peg",
    peg_coll_prefix="arch_peg_convex_coll",
    # 61 = the 0.02-threshold decomposition from commit 179df3d (was 18). Undercounting here is
    # silent -- only the first N hulls get attached, leaving the peg partly non-colliding.
    peg_num_coll=61,
    # Render the thread-free arch hole: the threads are visual clutter and the collider
    # (arch_hole_1) does not model them anyway.
    hole_vis=f"{MESH_ROOT}/arch_hole_no_threads.obj",
    hole_coll_dir=f"{MESH_ROOT}/arch_hole_1",
    hole_coll_prefix="arch_hole_convex_coll",
    # Piece count tracks the mesh repo: re-check it after pulling assembly_description (it has
    # been 21 / 10 / 454 / 15 / 67 across re-runs of COACD on the arch hole).
    # 17 = commit e9842e8 ("arch no threads hole dims increased").
    hole_num_coll=17,
    # hole_coll_dir=f"{MESH_ROOT}/arch_hole_1mm",
    # hole_coll_prefix="arch_hole_1mm_convex_coll",
    # hole_num_coll=26,
)

RECTANGLE_GEOMETRY = PegHoleGeometry(
    name="rectangle",
    # NB: visual files are spelled "rectangular_*" while the collision dirs are "rectangle_*".
    peg_vis=f"{MESH_ROOT}/rectangular_peg.obj",
    peg_coll_dir=f"{MESH_ROOT}/rectangle_peg",
    peg_coll_prefix="rectangle_peg_convex_coll",
    peg_num_coll=17,
    hole_vis=f"{MESH_ROOT}/rectangular_hole.obj",
    hole_coll_dir=f"{MESH_ROOT}/rectangle_hole",
    hole_coll_prefix="rectangle_hole_convex_coll",
    hole_num_coll=9,
    # The COACD convex hulls inflate the rectangular peg ~1 mm/side past its true (already
    # near-zero-clearance) shape, so the collider is wider than the hole opening and the
    # peg jams at the entrance. Thin the cross-section (peg-local X/Z) to restore
    # clearance; keep the shaft length (Y) so the peg-tip site still marks the real tip.
    peg_coll_scale=(0.90, 1.0, 0.90),
)

ELLIPSE_GEOMETRY = PegHoleGeometry(
    name="ellipse",
    peg_vis=f"{MESH_ROOT}/ellipse_peg.obj",
    peg_coll_dir=f"{MESH_ROOT}/ellipse_peg",
    peg_coll_prefix="ellipse_peg_convex_coll",
    peg_num_coll=16,
    hole_vis=f"{MESH_ROOT}/ellipse_hole.obj",
    hole_coll_dir=f"{MESH_ROOT}/ellipse_hole",
    hole_coll_prefix="ellipse_hole_convex_coll",
    hole_num_coll=18,
    peg_coll_scale=(0.90, 1.0, 0.90),
)

ELLIPSE_TEETH_GEOMETRY = PegHoleGeometry(
    name="ellipse_teeth",
    peg_vis=f"{MESH_ROOT}/ellipse_peg_teeth.obj",
    peg_coll_dir=f"{MESH_ROOT}/ellipse_peg_teeth",
    peg_coll_prefix="ellipse_peg_teeth_convex_coll",
    peg_num_coll=17,
    hole_vis=f"{MESH_ROOT}/ellipse_hole_teeth.obj",
    hole_coll_dir=f"{MESH_ROOT}/ellipse_hole_teeth",
    hole_coll_prefix="ellipse_hole_teeth_convex_coll",
    hole_num_coll=29,
    peg_coll_scale=(0.90, 1.0, 0.90),
)

RECTANGLE_TEETH_GEOMETRY = PegHoleGeometry(
    name="rectangle_teeth",
    peg_vis=f"{MESH_ROOT}/rectangle_peg_teeth.obj",
    peg_coll_dir=f"{MESH_ROOT}/rectangle_peg_teeth",
    peg_coll_prefix="rectangle_peg_teeth_convex_coll",
    peg_num_coll=20,
    hole_vis=f"{MESH_ROOT}/rectangle_hole_teeth.obj",
    hole_coll_dir=f"{MESH_ROOT}/rectangle_hole_teeth",
    hole_coll_prefix="rectangle_hole_teeth_convex_coll",
    hole_num_coll=22,
    # Thinned less than the toothless shapes: the teeth stick out laterally, and mesh scale
    # is about the mesh origin, so the thin both shrinks each tooth and pulls it toward the
    # shaft axis. At 0.90 that loosens the tooth engagement well past the ~1 mm hull
    # inflation it corrects for; 0.95 keeps entrance clearance with the teeth still meshing.
    peg_coll_scale=(0.95, 1.0, 0.95),
)


@slider("w_lateral", 0.0, 50.0, 0.5)
@slider("w_axial", 0.0, 50.0, 0.5)
@slider("w_align", 0.0, 50.0, 0.5)
@slider("w_tilt", 0.0, 200.0, 1.0)
@slider("tilt_limit_deg", 0.0, 90.0, 1.0)
@slider("w_qvel", 0.0, 1.0, 0.005)
@slider("approach_tol", 0.001, 0.2, 0.001)
@slider("w_contact", 0.0, 50.0, 0.5)
@slider("contact_radius", 0.002, 0.05, 0.001)
@slider("hole_x", -0.2, 0.2, 0.005)
@slider("hole_y", -0.2, 0.2, 0.005)
@slider("hole_z", -0.2, 0.2, 0.005)
@slider("hole_yaw", -3.14159, 3.14159, 0.02)
@dataclass
class PegInsertionTwistConfig(TaskConfig):
    """Reward configuration for the twist-controlled peg-insertion task."""

    w_lateral: float = 20.0  # align the peg over the hole (lateral error to hole axis)
    w_axial: float = 8.0  # drive the tip toward the (gated) axial target
    w_align: float = 10.0  # full SO(3) alignment of the peg-tip frame with the goal frame
    # One-sided (hinge) penalty on the peg's tilt away from vertical. Zero while the peg
    # stays within +-tilt_limit_deg about world x and world y; grows quadratically past it.
    # Acts as a soft constraint, so set it high relative to the other weights.
    w_tilt: float = 80.0
    tilt_limit_deg: float = 30.0  # max allowed tilt (deg) about world x and world y
    w_qvel: float = 0.02  # damp the commanded twist for smoother motions
    # Lateral centering tolerance (m) that gates the axial descent: the peg only starts
    # descending once it is centered over the hole axis to within ~this distance.
    approach_tol: float = 0.005
    # Contact-avoidance: repel the peg tip (laterally) from an estimated contact hotspot
    # passed in via system_metadata["contact_pt"] (judo-world xy). Off when no hotspot is
    # provided or w_contact == 0. The penalty is a Gaussian bump exp(-0.5*(d_xy/r)^2) that
    # peaks at the hotspot, so keep w_contact small and contact_radius tight or it will
    # fight the hole-centering costs and stall the insertion.
    # DEFAULT 0: the contact signal now shapes the sampling KERNEL (anisotropic covariance)
    # instead of the reward -- adding it as a cost too gave the robot conflicting signals.
    # Set >0 only to deliberately re-enable the reward-space repulsion term.
    w_contact: float = 0.0  # weight of the lateral contact-repulsion bump
    contact_radius: float = 0.006  # m; Gaussian radius of the repulsion (sigma)
    # Runtime pose of the hole carriage, as offsets from the nominal hole pose
    # (world (0.5, 0, 0.2)). Driven by GUI sliders and written into the hole carriage
    # joints each sim step; since those joints are part of the state, the planner's
    # rollouts track the moved hole automatically.
    hole_x: float = 0.0  # lateral offset along world +x (m)
    hole_y: float = 0.0  # lateral offset along world +y (m)
    hole_z: float = 0.0  # vertical offset along world +z (m)
    hole_yaw: float = 0.0  # rotation about the hole's vertical axis (rad)


class PegInsertionTwist(Task[PegInsertionTwistConfig]):
    """Peg-in-hole insertion where the peg is driven directly by an end-effector twist."""

    name: str = "peg_insertion_twist"
    # Depth gate used by ``success``; per-geometry so the toothed shapes can seat shallower.
    success_depth_frac: float = SUCCESS_DEPTH_FRAC
    config_t: type[PegInsertionTwistConfig] = PegInsertionTwistConfig
    # peg/hole shape used to build the model; override in subclasses for other shapes.
    geometry: PegHoleGeometry = ARCH_GEOMETRY

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Build the model (peg + hole meshes are attached in ``_process_spec``)."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)

        # zero-twist command holds the peg at its home pose over the hole
        self.reset_command = TWIST_HOME.copy()

        # sensor start indices (each framepos sensor is 3-dim, each framequat is 4-dim)
        self.tip_adr = self.get_sensor_start_index("tip_pos")
        self.base_adr = self.get_sensor_start_index("base_pos")
        self.entrance_adr = self.get_sensor_start_index("entrance_pos")
        self.bottom_adr = self.get_sensor_start_index("bottom_pos")
        self.tip_quat_adr = self.get_sensor_start_index("tip_quat")
        self.hole_quat_adr = self.get_sensor_start_index("hole_quat")

        # qpos/qvel addresses of the movable hole carriage joints (x, y, z, yaw)
        self.hole_qpos_adr = np.array(
            [self.get_joint_position_start_index(j) for j in ("hole_tx", "hole_ty", "hole_tz", "hole_yaw")]
        )
        self.hole_dof_adr = np.array(
            [self.model.jnt_dofadr[self.model.joint(j).id] for j in ("hole_tx", "hole_ty", "hole_tz", "hole_yaw")]
        )

        self.reset()

    def _apply_hole_pose(self) -> None:
        """Pin the hole carriage joints to the slider-commanded pose.

        The carriage has no actuators, so it is held purely kinematically: we write the
        commanded (x, y, yaw) into the joint qpos and zero their velocity. Because these
        joints are part of the state, the planner's rollouts (seeded from the current
        state) inherit the moved hole without any change to the rollout backend.
        """
        c = self.config
        self.data.qpos[self.hole_qpos_adr] = [c.hole_x, c.hole_y, c.hole_z, c.hole_yaw]
        self.data.qvel[self.hole_dof_adr] = 0.0

    def _process_spec(self) -> None:
        """Attach the hole meshes and the peg (for the selected shape), plus task sensors."""
        spec = self.spec
        geom = self.geometry

        # ---- hole: visual mesh + convex collision pieces on the (static) "hole" body ----
        spec.add_mesh(name="hole_vis_mesh", file=geom.hole_vis)
        for i in range(geom.hole_num_coll):
            spec.add_mesh(name=f"hole_coll_{i}", file=f"{geom.hole_coll_dir}/{geom.hole_coll_prefix}_{i:03d}.obj")

        hole = spec.body("hole")
        hole.add_geom(
            name="hole_vis",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="hole_vis_mesh",
            group=2,
            contype=0,
            conaffinity=0,
            rgba=[0.4, 0.4, 0.8, 1.0],
        )
        # collision pieces are "obstacle" colliders (contype=2, conaffinity=1): they
        # collide with the peg (contype=1, conaffinity=2) but not with each other.
        for i in range(geom.hole_num_coll):
            hole.add_geom(
                name=f"hole_collision_{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"hole_coll_{i}",
                group=3,
                contype=2,
                conaffinity=1,
                rgba=[0.3, 0.3, 0.7, 1.0],
            )

        # ---- peg: visual mesh + convex collision pieces on the carriage "eef" body ----
        spec.add_mesh(name="peg_vis_mesh", file=geom.peg_vis)
        for i in range(geom.peg_num_coll):
            spec.add_mesh(
                name=f"peg_coll_{i}",
                file=f"{geom.peg_coll_dir}/{geom.peg_coll_prefix}_{i:03d}.obj",
                scale=list(geom.peg_coll_scale),  # thin the collider so the peg fits the hole
            )

        # rigid child body of the carriage tip => moves exactly with the commanded twist
        eef = spec.body("eef")
        peg = eef.add_body(name="peg", pos=PEG_POS_IN_EEF, quat=PEG_QUAT_IN_EEF)

        # visual geom: full mesh, no collisions (contype/conaffinity = 0)
        peg.add_geom(
            name="peg_vis",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="peg_vis_mesh",
            contype=0,
            conaffinity=0,
            rgba=[0.49, 0.27, 0.27, 1.0],
        )
        # collision geoms: COACD convex pieces. The peg is a "robot" collider
        # (contype=1, conaffinity=2), so it collides with the "obstacle" geoms
        # (the hole pieces = contype=2, conaffinity=1).
        for i in range(geom.peg_num_coll):
            peg.add_geom(
                name=f"peg_collision_{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"peg_coll_{i}",
                group=3,  # collision group (hidden by default, like the hole pieces)
                contype=1,
                conaffinity=2,
                rgba=[0.49, 0.27, 0.27, 1.0],
            )
        # peg_link: a child frame at the tip rotated into the REAL robot's peg_link
        # convention, so judo's peg_tip frame (and its tip_quat sensor) matches what the
        # robot reports. In peg_link the insertion axis is local +Z (vs the peg body's -Y).
        peg_link = peg.add_body(name="peg_link", pos=[0.0, -PEG_TIP_SHAFT, 0.0], quat=PEG_LINK_QUAT_IN_PEG)

        # the tip site lives on peg_link (orientation now matches the real robot); peg_base
        # stays on the shaft for the orientation-free base->tip insertion axis.
        peg_link.add_site(name="peg_tip", pos=[0.0, 0.0, 0.0], size=[0.004, 0.004, 0.004], rgba=[1, 1, 0, 0.9])
        peg.add_site(name="peg_base", pos=[0.0, 0.03, 0.0], size=[0.004, 0.004, 0.004], rgba=[1, 0.5, 0, 0.9])

        # visual-only marker (the judo viewer renders geoms, not sites): RED sphere at
        # the peg_tip -- the robot point the cost drives toward the (green) goal.
        peg_link.add_geom(
            name="peg_tip_marker",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[0.0, 0.0, 0.0],
            size=[0.01, 0.0, 0.0],
            contype=0,
            conaffinity=0,
            rgba=[1, 0, 0, 1.0],
        )
        # RGB orientation triad at the peg_link frame (x=red, y=green, z=blue). Visual only;
        # drawn in peg_link's local frame so it shows the real robot's peg_link axes.
        _add_frame_triad(peg_link, "peg_tip_axis", origin=[0.0, 0.0, 0.0])

        # Goal-orientation frame: a static child of the hole, offset by GOAL_OFFSET_QUAT_PEGLINK
        # (the goal in the peg_link convention, matching the re-conventioned tip).
        goal = hole.add_body(name="goal_frame", pos=[0.0, 0.0, 0.05], quat=list(GOAL_OFFSET_QUAT_PEGLINK))
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

        Identical to the arm task: lateral alignment to the hole axis, a *gated*
        axial-depth target (hover at the entrance until centered, then descend to the
        bottom), full SO(3) orientation alignment, and twist-velocity damping. All
        quantities are derived from world-frame sites, so the (known) hole pose is
        encoded directly by the sensor placement on the hole body.
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
        # descend. The axial *target* is gated by lateral alignment.
        axial_entrance = np.sum((entrance - bottom) * hole_axis, axis=-1)  # (R, T) entrance height above bottom
        approach_tol = max(self.config.approach_tol, 1e-4)
        descend_gate = np.exp(-((lateral_dist / approach_tol) ** 2))  # (R, T) in [0, 1]
        axial_target = (1.0 - descend_gate) * axial_entrance  # entrance height -> 0 (bottom) as it centers
        axial_err = np.abs(axial - axial_target)  # (R, T)

        # full SO(3) orientation error: geodesic angle between the peg-tip frame and the
        # goal frame (hole frame (x) GOAL_OFFSET_QUAT). In [0, pi].
        tip_quat = sensors[..., self.tip_quat_adr : self.tip_quat_adr + 4]  # (R, T, 4)
        hole_quat = sensors[..., self.hole_quat_adr : self.hole_quat_adr + 4]
        goal_quat = _quat_mul(hole_quat, GOAL_OFFSET_QUAT_PEGLINK)  # (R, T, 4)
        dot = np.abs(np.sum(tip_quat * goal_quat, axis=-1))  # (R, T), |<q_tip, q_goal>|
        align_err = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))  # (R, T) geodesic angle

        # Tilt limit: keep the peg shaft within a +-tilt_limit_deg cone of vertical. The
        # shaft direction in the world is base->tip (peg-local +Y -> world +Z at the goal
        # pose, i.e. pointing "up"). We decompose its tilt into a lean about world x (into
        # the y-z plane) and a lean about world y (into the x-z plane), then apply a
        # one-sided quadratic hinge so the cost is exactly 0 inside the allowed cone.
        base = sensors[..., self.base_adr : self.base_adr + 3]  # (R, T, 3)
        shaft = base - tip
        shaft = shaft / (np.linalg.norm(shaft, axis=-1, keepdims=True) + 1e-9)
        ux, uy, uz = shaft[..., 0], shaft[..., 1], shaft[..., 2]
        tilt_x = np.arctan2(uy, uz)  # (R, T) lean about world x
        tilt_y = np.arctan2(ux, uz)  # (R, T) lean about world y
        tilt_limit = np.deg2rad(self.config.tilt_limit_deg)
        tilt_excess = (
            np.maximum(np.abs(tilt_x) - tilt_limit, 0.0) ** 2
            + np.maximum(np.abs(tilt_y) - tilt_limit, 0.0) ** 2
        )  # (R, T) rad^2, zero inside the cone

        # damp the carriage velocity (== the commanded twist at steady state)
        qvel = states[..., self.model.nq : self.model.nq + self.model.nv]
        qvel_norm = np.linalg.norm(qvel, axis=-1)  # (R, T)

        c = self.config
        cost = (
            c.w_lateral * lateral_dist
            + c.w_axial * axial_err
            + c.w_align * align_err
            + c.w_tilt * tilt_excess
            + c.w_qvel * qvel_norm
        )

        # Contact avoidance: repel the peg tip (laterally) from an estimated contact hotspot.
        # The hotspot (judo-world xy) is supplied at runtime via system_metadata; we add a
        # Gaussian bump that is large when the tip is near it and decays over contact_radius,
        # so the sampler prefers rollouts that keep the tip away from where binding is sensed.
        contact_pt = None if system_metadata is None else system_metadata.get("contact_pt")
        if contact_pt is not None and c.w_contact > 0.0:
            cp_xy = np.asarray(contact_pt, dtype=float)[:2]  # judo-world x, y
            d_xy = np.linalg.norm(tip[..., :2] - cp_xy, axis=-1)  # (R, T) lateral distance
            r = max(c.contact_radius, 1e-3)
            cost = cost + c.w_contact * np.exp(-0.5 * (d_xy / r) ** 2)  # (R, T) bump in [0, w_contact]

        return -cost.sum(axis=-1)  # (R,)

    def success(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict[str, Any] | None = None) -> bool:
        """True once the peg tip is inserted deep enough AND centered on the hole axis.

        Uses the same world-frame sites as ``reward``: the hole axis is entrance->bottom,
        the tip's signed height above the bottom is projected onto it, and insertion depth
        is measured downward from the entrance. Success requires the tip to be at least
        ``self.success_depth_frac`` of the way to the bottom while staying within
        ``SUCCESS_LATERAL_TOL`` of the axis, so a tip skimming past the rim doesn't count.

        Instantaneous by design (like the other tasks' ``success``); any debouncing /
        consecutive-frame hold belongs to the caller.
        """
        s = data.sensordata
        tip = s[self.tip_adr : self.tip_adr + 3]
        entrance = s[self.entrance_adr : self.entrance_adr + 3]
        bottom = s[self.bottom_adr : self.bottom_adr + 3]

        span = entrance - bottom  # hole axis, entrance is "up/out"
        depth_total = float(np.linalg.norm(span))
        if depth_total < 1e-9:  # degenerate geometry -> never "succeed"
            return False
        hole_axis = span / depth_total

        vec = tip - bottom
        axial = float(np.dot(vec, hole_axis))  # tip height above the bottom
        lateral_dist = float(np.linalg.norm(vec - axial * hole_axis))

        inserted_frac = (depth_total - axial) / depth_total  # 0 at entrance, 1 at bottom
        return bool(inserted_frac >= self.success_depth_frac and lateral_dist <= SUCCESS_LATERAL_TOL)

    def pre_sim_step(self) -> None:
        """Hold the hole carriage at the slider-commanded pose each sim step."""
        self._apply_hole_pose()

    def reset(self) -> None:
        """Reset the carriage to its home pose (peg hovering over the hole, zero twist)."""
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self._apply_hole_pose()  # honor any nonzero hole-pose sliders at reset
        self.data.ctrl[:] = self.reset_command
        mujoco.mj_forward(self.model, self.data)


class PegInsertionTwistRectangle(PegInsertionTwist):
    """Twist-controlled peg insertion using the rectangular peg/hole shape."""

    name: str = "peg_insertion_twist_rectangle"
    geometry: PegHoleGeometry = RECTANGLE_GEOMETRY


class PegInsertionTwistEllipse(PegInsertionTwist):
    """Twist-controlled peg insertion using the elliptical peg/hole shape."""

    name: str = "peg_insertion_twist_ellipse"
    geometry: PegHoleGeometry = ELLIPSE_GEOMETRY


class PegInsertionTwistEllipseTeeth(PegInsertionTwist):
    """Twist-controlled peg insertion using the toothed elliptical peg/hole shape."""

    name: str = "peg_insertion_twist_ellipse_teeth"
    geometry: PegHoleGeometry = ELLIPSE_TEETH_GEOMETRY
    success_depth_frac: float = TEETH_SUCCESS_DEPTH_FRAC


class PegInsertionTwistRectangleTeeth(PegInsertionTwist):
    """Twist-controlled peg insertion using the toothed rectangular peg/hole shape."""

    name: str = "peg_insertion_twist_rectangle_teeth"
    geometry: PegHoleGeometry = RECTANGLE_TEETH_GEOMETRY
    success_depth_frac: float = TEETH_SUCCESS_DEPTH_FRAC
