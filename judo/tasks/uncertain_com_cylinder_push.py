# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Cylinder push to a goal position and angle, with an unknown centre of mass (CoM).

The cart's pose is observed (with noise); what is uncertain is where its CoM sits inside it. The scene is the cylinder
push scene with two physical changes that make the CoM matter: the cart can rotate and rests on the floor with friction
(0.2), and the pusher grips it (friction 0.5 instead of 0.01). A push then spins the cart one way or the other depending on
which side of the push line the CoM lies, so reaching a goal angle needs the CoM.

Like `uncertain_cylinder_push`, the same task class plays two roles:
  * In the simulation process it draws the true CoM, records noisy observations of every transition, and publishes a
    sliding window of them (with the executed controls) through `get_sim_metadata`.
  * In the controller process it provides the belief over the CoM (`ParameterBelief`) and the belief-aware reward
    (task reward averaged over CoM guesses + lambda * expected information gain), used by
    `judo.controller.param_belief_controller.ParamBeliefController`.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo.belief import DynamicsEnergyParams, ParameterBelief, expected_info_gain_dynamics
from judo.gui import slider
from judo.tasks.cost_functions import quadratic_norm
from judo.tasks.cylinder_push import GOAL_DISTANCE_THRESHOLD, XML_PATH, CylinderPush, CylinderPushConfig
from judo.utils.fields import np_1d_field

PUSHER_QPOS_IDX = np.array([0, 1])
CART_XY_IDX = np.array([2, 3])
CART_Z_IDX = 4
CART_YAW_IDX = 5

# Sliding friction. MuJoCo uses the larger of the two geoms' coefficients, so pusher-cart contacts get 0.5 (the pusher
# grips the cart, so off-centre pushes apply torque) and cart-floor contacts get 0.2. At 0.5 on the floor the cart needs
# ~4.9 N to slide, which the pusher's position actuator (kp = 10) only produces with targets ~0.5 m ahead of it.
PUSHER_FRICTION = 0.5
FLOOR_FRICTION = 0.2
# MuJoCo gotcha: if the cart is compiled with its CoM exactly at the body origin, the compiler marks the body as
# "simple" and leaves the slide-yaw coupling out of the mass matrix's sparsity pattern. Setting body_ipos later (as
# every CoM hypothesis does) is then silently ignored, even after mj_setConst. Compiling with a 1 mm offset keeps the
# coupling in the pattern; set_com() then overwrites the offset with the actual hypothesis.
BASE_COM_OFFSET = np.array([1e-3, 1e-3])


def set_com(models: list[mujoco.MjModel], datas: list[mujoco.MjData], body_id: int, offsets: np.ndarray) -> None:
    """Set the planar CoM offset (in the body frame) of one body in each model, then refresh the model constants.

    Args:
        models: Models to modify (one per row of `offsets`).
        datas: Scratch data, one per model (mj_setConst overwrites them).
        body_id: Body whose CoM is set.
        offsets: CoM offsets, shape (len(models), 2).
    """
    assert len(offsets) == len(models) <= len(datas)
    for model, data, offset in zip(models, datas, offsets, strict=False):
        model.body_ipos[body_id, :2] = offset
        mujoco.mj_setConst(model, data)  # the mass matrix constants are cached (dof_M0 etc.)


@slider("goal_yaw", -3.14, 3.14, 0.01)
@slider("w_cart_yaw", 0.0, 5.0, 0.1)
@slider("yaw_tol", 0.01, 0.5, 0.01)
@slider("com_radius", 0.0, 0.2, 0.01)
@slider("w_info", 0.0, 25.0, 0.1)
@slider("num_plan_particles", 1, 32, 1)
@slider("num_particles", 8, 1024, 1)
@slider("com_jitter", 0.0, 0.02, 0.001)
@slider("obs_noise_pos", 0.0, 0.02, 0.001)
@slider("obs_noise_vel", 0.0, 0.1, 0.005)
@slider("sigma_pos", 0.001, 0.05, 0.001)
@slider("sigma_vel", 0.005, 0.3, 0.005)
@slider("p_outlier", 0.001, 0.5, 0.005)
@slider("info_subsample", 1, 20, 1)
@slider("info_gate_std", 0.0, 0.1, 0.005)
@dataclass
class UncertainComCylinderPushConfig(CylinderPushConfig):
    """Config for the cylinder push task with an unknown centre of mass."""

    # Goal pose. Same goal_pos as cylinder push, but without the GUI's generic goal sphere: the scene draws a
    # see-through copy of the cart at the goal pose instead (the "goal_ghost" mocap body), so the goal angle is visible.
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0]),
        names=["x", "y"],
        mins=[-1.0, -1.0],
        maxs=[1.0, 1.0],
        steps=[0.01, 0.01],
    )
    goal_yaw: float = 0.0
    w_cart_yaw: float = 1.0
    yaw_tol: float = 0.1
    # prior over the CoM offset: uniform over a disk of this radius (the cart's radius is 0.25 m)
    com_radius: float = 0.15
    # belief
    num_particles: int = 30
    num_plan_particles: int = 15
    com_jitter: float = 0.003
    # observation noise (sim side); angle noise is the position noise divided by the cart radius
    obs_noise_pos: float = 0.002
    obs_noise_vel: float = 0.01
    # motion-prediction energy
    sigma_pos: float = 0.005
    sigma_vel: float = 0.05
    p_outlier: float = 0.05
    # belief-aware planning
    w_info: float = 1.0
    info_subsample: int = 5
    # the info term is off while the belief std is below info_gate_std and ramps to full strength at 2x that
    info_gate_std: float = 0.02
    obs_window: int = 100


class UncertainComCylinderPush(CylinderPush):
    """Cylinder push to a goal position and angle, where the cart's CoM is latent and inferred from its motion."""

    name: str = "uncertain_com_cylinder_push"
    config_t: type[UncertainComCylinderPushConfig] = UncertainComCylinderPushConfig  # type: ignore[assignment]
    config: UncertainComCylinderPushConfig
    belief_geom_name: str = "com_marker"  # CoM guesses are drawn as copies of this small disc

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the task."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)
        self.pusher_geom_id = self.model.geom("pusher").id
        self.cart_geom_id = self.model.geom("cart").id

    # ------------------------------------------------------------------ scene
    def _process_spec(self) -> None:
        """Let the cart rotate and rest on a frictional floor; add an angle stripe and the CoM-guess marker."""
        spec = self.spec
        cart_geom = spec.geom("cart")
        radius, half_height = float(cart_geom.size[0]), float(cart_geom.size[1])

        floor = spec.geom("floor")
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.pos = [0.0, 0.0, -half_height]
        floor.friction[0] = FLOOR_FRICTION
        # collision filtering: floor-cart and pusher-cart only (the pusher floats in the plane as before)
        floor.contype, floor.conaffinity = 1, 1
        spec.geom("pusher").contype, spec.geom("pusher").conaffinity = 2, 2
        cart_geom.contype, cart_geom.conaffinity = 3, 3
        spec.geom("pusher").friction[0] = PUSHER_FRICTION
        cart_geom.friction[0] = FLOOR_FRICTION

        cart = spec.body("cart")
        for joint_name in ("slider_cart_x", "slider_cart_y"):
            spec.joint(joint_name).damping = 0.0  # floor friction replaces the viscous damping
        z = cart.add_joint()
        z.name, z.type, z.axis = "slider_cart_z", mujoco.mjtJoint.mjJNT_SLIDE, [0.0, 0.0, 1.0]
        yaw = cart.add_joint()
        yaw.name, yaw.type, yaw.axis = "hinge_cart_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0.0, 0.0, 1.0]

        mass = float(cart_geom.mass)
        cart_geom.mass = 0.0
        cart.explicitinertial = True
        cart.mass = mass
        cart.ipos = [*BASE_COM_OFFSET, 0.0]
        # uniform solid cylinder about its own axes (the inertia is not part of the uncertainty)
        i_xy = mass * (3 * radius**2 + (2 * half_height) ** 2) / 12
        cart.inertia = [i_xy, i_xy, 0.5 * mass * radius**2]

        stripe = cart.add_geom()  # shows the cart's angle; yaw = 0 points the stripe along +x
        stripe.name, stripe.type = "cart_stripe", mujoco.mjtGeom.mjGEOM_BOX
        stripe.size = [0.5 * radius, 0.02, 0.004]
        stripe.pos = [0.5 * radius, 0.0, half_height + 0.004]
        stripe.rgba = [1.0, 1.0, 1.0, 1.0]
        stripe.contype, stripe.conaffinity, stripe.mass = 0, 0, 0.0

        marker = spec.worldbody.add_geom()  # invisible; its size and height are used to draw the CoM guesses
        marker.name, marker.type = "com_marker", mujoco.mjtGeom.mjGEOM_CYLINDER
        marker.size = [0.025, 0.004, 0.0]
        marker.pos = [0.0, 0.0, half_height + 0.015]
        marker.rgba = [1.0, 0.55, 0.0, 0.0]
        marker.contype, marker.conaffinity, marker.mass = 0, 0, 0.0

        # Goal pose marker: a flat see-through disc the size of the cart, with a stripe along the goal angle. A mocap
        # body, so the sim can move it to the configured goal (the viewer draws every body from the sim's poses).
        # Its geoms never collide and it has no joints, so the physics are unchanged.
        ghost = spec.worldbody.add_body()
        ghost.name, ghost.mocap = "goal_ghost", True
        ghost.pos = [0.0, 0.0, -half_height + 0.003]
        disc = ghost.add_geom()
        disc.name, disc.type = "goal_disc", mujoco.mjtGeom.mjGEOM_CYLINDER
        disc.size = [radius, 0.003, 0.0]
        disc.rgba = [0.2, 0.7, 0.35, 0.35]
        goal_stripe = ghost.add_geom()
        goal_stripe.name, goal_stripe.type = "goal_stripe", mujoco.mjtGeom.mjGEOM_BOX
        goal_stripe.size = [0.5 * radius, 0.02, 0.002]
        goal_stripe.pos = [0.5 * radius, 0.0, 0.004]
        goal_stripe.rgba = [0.1, 0.45, 0.2, 0.9]
        for geom in (disc, goal_stripe):
            geom.contype, geom.conaffinity, geom.mass = 0, 0, 0.0

        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    def update_goal_marker(self, data: mujoco.MjData | None = None) -> None:
        """Move the goal ghost to the configured goal position and angle."""
        data = self.data if data is None else data
        mocap_id = self.model.body("goal_ghost").mocapid[0]
        data.mocap_pos[mocap_id, :2] = self.config.goal_pos[:2]
        half_yaw = 0.5 * self.config.goal_yaw
        data.mocap_quat[mocap_id] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]

    def pre_sim_step(self) -> None:
        """Keep the goal ghost on the goal (the GUI can change it at any time)."""
        self.update_goal_marker()

    @property
    def cart_body_id(self) -> int:
        """Body id of the cart."""
        return self.model.body("cart").id

    @property
    def cart_radius(self) -> float:
        """Radius of the cart cylinder."""
        return float(self.model.geom("cart").size[0])

    # ------------------------------------------------------------------ prior over the CoM
    def sample_prior(self, n: int) -> np.ndarray:
        """Uniform samples over the disk of radius com_radius, shape (n, 2)."""
        r = self.config.com_radius * np.sqrt(np.random.rand(n))
        phi = 2 * np.pi * np.random.rand(n)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=-1)

    def project(self, offsets: np.ndarray) -> np.ndarray:
        """Pull offsets that left the prior disk back onto its edge."""
        norm = np.linalg.norm(offsets, axis=-1, keepdims=True)
        scale = np.minimum(1.0, self.config.com_radius / np.maximum(norm, 1e-12))
        return offsets * scale

    def apply_params(self, models: list[mujoco.MjModel], datas: list[mujoco.MjData], offsets: np.ndarray) -> None:
        """Give each model its own CoM hypothesis."""
        set_com(models, datas, self.cart_body_id, offsets)

    # ------------------------------------------------------------------ sim-side: true CoM and observations
    def reset(self) -> None:
        """Random pusher/cart positions and cart angle, and a new true CoM drawn from the prior."""
        theta = 2 * np.pi * np.random.rand(2)
        qpos = np.zeros(self.model.nq)
        qpos[PUSHER_QPOS_IDX] = [np.cos(theta[0]), np.sin(theta[0])]
        qpos[CART_XY_IDX] = [2 * np.cos(theta[1]), 2 * np.sin(theta[1])]
        qpos[CART_YAW_IDX] = np.pi * (2 * np.random.rand() - 1)
        self.data.qpos = qpos
        self.data.qvel = np.zeros(self.model.nv)

        self.episode_id = int(np.random.randint(1, 2**31 - 1))
        self.true_com = self.sample_prior(1)[0]
        set_com([self.sim_model], [mujoco.MjData(self.sim_model)], self.cart_body_id, self.true_com[None])
        self.update_goal_marker()
        mujoco.mj_forward(self.model, self.data)

        self._obs_buffer: deque = deque(maxlen=self.config.obs_window)
        self._last_obs = self._observe()

    def _observe(self) -> np.ndarray:
        """Full state with the cart's planar pose and velocity corrupted by noise (the pusher is proprioceptive)."""
        state = np.concatenate([self.data.qpos, self.data.qvel])
        nq, r = self.model.nq, self.cart_radius
        sp, sv = self.config.obs_noise_pos, self.config.obs_noise_vel
        state[CART_XY_IDX] += sp * np.random.randn(2)
        state[CART_YAW_IDX] += sp / r * np.random.randn()
        state[nq + CART_XY_IDX] += sv * np.random.randn(2)
        state[nq + CART_YAW_IDX] += sv / r * np.random.randn()
        return state

    def post_sim_step(self) -> None:
        """Record the observed transition (previous observation, executed control, new observation)."""
        obs = self._observe()
        self._obs_buffer.append((float(self.data.time), self._last_obs, self.data.ctrl.copy(), obs))
        self._last_obs = obs

    def get_sim_metadata(self) -> dict[str, Any]:
        """Sliding window of observed transitions, plus the true CoM (for the oracle baseline and logging only)."""
        buf = list(self._obs_buffer)
        nx = self.model.nq + self.model.nv
        return {
            "episode_id": self.episode_id,
            "obs_time": np.array([b[0] for b in buf], dtype=float),
            "obs_pre_state": np.array([b[1] for b in buf], dtype=float).reshape(-1, nx),
            "obs_ctrl": np.array([b[2] for b in buf], dtype=float).reshape(-1, self.model.nu),
            "obs_post_state": np.array([b[3] for b in buf], dtype=float).reshape(-1, nx),
            # Never read by the filter or the planner; used by scripts/baselines_com/oracle.py and for logging.
            "true_com": np.asarray(self.true_com, dtype=float),
        }

    # ------------------------------------------------------------------ controller-side: belief + action energy
    @property
    def energy_params(self) -> DynamicsEnergyParams:
        """Motion-prediction energy parameters (features: x, y, r*yaw, vx, vy, r*yaw_rate)."""
        return DynamicsEnergyParams(
            num_pos=3,
            sigma_pos=self.config.sigma_pos,
            sigma_vel=self.config.sigma_vel,
            p_outlier=self.config.p_outlier,
        )

    def cart_features(self, states: np.ndarray) -> np.ndarray:
        """Cart features compared by the energy, (..., nq + nv) -> (..., 6).

        The angle and angular velocity are scaled by the cart radius (rim displacement / rim speed), so one sigma per
        group covers both translation and rotation.
        """
        nq, r = self.model.nq, self.cart_radius
        return np.concatenate(
            [
                states[..., CART_XY_IDX],
                r * states[..., CART_YAW_IDX : CART_YAW_IDX + 1],
                states[..., nq + CART_XY_IDX],
                r * states[..., nq + CART_YAW_IDX : nq + CART_YAW_IDX + 1],
            ],
            axis=-1,
        )

    def make_belief(self) -> ParameterBelief:
        """Particle belief over the CoM offset using this task's (known) model."""
        return ParameterBelief(
            self.model,
            params=self.energy_params,
            apply_params=self.apply_params,
            sample_prior=self.sample_prior,
            project=self.project,
            features=self.cart_features,
            num_particles=self.config.num_particles,
            jitter=self.belief_jitter,
        )

    @property
    def belief_jitter(self) -> float:
        """Std of the roughening noise added to resampled CoM particles (m)."""
        return self.config.com_jitter

    def info_weight(self, belief_std: float | None) -> float:
        """Effective info weight: 0 below info_gate_std, linear ramp up to w_info at 2 * info_gate_std."""
        gate = self.config.info_gate_std
        if belief_std is None or gate <= 0.0:
            return self.config.w_info
        return self.config.w_info * float(np.clip((belief_std - gate) / gate, 0.0, 1.0))

    def belief_viz_positions(self, offsets: np.ndarray, state: np.ndarray) -> np.ndarray:
        """World positions of CoM guesses on the cart in its current pose: cart_xy + R(yaw) offset, shape (N, 2)."""
        yaw = state[CART_YAW_IDX]
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return state[CART_XY_IDX] + offsets @ rot.T

    # ------------------------------------------------------------------ reward / success
    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """The cylinder push reward plus an angle term, summed over time.

        CylinderPush.reward hard-codes a 4-dof state, so the position terms are recomputed here with this scene's
        indices; they are otherwise identical: pusher proximity to the point behind the cart (seen from the goal),
        pusher velocity, cart-to-goal distance. The angle term is w_cart_yaw * (1 - cos(yaw - goal_yaw)).
        """
        nq = self.model.nq
        pusher_pos = states[..., PUSHER_QPOS_IDX]
        cart_pos = states[..., CART_XY_IDX]
        pusher_vel = states[..., nq + PUSHER_QPOS_IDX]
        yaw = states[..., CART_YAW_IDX]
        cart_goal = self.config.goal_pos[0:2]

        cart_to_goal = cart_goal - cart_pos
        cart_to_goal_direction = cart_to_goal / np.linalg.norm(cart_to_goal, axis=-1, keepdims=True)
        pusher_goal = cart_pos - self.config.pusher_goal_offset * cart_to_goal_direction

        pusher_reward = -self.config.w_pusher_proximity * quadratic_norm(pusher_pos - pusher_goal).sum(-1)
        velocity_reward = -self.config.w_pusher_velocity * quadratic_norm(pusher_vel).sum(-1)
        goal_reward = -self.config.w_cart_position * quadratic_norm(cart_pos - cart_goal).sum(-1)
        yaw_reward = -self.config.w_cart_yaw * (1.0 - np.cos(yaw - self.config.goal_yaw)).sum(-1)
        return pusher_reward + velocity_reward + goal_reward + yaw_reward

    def belief_reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        num_particles: int,
        particle_weights: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
        belief_std: float | None = None,
    ) -> np.ndarray:
        """Belief-aware reward: task reward averaged over CoM guesses + lambda * expected information gain.

        Args:
            states: Rollout states, shape (K * M, T, nq + nv), ordered action-major (rollout k*M + m uses guess m).
            sensors: Rollout sensors, shape (K * M, T, nsensordata).
            controls: Rollout controls, shape (K * M, T, nu).
            num_particles: M, number of CoM guesses per action sequence.
            particle_weights: Weights of the M guesses, shape (M,).
            system_metadata: Passed through to the task reward.
            belief_std: Spread sqrt(tr(cov)) of the full CoM belief (m), used to gate the information term.

        Returns:
            Rewards, shape (K,).
        """
        num_rollouts = states.shape[0] // num_particles
        task_reward = self.reward(states, sensors, controls, system_metadata).reshape(num_rollouts, num_particles)
        task_reward = task_reward @ particle_weights

        w_info = self.info_weight(belief_std)
        if w_info <= 0.0 or num_particles < 2:
            return task_reward

        sub = slice(None, None, max(1, self.config.info_subsample))
        traj = states.reshape(num_rollouts, num_particles, states.shape[1], -1)[:, :, sub]
        info_gain = expected_info_gain_dynamics(self.cart_features(traj), particle_weights, self.energy_params)
        return task_reward + w_info * info_gain

    def yaw_error(self, data: mujoco.MjData) -> float:
        """Absolute angle error to the goal angle, wrapped to [0, pi]."""
        diff = data.qpos[CART_YAW_IDX] - self.config.goal_yaw
        return float(abs(np.arctan2(np.sin(diff), np.cos(diff))))

    def success(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict[str, Any] | None = None) -> bool:
        """The cart is within the goal radius and within yaw_tol of the goal angle at the same time."""
        pos_err = np.linalg.norm(data.qpos[CART_XY_IDX] - self.config.goal_pos[0:2])
        return bool(pos_err < GOAL_DISTANCE_THRESHOLD and self.yaw_error(data) < self.config.yaw_tol)
