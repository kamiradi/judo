# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Poke and strike: send the cart to a goal beyond the pusher's reach, on a floor of unknown friction.

The pusher can only move in a limited region (a joint range on its x slide). The goal lies beyond it, so the cart has
to be struck: once it leaves the pusher's reach it coasts, and where it stops depends on the floor friction mu
(Coulomb sliding stops after v^2 / (2 mu g)). A wrong estimate at release cannot be corrected afterwards.

The friction is the latent parameter (theta = log mu). It can be learnt by watching the cart slide, so a gentle poke
that keeps the cart in reach is informative, whereas a strike reveals the friction mostly after release, when it is too
late to use. The information term therefore uses per-step increments and counts only steps at which the cart is still
reachable (`expected_info_gain_dynamics(mode="increment", mask=...)`).

The scene, observation window, motion-prediction energy and belief plumbing are those of `uncertain_com_cylinder_push`
(the cart rests on a frictional floor and can spin); the CoM is fixed at the centre.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo.belief import expected_info_gain_dynamics
from judo.gui import slider
from judo.tasks.cost_functions import quadratic_norm
from judo.tasks.cylinder_push import XML_PATH, CylinderPushConfig
from judo.tasks.uncertain_com_cylinder_push import CART_XY_IDX, PUSHER_QPOS_IDX, UncertainComCylinderPush
from judo.utils.fields import np_1d_field

GRAVITY = 9.81
CONTACT_DISTANCE = 0.5  # pusher radius + cart radius: centres closer than this are touching
REACH_MARGIN = 0.02  # the cart counts as reachable while it is within this of touching distance of the workspace
TOUCH_MARGIN = 0.005  # pusher and cart count as touching (for w_contact) when their centres are this close to contact


def _flat_box(parent: Any, name: str, pos: list[float], half_xy: list[float], rgba: list[float]) -> None:
    """A thin, non-colliding, massless box lying flat (for floor markings)."""
    geom = parent.add_geom()
    geom.name, geom.type = name, mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [*half_xy, 0.001]
    geom.pos = pos
    geom.rgba = rgba
    geom.contype, geom.conaffinity, geom.mass = 0, 0, 0.0


def _frame(
    parent: Any, name: str, centre: list[float], half_x: float, half_y: float, rgba: list[float], front: Any = None
) -> None:
    """A rectangular border of thin strips around `centre`; the +x edge can have its own colour (`front`)."""
    w = 0.015
    cx, cy, cz = centre
    _flat_box(parent, f"{name}_edge_back", [cx - half_x, cy, cz], [w, half_y + w], rgba)
    _flat_box(parent, f"{name}_edge_front", [cx + half_x, cy, cz], [w, half_y + w], front or rgba)
    _flat_box(parent, f"{name}_edge_left", [cx, cy + half_y, cz], [half_x, w], rgba)
    _flat_box(parent, f"{name}_edge_right", [cx, cy - half_y, cz], [half_x, w], rgba)


@slider("cart_x_min", -1.0, 0.5, 0.01)
@slider("cart_x_max", -1.0, 0.5, 0.01)
@slider("goal_dist_min", 0.5, 5.0, 0.1)
@slider("goal_dist_max", 0.5, 5.0, 0.1)
@slider("goal_half_size", 0.1, 1.0, 0.05)
@slider("rest_speed", 0.01, 0.2, 0.01)
@slider("mu_min", 0.02, 1.0, 0.01)
@slider("mu_max", 0.02, 1.0, 0.01)
@slider("w_rest", 0.0, 10.0, 0.1)
@slider("w_rest_running", 0.0, 1.0, 0.01)
@slider("w_contact", 0.0, 20.0, 0.1)
@slider("w_info", 0.0, 25.0, 0.1)
@slider("num_plan_particles", 1, 32, 1)
@slider("num_particles", 8, 1024, 1)
@slider("param_jitter", 0.0, 0.2, 0.005)
@slider("obs_noise_pos", 0.0, 0.02, 0.001)
@slider("obs_noise_vel", 0.0, 0.1, 0.005)
@slider("sigma_pos", 0.001, 0.05, 0.001)
@slider("sigma_vel", 0.005, 0.3, 0.005)
@slider("p_outlier", 0.001, 0.5, 0.005)
@slider("info_subsample", 1, 20, 1)
@slider("info_gate_std", 0.0, 0.5, 0.01)
@dataclass
class UncertainFrictionStrikeConfig(CylinderPushConfig):
    """Config for poke and strike with unknown floor friction."""

    # Goal centre, set by reset (beyond the reach limit). No GUI goal sphere: the scene draws the goal square.
    goal_pos: np.ndarray = np_1d_field(
        np.array([2.5, 0.0]),
        names=["x", "y"],
        mins=[-5.0, -5.0],
        maxs=[5.0, 5.0],
        steps=[0.01, 0.01],
    )
    # workspace: the pusher's centre stays in reach_x - workspace_depth <= x <= reach_x and |y| <= reach_y (compiled
    # into the joint ranges, so not adjustable at runtime). The floor box drawn around it is where the pusher's body
    # can be (the centre's range grown by the pusher's radius).
    reach_x: float = 0.0
    reach_y: float = 1.5
    workspace_depth: float = 2.5
    # layout drawn by reset: the cart centre's start x relative to reach_x (<= 0 puts the whole cart inside the box;
    # deeper starts leave room to poke without pushing the cart out of reach), and the goal centre's distance beyond
    # the farthest point the pusher can touch
    cart_x_min: float = -1.0
    cart_x_max: float = -0.6
    goal_dist_min: float = 1.5
    goal_dist_max: float = 3.0
    # success: the cart at rest with its centre inside the goal square (side 2 * goal_half_size; 1 m = 2 x 2 floor
    # grid squares)
    goal_half_size: float = 0.5
    rest_speed: float = 0.05
    # prior over the floor friction: log-uniform on [mu_min, mu_max]
    mu_min: float = 0.1
    mu_max: float = 0.4
    # reward: squared distance of the predicted resting point to the goal, at the end of the horizon (w_rest) and
    # averaged over it (w_rest_running, a mild preference for finishing sooner)
    w_rest: float = 1.0
    w_rest_running: float = 0.1
    # cost per unit fraction of the horizon spent with the pusher touching the cart. Long pushes reveal the friction
    # while the controller can still adjust, which would make deliberate probing unnecessary; this makes them
    # expensive for every controller, while short strikes and pokes stay cheap.
    w_contact: float = 2.0
    w_pusher_proximity: float = 0.1
    w_pusher_velocity: float = 0.0
    w_cart_position: float = 0.0
    # belief over log mu
    num_particles: int = 100
    num_plan_particles: int = 5
    param_jitter: float = 0.02
    # observation noise (sim side); angle noise is the position noise divided by the cart radius
    obs_noise_pos: float = 0.002
    obs_noise_vel: float = 0.005
    # motion-prediction energy
    sigma_pos: float = 0.005
    sigma_vel: float = 0.01
    p_outlier: float = 0.05
    # belief-aware planning: per-step ("increment") information, counted only while the cart is reachable
    w_info: float = 5.0
    info_subsample: int = 5
    info_mode: str = "increment"
    info_reachable_only: bool = True
    # the info term is off while the std of log mu is below info_gate_std and ramps to full strength at 2x that
    info_gate_std: float = 0.05
    obs_window: int = 100


class UncertainFrictionStrike(UncertainComCylinderPush):
    """Strike the cart to a goal beyond the pusher's reach; the floor friction is latent and inferred from motion."""

    name: str = "uncertain_friction_strike"
    config_t: type[UncertainFrictionStrikeConfig] = UncertainFrictionStrikeConfig  # type: ignore[assignment]
    config: UncertainFrictionStrikeConfig  # type: ignore[assignment]
    belief_geom_name: str = "mu_marker"  # friction guesses are drawn as copies of this disc on the friction ruler

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the task."""
        self.plan_params: np.ndarray | None = None  # set by ParamBeliefController before each planning step
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)  # calls reset(), which sets true_log_mu

    @property
    def floor_geom_id(self) -> int:
        """Geom id of the floor, whose friction is the latent parameter."""
        return self.model.geom("floor").id

    # ------------------------------------------------------------------ scene
    def _process_spec(self) -> None:
        """The CoM scene, plus the pusher's workspace (box on the floor), a square goal, and the friction ruler."""
        super()._process_spec()
        spec, cfg = self.spec, self.config_t()
        half_height = float(spec.geom("cart").size[1])

        # Cart-floor contacts use the floor's friction (the latent parameter). The pusher never touches the floor.
        spec.geom("floor").priority = 1

        # The pusher's workspace. Joint positions equal world positions (the pusher body sits at the origin).
        x_range = (cfg.reach_x - cfg.workspace_depth, cfg.reach_x)
        for joint_name, lo, hi in (("slider_x", *x_range), ("slider_y", -cfg.reach_y, cfg.reach_y)):
            joint = spec.joint(joint_name)
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            joint.range = [lo, hi]

        # Floor markings live in a fixed body (no joints): the viewer draws only geoms that belong to a body.
        marks = spec.worldbody.add_body()
        marks.name = "floor_markings"

        # Where the pusher's body can be: a see-through red box on the floor with a red border. Its front edge is the
        # farthest the pusher can reach.
        pusher_radius = float(spec.geom("pusher").size[0])
        x0, x1 = x_range[0] - pusher_radius, x_range[1] + pusher_radius
        y1 = cfg.reach_y + pusher_radius
        z = -half_height + 0.002
        red = [0.85, 0.2, 0.2, 0.9]
        _flat_box(marks, "workspace_fill", [0.5 * (x0 + x1), 0.0, z], [0.5 * (x1 - x0), y1], [0.85, 0.2, 0.2, 0.15])
        _frame(marks, "workspace", [0.5 * (x0 + x1), 0.0, z + 0.001], 0.5 * (x1 - x0), y1, red)

        # Square goal on the goal ghost (a mocap body, moved to each episode's goal): hide the cart-sized disc and the
        # angle stripe, and add the square.
        spec.geom("goal_stripe").rgba = [0.0, 0.0, 0.0, 0.0]
        spec.geom("goal_disc").rgba = [0.0, 0.0, 0.0, 0.0]
        ghost = spec.body("goal_ghost")
        h = cfg.goal_half_size
        _flat_box(ghost, "goal_square", [0.0, 0.0, 0.001], [h, h], [0.2, 0.7, 0.35, 0.35])
        _frame(ghost, "goal_square", [0.0, 0.0, 0.002], h, h, [0.1, 0.45, 0.2, 0.9])

        # Friction ruler beside the workspace: log mu maps linearly onto x between the ruler's ends.
        x0, x1, y = self._ruler_geometry(cfg)
        ruler = marks.add_geom()
        ruler.name, ruler.type = "mu_ruler", mujoco.mjtGeom.mjGEOM_BOX
        ruler.size = [0.5 * (x1 - x0), 0.01, 0.002]
        ruler.pos = [0.5 * (x0 + x1), y, -half_height + 0.002]
        ruler.rgba = [0.9, 0.9, 0.9, 0.9]
        ruler.contype, ruler.conaffinity, ruler.mass = 0, 0, 0.0
        lo, hi = np.log(cfg.mu_min), np.log(cfg.mu_max)
        for mu in (0.1, 0.2, 0.4):
            if not lo <= np.log(mu) <= hi:
                continue
            tick = marks.add_geom()
            tick.name, tick.type = f"mu_tick_{mu:g}", mujoco.mjtGeom.mjGEOM_BOX
            tick.size = [0.008, 0.06, 0.002]
            tick.pos = [x0 + (np.log(mu) - lo) / (hi - lo) * (x1 - x0), y, -half_height + 0.003]
            tick.rgba = [0.9, 0.9, 0.9, 0.9]
            tick.contype, tick.conaffinity, tick.mass = 0, 0, 0.0

        marker = spec.worldbody.add_geom()  # invisible; its size and height are used to draw the friction guesses
        marker.name, marker.type = "mu_marker", mujoco.mjtGeom.mjGEOM_CYLINDER
        marker.size = [0.03, 0.004, 0.0]
        marker.pos = [0.0, 0.0, -half_height + 0.01]
        marker.rgba = [1.0, 0.55, 0.0, 0.0]
        marker.contype, marker.conaffinity, marker.mass = 0, 0, 0.0

    @staticmethod
    def _ruler_geometry(cfg: UncertainFrictionStrikeConfig) -> tuple[float, float, float]:
        """The ruler's x at its low- and high-friction ends, and its y (beside the workspace)."""
        return cfg.reach_x - 1.5, cfg.reach_x + 1.5, -(cfg.reach_y + 0.9)

    def update_goal_marker(self, data: mujoco.MjData | None = None) -> None:
        """Move the goal ghost to the configured goal position."""
        data = self.data if data is None else data
        mocap_id = self.model.body("goal_ghost").mocapid[0]
        data.mocap_pos[mocap_id, :2] = self.config.goal_pos[:2]
        data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------------ prior over log mu
    @property
    def log_mu_range(self) -> tuple[float, float]:
        """Support of the prior over log mu."""
        return float(np.log(self.config.mu_min)), float(np.log(self.config.mu_max))

    def sample_prior(self, n: int) -> np.ndarray:
        """Log-uniform friction samples, as log mu, shape (n, 1)."""
        lo, hi = self.log_mu_range
        return np.random.uniform(lo, hi, size=(n, 1))

    def project(self, log_mu: np.ndarray) -> np.ndarray:
        """Clip log mu into the prior's support."""
        lo, hi = self.log_mu_range
        return np.clip(log_mu, lo, hi)

    def apply_params(self, models: list[mujoco.MjModel], datas: list[mujoco.MjData], log_mu: np.ndarray) -> None:
        """Give each model its own floor friction (no model constants depend on friction, so no mj_setConst)."""
        assert len(log_mu) == len(models)
        for model, theta in zip(models, log_mu, strict=True):
            model.geom_friction[self.floor_geom_id, 0] = np.exp(theta[0])

    @property
    def belief_jitter(self) -> float:
        """Std of the roughening noise added to resampled particles (in log mu)."""
        return self.config.param_jitter

    # ------------------------------------------------------------------ sim side: layout, true friction, observations
    def reset(self) -> None:
        """Cart inside the pusher's workspace, goal beyond its reach, pusher behind the cart; a new true friction."""
        cfg = self.config
        cart_x = cfg.reach_x + np.random.uniform(cfg.cart_x_min, cfg.cart_x_max)
        cart_y = np.random.uniform(-0.5, 0.5)
        goal_x = cfg.reach_x + CONTACT_DISTANCE + np.random.uniform(cfg.goal_dist_min, cfg.goal_dist_max)
        cfg.goal_pos = np.array([goal_x, np.random.uniform(-0.75, 0.75)])

        # the pusher starts behind the cart (at least 0.2 m clear of it) and inside its workspace
        pusher_x = np.random.uniform(cfg.reach_x - cfg.workspace_depth + 0.1, cart_x - CONTACT_DISTANCE - 0.2)
        qpos = np.zeros(self.model.nq)
        qpos[PUSHER_QPOS_IDX] = [pusher_x, np.random.uniform(-1.0, 1.0)]
        qpos[CART_XY_IDX] = [cart_x, cart_y]
        self.data.qpos = qpos
        self.data.qvel = np.zeros(self.model.nv)

        self.episode_id = int(np.random.randint(1, 2**31 - 1))
        self.true_com = np.zeros(2)  # the CoM is not uncertain here (it stays at the compiled near-centre offset)
        self.true_log_mu = float(self.sample_prior(1)[0, 0])
        self.sim_model.geom_friction[self.floor_geom_id, 0] = np.exp(self.true_log_mu)
        self.update_goal_marker()
        mujoco.mj_forward(self.model, self.data)

        self._obs_buffer: deque = deque(maxlen=cfg.obs_window)
        self._last_obs = self._observe()

    def get_sim_metadata(self) -> dict[str, Any]:
        """Sliding window of observed transitions, plus the true log friction (for the oracle and logging only)."""
        metadata = super().get_sim_metadata()
        metadata.pop("true_com")
        # Never read by the filter or the planner; used by scripts/baselines_strike/oracle.py and for logging.
        metadata["true_log_mu"] = np.array([self.true_log_mu])
        # The goal is drawn by reset in the simulation process; the controller's copy of the task reads it from here.
        metadata["goal_pos"] = np.asarray(self.config.goal_pos[:2], dtype=float)
        return metadata

    def goal(self, system_metadata: dict[str, Any] | None) -> np.ndarray:
        """The episode's goal: from the simulator's metadata when available, else the config."""
        if system_metadata is not None and "goal_pos" in system_metadata:
            return np.asarray(system_metadata["goal_pos"], dtype=float)[:2]
        return np.asarray(self.config.goal_pos[:2], dtype=float)

    # ------------------------------------------------------------------ geometry helpers
    def reachable(self, states: np.ndarray) -> np.ndarray:
        """Whether the pusher's workspace can still touch the cart, (..., nq + nv) -> (...) bool."""
        cfg = self.config
        cart = states[..., CART_XY_IDX]
        dx = np.maximum(0.0, cart[..., 0] - cfg.reach_x)
        dy = np.maximum(0.0, np.abs(cart[..., 1]) - cfg.reach_y)
        return np.hypot(dx, dy) < CONTACT_DISTANCE + REACH_MARGIN

    def rest_point(self, states: np.ndarray, mu: np.ndarray | float) -> np.ndarray:
        """Where the cart stops if it slides freely from here: x + v |v| / (2 mu g), (..., nq + nv) -> (..., 2).

        Args:
            states: Full states, shape (..., nq + nv).
            mu: Floor friction, broadcastable to states.shape[:-1].
        """
        nq = self.model.nq
        vel = states[..., nq + CART_XY_IDX]
        speed = np.linalg.norm(vel, axis=-1, keepdims=True)
        return states[..., CART_XY_IDX] + vel * speed / (2 * GRAVITY * np.asarray(mu)[..., None])

    def cart_speed(self, data: mujoco.MjData) -> float:
        """Planar speed of the cart."""
        return float(np.linalg.norm(data.qvel[CART_XY_IDX]))

    # ------------------------------------------------------------------ reward / success
    def _reward_with_mu(self, states: np.ndarray, mu: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Task reward of rollouts (R, T, nx), each under its own friction mu (R,)."""
        cfg = self.config
        nq = self.model.nq
        rest_err = quadratic_norm(self.rest_point(states, mu[:, None]) - goal)  # (R, T)
        rest_reward = -cfg.w_rest * rest_err[:, -1] - cfg.w_rest_running * rest_err.mean(-1)

        cart_pos = states[..., CART_XY_IDX]
        cart_to_goal = goal - cart_pos
        direction = cart_to_goal / np.maximum(np.linalg.norm(cart_to_goal, axis=-1, keepdims=True), 1e-9)
        pusher_goal = cart_pos - cfg.pusher_goal_offset * direction
        pusher_reward = -cfg.w_pusher_proximity * quadratic_norm(states[..., PUSHER_QPOS_IDX] - pusher_goal).mean(-1)
        velocity_reward = -cfg.w_pusher_velocity * quadratic_norm(states[..., nq + PUSHER_QPOS_IDX]).mean(-1)
        gap = np.linalg.norm(states[..., PUSHER_QPOS_IDX] - cart_pos, axis=-1)
        contact_reward = -cfg.w_contact * (gap < CONTACT_DISTANCE + TOUCH_MARGIN).mean(-1)
        return rest_reward + pusher_reward + velocity_reward + contact_reward

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Task reward with one friction for every rollout: the first plan guess if set, else the prior's centre."""
        log_mu = float(self.plan_params[0, 0]) if self.plan_params is not None else float(np.mean(self.log_mu_range))
        return self._reward_with_mu(states, np.full(states.shape[0], np.exp(log_mu)), self.goal(system_metadata))

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
        """Task reward averaged over friction guesses (each predicts its own resting point) + lambda * info gain.

        Args:
            states: Rollout states, shape (K * M, T, nq + nv), ordered action-major (rollout k*M + m uses guess m).
            sensors: Rollout sensors, shape (K * M, T, nsensordata).
            controls: Rollout controls, shape (K * M, T, nu).
            num_particles: M, number of friction guesses per action sequence.
            particle_weights: Weights of the M guesses, shape (M,).
            system_metadata: Carries the episode's goal (`goal`).
            belief_std: Std of log mu under the full belief, used to gate the information term.

        Returns:
            Rewards, shape (K,).
        """
        assert self.plan_params is not None and len(self.plan_params) == num_particles
        num_rollouts = states.shape[0] // num_particles
        mu = np.tile(np.exp(self.plan_params[:, 0]), num_rollouts)
        task_reward = self._reward_with_mu(states, mu, self.goal(system_metadata))
        task_reward = task_reward.reshape(num_rollouts, num_particles) @ particle_weights

        w_info = self.info_weight(belief_std)
        if w_info <= 0.0 or num_particles < 2:
            return task_reward

        cfg = self.config
        stride = max(1, cfg.info_subsample)
        traj = states.reshape(num_rollouts, num_particles, states.shape[1], -1)[:, :, ::stride]
        mask = self.reachable(traj) if cfg.info_reachable_only else None
        info_gain = expected_info_gain_dynamics(
            self.cart_features(traj), particle_weights, self.energy_params, mode=cfg.info_mode, mask=mask, stride=stride
        )
        return task_reward + w_info * info_gain

    def success(self, model: mujoco.MjModel, data: mujoco.MjData, metadata: dict[str, Any] | None = None) -> bool:
        """The cart is at rest with its centre inside the goal square."""
        offset = np.abs(data.qpos[CART_XY_IDX] - self.config.goal_pos[:2])
        return bool(np.all(offset <= self.config.goal_half_size) and self.cart_speed(data) < self.config.rest_speed)

    def yaw_error(self, data: mujoco.MjData) -> float:
        """No angle goal in this task."""
        return float("nan")

    # ------------------------------------------------------------------ visualization
    def belief_viz_positions(self, log_mu: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Friction guesses as points on the friction ruler beside the workspace, shape (N, 2)."""
        x0, x1, y = self._ruler_geometry(self.config)
        lo, hi = self.log_mu_range
        x = x0 + (np.asarray(log_mu)[:, 0] - lo) / (hi - lo) * (x1 - x0)
        return np.stack([x, np.full_like(x, y)], axis=-1)
