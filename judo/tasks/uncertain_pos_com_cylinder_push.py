# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Cylinder push to a goal position and angle, with an unknown cart position AND an unknown centre of mass (CoM).

This combines the two other uncertain push tasks:
  * State uncertainty, as in `uncertain_cylinder_push`: the cart's planar position is never observed. It is inferred
    from the pusher's contacts (a contact flag and a noisy contact point), starting from a noisy prior mean.
  * Dynamics uncertainty, as in `uncertain_com_cylinder_push`: the cart's CoM offset is latent and inferred from how
    the cart turns when pushed. The scene, goal pose, reward and success test are that task's.

What the cart reveals: touch, plus an on-board IMU that measures its yaw and yaw rate (noisy). The cart is a disc, so
touch alone could never reveal its angle; an IMU senses orientation but not position.

Estimation is gated (see `judo.belief.pos_com_belief.PosComBelief`): the CoM belief is only updated while the cart is
localized (position belief std < `localized_std`). A sideways position error and a sideways CoM offset produce the same
torque, so updating the CoM before the cart is found would fit a CoM that compensates for the wrong position.

The planner's information term has the same structure: a contact term (as in the position task) while the position is
uncertain, and a yaw-dynamics term (as in the CoM task) only while the cart is localized and the CoM is uncertain.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from judo.belief import (
    ContactEnergyParams,
    DynamicsEnergyParams,
    ParameterBelief,
    ParticleBelief,
    PosComBelief,
    expected_info_gain,
    expected_info_gain_dynamics,
    predicted_contact,
)
from judo.gui import slider
from judo.tasks.cylinder_push import CylinderPushConfig
from judo.tasks.uncertain_com_cylinder_push import (
    CART_XY_IDX,
    CART_YAW_IDX,
    PUSHER_QPOS_IDX,
    UncertainComCylinderPush,
)
from judo.utils.fields import np_1d_field

PUSHER_QVEL_IDX = PUSHER_QPOS_IDX  # the pusher's two slide joints come first in qpos and qvel
CART_XY_VEL_IDX = CART_XY_IDX  # likewise the cart's x / y slides


def info_ramp(std: float | None, gate: float) -> float:
    """0 while std is below gate, ramping linearly to 1 at 2 * gate (1 if either is unset)."""
    if std is None or gate <= 0.0:
        return 1.0
    return float(np.clip((std - gate) / gate, 0.0, 1.0))


@slider("goal_yaw", -3.14, 3.14, 0.01)
@slider("w_cart_yaw", 0.0, 5.0, 0.1)
@slider("yaw_tol", 0.01, 0.5, 0.01)
@slider("com_radius", 0.0, 0.2, 0.01)
@slider("prior_sigma", 0.0, 1.5, 0.05)
@slider("num_plan_particles", 1, 32, 1)
@slider("num_particles", 8, 1024, 1)
@slider("num_com_particles", 8, 1024, 1)
@slider("localized_std", 0.0, 0.2, 0.005)
@slider("w_info_pos", 0.0, 50.0, 0.5)
@slider("w_info_com", 0.0, 25.0, 0.1)
@slider("info_gate_std", 0.0, 0.5, 0.01)
@slider("com_info_gate_std", 0.0, 0.1, 0.005)
@slider("info_subsample", 1, 20, 1)
@slider("process_noise", 0.0, 0.02, 0.0005)
@slider("com_jitter", 0.0, 0.02, 0.001)
@slider("sigma_gap", 0.002, 0.1, 0.002)
@slider("sigma_contact", 0.005, 0.2, 0.005)
@slider("mismatch_energy", 0.0, 20.0, 0.5)
@slider("contact_tol", 0.0, 0.05, 0.001)
@slider("p_outlier", 0.001, 0.5, 0.005)
@slider("sigma_pos", 0.001, 0.05, 0.001)
@slider("sigma_vel", 0.005, 0.3, 0.005)
@slider("obs_noise_contact", 0.0, 0.1, 0.005)
@slider("obs_noise_pos", 0.0, 0.02, 0.001)
@slider("obs_noise_vel", 0.0, 0.1, 0.005)
@dataclass
class UncertainPosComCylinderPushConfig(CylinderPushConfig):
    """Config for the cylinder push task with an unknown cart position and centre of mass."""

    # goal pose (drawn as the CoM task's see-through goal cart, not the GUI's generic goal sphere)
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
    w_pusher_proximity: float = 2.0
    # priors: position ~ N(prior_mean, prior_sigma^2) around a noisy prior mean; CoM uniform over a disk (radius in m)
    prior_sigma: float = 0.5
    com_radius: float = 0.15
    # belief: num_particles position particles, num_com_particles CoM particles; plans use num_plan_particles pairs
    num_particles: int = 100
    num_com_particles: int = 300
    num_plan_particles: int = 10
    process_noise: float = 0.002
    com_jitter: float = 0.003
    # the CoM belief is updated (and the CoM info term is on) only while the position belief std is below this (m)
    localized_std: float = 0.03
    # contact energy (position)
    sigma_gap: float = 0.02
    sigma_contact: float = 0.03
    mismatch_energy: float = 5.0
    contact_tol: float = 0.005
    # yaw-dynamics energy (CoM); sigmas are rim displacement / rim speed (cart radius x angle / angular velocity)
    sigma_pos: float = 0.005
    sigma_vel: float = 0.05
    # outlier mixture shared by both energies
    p_outlier: float = 0.05
    # observation noise (sim side): contact point (m); IMU yaw and yaw rate, given as rim displacement / rim speed
    obs_noise_contact: float = 0.01
    obs_noise_pos: float = 0.002
    obs_noise_vel: float = 0.01
    # belief-aware planning: contact info term (on while the position std > info_gate_std, full at 2x) and yaw-dynamics
    # info term (on while localized and the CoM std > com_info_gate_std, full at 2x)
    w_info_pos: float = 10.0
    w_info_com: float = 1.0
    info_gate_std: float = 0.05
    com_info_gate_std: float = 0.02
    info_subsample: int = 5
    obs_window: int = 100


class UncertainPosComCylinderPush(UncertainComCylinderPush):
    """Cylinder push to a goal pose where the cart's position and CoM are both latent."""

    name: str = "uncertain_pos_com_cylinder_push"
    config_t: type[UncertainPosComCylinderPushConfig] = UncertainPosComCylinderPushConfig  # type: ignore[assignment]
    config: UncertainPosComCylinderPushConfig  # type: ignore[assignment]
    # Both beliefs are drawn with the CoM marker disc: cart-centre guesses until the cart is localized, then CoM guesses
    # on the estimated cart (see PosComBeliefController.belief_viz).
    belief_geom_name: str = "com_marker"

    # ------------------------------------------------------------------ energies
    @property
    def contact_params(self) -> ContactEnergyParams:
        """Contact energy parameters (geometry is known, taken from the model)."""
        return ContactEnergyParams(
            pusher_radius=float(self.model.geom("pusher").size[0]),
            object_radius=self.cart_radius,
            sigma_gap=self.config.sigma_gap,
            sigma_contact=self.config.sigma_contact,
            p_outlier=self.config.p_outlier,
            contact_tol=self.config.contact_tol,
            mismatch_energy=self.config.mismatch_energy,
        )

    @property
    def energy_params(self) -> DynamicsEnergyParams:
        """Yaw-dynamics energy parameters (features: r*yaw, r*yaw_rate)."""
        return DynamicsEnergyParams(
            num_pos=1,
            sigma_pos=self.config.sigma_pos,
            sigma_vel=self.config.sigma_vel,
            p_outlier=self.config.p_outlier,
        )

    def cart_features(self, states: np.ndarray) -> np.ndarray:
        """What the IMU measures, (..., nq + nv) -> (..., 2): rim displacement r*yaw and rim speed r*yaw_rate."""
        nq, r = self.model.nq, self.cart_radius
        return r * np.stack([states[..., CART_YAW_IDX], states[..., nq + CART_YAW_IDX]], axis=-1)

    # ------------------------------------------------------------------ sim-side: true state, CoM and observations
    def reset(self) -> None:
        """The CoM task's random start and true CoM, plus a noisy prior mean for the cart position."""
        super().reset()
        self.prior_mean = self.data.qpos[CART_XY_IDX] + self.config.prior_sigma * np.random.randn(2)
        self._last_true_cart = self._true_cart_state()

    def _true_cart_state(self) -> np.ndarray:
        """True cart [x, y, vx, vy] (for the oracle baselines only)."""
        return np.concatenate([self.data.qpos[CART_XY_IDX], self.data.qvel[CART_XY_VEL_IDX]])

    def _observe(self) -> np.ndarray:
        """Full state with the cart's x/y position and velocity zeroed and its yaw / yaw rate noisy (IMU).

        The pusher is proprioceptive. The cart's height (it rests on the floor) is not position information.
        """
        state = np.concatenate([self.data.qpos, self.data.qvel])
        nq, r = self.model.nq, self.cart_radius
        state[CART_XY_IDX] = 0.0
        state[nq + CART_XY_VEL_IDX] = 0.0
        state[CART_YAW_IDX] += self.config.obs_noise_pos / r * np.random.randn()
        state[nq + CART_YAW_IDX] += self.config.obs_noise_vel / r * np.random.randn()
        return state

    def _observe_contact(self) -> tuple[float, np.ndarray]:
        """Touch sensor: contact flag and noisy contact point (zero point when not touching), after the step.

        The sensor fires when the true pusher-cart surface gap is within contact_tol, and reports the point on the
        pusher's rim facing the cart: the filter's own observation model (`predicted_contact`) applied to the true
        state. MuJoCo's contact list is not used, for two reasons: after mj_step it describes the state before the
        step, and with the cart riding on the floor the pusher-cart contact chatters by a few mm while pushing, so the
        list drops out at real penetrations. Each false "no contact" penalises exactly the hypotheses in contact.
        """
        pusher_xy, cart_xy = self.data.qpos[PUSHER_QPOS_IDX], self.data.qpos[CART_XY_IDX]
        y_hat, q_hat, _ = predicted_contact(pusher_xy, cart_xy, self.contact_params)
        if y_hat < 0.5:
            return 0.0, np.zeros(2)
        return 1.0, q_hat + self.config.obs_noise_contact * np.random.randn(2)

    def post_sim_step(self) -> None:
        """Record the observed transition and the contact observation after the physics step."""
        obs = self._observe()
        y, q = self._observe_contact()
        self._obs_buffer.append(
            (float(self.data.time), self._last_obs, self.data.ctrl.copy(), obs, y, q, self._last_true_cart)
        )
        self._last_obs = obs
        self._last_true_cart = self._true_cart_state()

    def get_sim_metadata(self) -> dict[str, Any]:
        """Sliding window of observed transitions and contacts, the prior, and the true CoM (oracle/logging only)."""
        buf = list(self._obs_buffer)
        nx = self.model.nq + self.model.nv
        return {
            "episode_id": self.episode_id,
            "prior_mean": np.asarray(self.prior_mean, dtype=float),
            "obs_time": np.array([b[0] for b in buf], dtype=float),
            "obs_pre_state": np.array([b[1] for b in buf], dtype=float).reshape(-1, nx),
            "obs_ctrl": np.array([b[2] for b in buf], dtype=float).reshape(-1, self.model.nu),
            "obs_post_state": np.array([b[3] for b in buf], dtype=float).reshape(-1, nx),
            "obs_y": np.array([b[4] for b in buf], dtype=float),
            "obs_q": np.array([b[5] for b in buf], dtype=float).reshape(-1, 2),
            # Never read by the filter or the planner; used by scripts/baselines_pos_com/oracle*.py and for logging.
            "true_com": np.asarray(self.true_com, dtype=float),
            "obs_true_cart_pre": np.array([b[6] for b in buf], dtype=float).reshape(-1, 4),  # [x, y, vx, vy]
        }

    # ------------------------------------------------------------------ controller-side: belief + action energy
    def make_belief(self) -> PosComBelief:
        """Factored belief: position particles (contact energy) x CoM particles (yaw-dynamics energy), gated."""
        position = ParticleBelief(
            self.model,
            robot_qpos_idx=PUSHER_QPOS_IDX,
            robot_qvel_idx=PUSHER_QVEL_IDX,
            object_qpos_idx=CART_XY_IDX,
            object_qvel_idx=CART_XY_VEL_IDX,
            params=self.contact_params,
            num_particles=self.config.num_particles,
            process_noise=self.config.process_noise,
        )
        com = ParameterBelief(
            self.model,
            params=self.energy_params,
            apply_params=self.apply_params,
            sample_prior=self.sample_prior,
            project=self.project,
            features=self.cart_features,
            num_particles=self.config.num_com_particles,
            jitter=self.config.com_jitter,
        )
        return PosComBelief(position, com, apply_params=self.apply_params, localized_std=self.config.localized_std)

    def info_weights(self, pos_std: float | None, com_std: float | None) -> tuple[float, float]:
        """Effective (contact, yaw-dynamics) info weights given the position and CoM belief stds."""
        cfg = self.config
        w_pos = cfg.w_info_pos * info_ramp(pos_std, cfg.info_gate_std)
        localized = pos_std is None or pos_std < cfg.localized_std
        w_com = cfg.w_info_com * info_ramp(com_std, cfg.com_info_gate_std) if localized else 0.0
        return w_pos, w_com

    def belief_reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        num_particles: int,
        particle_weights: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
        belief_std: float | None = None,
        com_std: float | None = None,
    ) -> np.ndarray:
        """Belief-aware reward: task reward averaged over (position, CoM) guesses + the two gated info terms.

        Args:
            states: Rollout states, shape (K * M, T, nq + nv), ordered action-major (rollout k*M + m uses pair m).
            sensors: Rollout sensors, shape (K * M, T, nsensordata).
            controls: Rollout controls, shape (K * M, T, nu).
            num_particles: M, number of (position, CoM) pairs per action sequence.
            particle_weights: Weights of the M pairs, shape (M,).
            system_metadata: Passed through to the task reward.
            belief_std: Spread sqrt(tr(cov)) of the full position belief (m).
            com_std: Spread sqrt(tr(cov)) of the full CoM belief (m).

        Returns:
            Rewards, shape (K,).
        """
        num_rollouts = states.shape[0] // num_particles
        reward = self.reward(states, sensors, controls, system_metadata).reshape(num_rollouts, num_particles)
        reward = reward @ particle_weights

        w_pos, w_com = self.info_weights(belief_std, com_std)
        if num_particles < 2 or (w_pos <= 0.0 and w_com <= 0.0):
            return reward

        sub = slice(None, None, max(1, self.config.info_subsample))
        traj = states.reshape(num_rollouts, num_particles, states.shape[1], -1)[:, :, sub]
        if w_pos > 0.0:
            reward = reward + w_pos * expected_info_gain(
                traj[..., PUSHER_QPOS_IDX], traj[..., CART_XY_IDX], particle_weights, self.contact_params
            )
        if w_com > 0.0:
            reward = reward + w_com * expected_info_gain_dynamics(
                self.cart_features(traj), particle_weights, self.energy_params
            )
        return reward
