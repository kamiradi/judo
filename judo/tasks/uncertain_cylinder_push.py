# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Cylinder push with an uncertain object (cart) position, observed only through pusher contacts.

The same task class plays two roles:
  * In the simulation process it generates noisy contact observations from the true MuJoCo state and publishes a
    sliding window of them (together with the executed controls) through `get_sim_metadata`.
  * In the controller process it provides the belief (particle filter) and the belief-aware reward
    (task energy averaged over belief particles + lambda * expected information gain), used by
    `judo.controller.belief_controller.BeliefAwareController`.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from judo.belief import ContactEnergyParams, ParticleBelief, expected_info_gain
from judo.gui import slider
from judo.tasks.cylinder_push import XML_PATH, CylinderPush, CylinderPushConfig

PUSHER_QPOS_IDX = np.array([0, 1])
CART_QPOS_IDX = np.array([2, 3])
PUSHER_QVEL_IDX = np.array([0, 1])
CART_QVEL_IDX = np.array([2, 3])


@slider("w_info", 0.0, 10.0, 0.1)
@slider("num_plan_particles", 1, 32, 1)
@slider("num_particles", 8, 1024, 1)
@slider("prior_sigma", 0.0, 1.5, 0.05)
@slider("process_noise", 0.0, 0.02, 0.0005)
@slider("sigma_gap", 0.002, 0.1, 0.002)
@slider("sigma_contact", 0.005, 0.2, 0.005)
@slider("p_outlier", 0.001, 0.5, 0.005)
@slider("mismatch_energy", 0.0, 20.0, 0.5)
@slider("contact_tol", 0.0, 0.05, 0.001)
@slider("obs_noise", 0.0, 0.1, 0.005)
@slider("p_flip", 0.0, 0.5, 0.01)
@slider("info_subsample", 1, 20, 1)
@dataclass
class UncertainCylinderPushConfig(CylinderPushConfig):
    """Config for the uncertain cylinder push task."""

    # belief
    num_particles: int = 30
    num_plan_particles: int = 8
    prior_sigma: float = 0.5
    process_noise: float = 0.002
    # contact observation model / energy
    sigma_gap: float = 0.02
    sigma_contact: float = 0.03
    p_outlier: float = 0.05
    mismatch_energy: float = 5.0
    contact_tol: float = 0.005
    obs_noise: float = 0.01
    p_flip: float = 0.0
    # belief-aware planning
    w_info: float = 1.0
    info_subsample: int = 5
    obs_window: int = 100


class UncertainCylinderPush(CylinderPush):
    """Cylinder push where the cart position is latent and must be inferred from contact observations."""

    name: str = "uncertain_cylinder_push"
    config_t: type[UncertainCylinderPushConfig] = UncertainCylinderPushConfig  # type: ignore[assignment]
    config: UncertainCylinderPushConfig
    belief_geom_name: str = "cart"  # geometry used to render belief particles in the visualizer

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the uncertain cylinder push task."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)
        self.pusher_geom_id = self.model.geom("pusher").id
        self.cart_geom_id = self.model.geom("cart").id

    # ------------------------------------------------------------------ geometry / energy
    @property
    def contact_params(self) -> ContactEnergyParams:
        """Contact energy parameters (geometry is known, taken from the model)."""
        return ContactEnergyParams(
            pusher_radius=float(self.model.geom("pusher").size[0]),
            object_radius=float(self.model.geom("cart").size[0]),
            sigma_gap=self.config.sigma_gap,
            sigma_contact=self.config.sigma_contact,
            p_outlier=self.config.p_outlier,
            contact_tol=self.config.contact_tol,
            mismatch_energy=self.config.mismatch_energy,
        )

    # ------------------------------------------------------------------ sim-side: observation generation
    def reset(self) -> None:
        """Resets the true state and draws a new (noisy) prior mean for the cart position."""
        super().reset()
        self.episode_id = int(np.random.randint(1, 2**31 - 1))
        self.prior_mean = self.data.qpos[CART_QPOS_IDX] + self.config.prior_sigma * np.random.randn(2)
        self._obs_buffer: deque = deque(maxlen=self.config.obs_window)
        self._pre_step_state = np.concatenate([self.data.qpos, self.data.qvel])

    def pre_sim_step(self) -> None:
        """Record the pre-step state (only the robot part is used by the estimator)."""
        self._pre_step_state = np.concatenate([self.data.qpos, self.data.qvel])

    def post_sim_step(self) -> None:
        """Record a noisy contact observation after the physics step."""
        y, q = 0.0, np.zeros(2)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if {c.geom1, c.geom2} == {self.pusher_geom_id, self.cart_geom_id}:
                y, q = 1.0, c.pos[:2].copy()
                break
        if np.random.rand() < self.config.p_flip:
            y = 1.0 - y
        if y > 0.5:
            q = q + self.config.obs_noise * np.random.randn(2)
        robot_state = self._pre_step_state.copy()
        robot_state[CART_QPOS_IDX] = 0.0  # never leak the true object state to the estimator
        robot_state[self.model.nq + CART_QVEL_IDX] = 0.0
        self._obs_buffer.append(
            (
                float(self.data.time),
                robot_state,
                self.data.ctrl.copy(),
                self.data.qpos[PUSHER_QPOS_IDX].copy(),
                y,
                q,
            )
        )

    def get_sim_metadata(self) -> dict[str, Any]:
        """Sliding window of contact observations O^- and executed actions A^-, plus the prior."""
        buf = list(self._obs_buffer)
        nx = self.model.nq + self.model.nv
        return {
            "episode_id": self.episode_id,
            "prior_mean": np.asarray(self.prior_mean, dtype=float),
            "obs_time": np.array([b[0] for b in buf], dtype=float),
            "obs_pre_state": np.array([b[1] for b in buf], dtype=float).reshape(-1, nx),
            "obs_ctrl": np.array([b[2] for b in buf], dtype=float).reshape(-1, self.model.nu),
            "obs_pusher_xy": np.array([b[3] for b in buf], dtype=float).reshape(-1, 2),
            "obs_y": np.array([b[4] for b in buf], dtype=float),
            "obs_q": np.array([b[5] for b in buf], dtype=float).reshape(-1, 2),
        }

    # ------------------------------------------------------------------ controller-side: belief + action energy
    def make_belief(self) -> ParticleBelief:
        """Creates a particle belief over the cart state using this task's (known) model."""
        return ParticleBelief(
            self.model,
            robot_qpos_idx=PUSHER_QPOS_IDX,
            robot_qvel_idx=PUSHER_QVEL_IDX,
            object_qpos_idx=CART_QPOS_IDX,
            object_qvel_idx=CART_QVEL_IDX,
            params=self.contact_params,
            num_particles=self.config.num_particles,
            process_noise=self.config.process_noise,
        )

    def belief_reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        num_particles: int,
        particle_weights: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Belief-aware reward: -(E_task + lambda * E_info) per action sequence.

        Args:
            states: Rollout states, shape (K * M, T, nq + nv), ordered action-major (rollout k*M + m).
            sensors: Rollout sensors, shape (K * M, T, nsensordata).
            controls: Rollout controls, shape (K * M, T, nu).
            num_particles: M, number of belief particles per action sequence.
            particle_weights: Weights of the M plan particles, shape (M,).
            system_metadata: Passed through to the task reward.

        Returns:
            Rewards, shape (K,).
        """
        num_rollouts = states.shape[0] // num_particles
        # Task energy under the belief: the standard cylinder push reward averaged over particles.
        task_reward = self.reward(states, sensors, controls, system_metadata).reshape(num_rollouts, num_particles)
        task_reward = task_reward @ particle_weights

        if self.config.w_info <= 0.0 or num_particles < 2:
            return task_reward

        sub = slice(None, None, max(1, self.config.info_subsample))
        traj = states.reshape(num_rollouts, num_particles, states.shape[1], -1)[:, :, sub]
        info_gain = expected_info_gain(
            traj[..., PUSHER_QPOS_IDX], traj[..., CART_QPOS_IDX], particle_weights, self.contact_params
        )
        return task_reward + self.config.w_info * info_gain
