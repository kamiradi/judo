# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Particle belief over a planar object's position, propagated with MuJoCo and updated with the contact energy."""

import numpy as np
from mujoco import MjModel

from judo.belief.contact_energy import ContactEnergyParams, contact_energy
from judo.utils.mj_rollout_backend import MJRolloutBackend


def systematic_resample(weights: np.ndarray, num_samples: int) -> np.ndarray:
    """Systematic resampling. Returns indices of shape (num_samples,)."""
    positions = (np.random.rand() + np.arange(num_samples)) / num_samples
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions)


class ParticleBelief:
    """Weighted particles over an object's planar state [x, y, vx, vy].

    The robot (pusher) state is assumed known (proprioception); only the object's position/velocity are latent. Known
    physical parameters enter through the MuJoCo model used for propagation.

    Args:
        model: MuJoCo model of the scene (robot + object).
        robot_qpos_idx / robot_qvel_idx: indices of the robot's planar position/velocity in qpos/qvel.
        object_qpos_idx / object_qvel_idx: indices of the object's planar position/velocity in qpos/qvel.
        params: Contact energy parameters.
        num_particles: Number of particles.
        process_noise: Std of the Gaussian position noise added at every propagation step.
    """

    def __init__(
        self,
        model: MjModel,
        robot_qpos_idx: np.ndarray,
        robot_qvel_idx: np.ndarray,
        object_qpos_idx: np.ndarray,
        object_qvel_idx: np.ndarray,
        params: ContactEnergyParams,
        num_particles: int = 256,
        process_noise: float = 0.005,
    ) -> None:
        """Initialize the particle belief (call `init_from_prior` before use)."""
        self.model = model
        self.nq, self.nv = model.nq, model.nv
        self.robot_qpos_idx = np.asarray(robot_qpos_idx)
        self.robot_qvel_idx = np.asarray(robot_qvel_idx)
        self.object_qpos_idx = np.asarray(object_qpos_idx)
        self.object_qvel_idx = np.asarray(object_qvel_idx)
        self.params = params
        self.num_particles = num_particles
        self.process_noise = process_noise
        self.particles = np.zeros((num_particles, 4))
        self.weights = np.full(num_particles, 1.0 / num_particles)
        self._backend = MJRolloutBackend(model, num_threads=num_particles)

    # ---------------------------------------------------------------- helpers
    @property
    def xy(self) -> np.ndarray:
        """Particle positions, shape (N, 2)."""
        return self.particles[:, :2]

    def mean(self) -> np.ndarray:
        """Weighted mean position."""
        return self.weights @ self.xy

    def cov(self) -> np.ndarray:
        """Weighted position covariance."""
        d = self.xy - self.mean()
        return (self.weights[:, None] * d).T @ d

    def ess(self) -> float:
        """Effective sample size."""
        return float(1.0 / np.sum(np.square(self.weights)))

    def full_states(self, robot_state: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """Build full MuJoCo states [qpos, qvel] with the known robot state and hypothesized object states.

        Args:
            robot_state: A full state (nq + nv,) whose robot entries are used (object entries are overwritten).
            particles: Object states, shape (P, 4).

        Returns:
            States, shape (P, nq + nv).
        """
        states = np.tile(robot_state, (len(particles), 1))
        states[:, self.object_qpos_idx] = particles[:, :2]
        states[:, self.nq + self.object_qvel_idx] = particles[:, 2:]
        return states

    def _overlaps_robot(self, robot_xy: np.ndarray, xy: np.ndarray) -> np.ndarray:
        return np.linalg.norm(xy - robot_xy, axis=-1) < self.params.pusher_radius + self.params.object_radius

    def _resample_away_from_robot(self, robot_xy: np.ndarray, xy: np.ndarray, sigma: float, center: np.ndarray) -> None:
        """Redraw any particles that penetrate the robot (keeps rollouts physically valid)."""
        for _ in range(100):
            bad = self._overlaps_robot(robot_xy, xy)
            if not bad.any():
                return
            xy[bad] = center[bad] + sigma * np.random.randn(int(bad.sum()), 2)

    # ---------------------------------------------------------------- filter
    def init_from_prior(self, mean: np.ndarray, sigma: float, robot_xy: np.ndarray) -> None:
        """Gaussian prior over position, zero velocity; particles inside the robot are rejected."""
        xy = mean + sigma * np.random.randn(self.num_particles, 2)
        self._resample_away_from_robot(robot_xy, xy, sigma, np.tile(mean, (self.num_particles, 1)))
        self.particles = np.concatenate([xy, np.zeros_like(xy)], axis=-1)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

    def propagate(self, robot_state: np.ndarray, ctrl: np.ndarray) -> None:
        """Advance all particles by one physics step under the executed control.

        Args:
            robot_state: Full pre-step state (nq + nv,); its robot entries are the measured robot state.
            ctrl: Executed control (nu,).
        """
        x0 = self.full_states(robot_state, self.particles)
        controls = np.tile(ctrl, (self.num_particles, 1, 1))
        states, _, _ = self._backend.rollout(x0, controls)
        last = states[:, -1]
        self.particles[:, :2] = last[:, self.object_qpos_idx] + self.process_noise * np.random.randn(
            self.num_particles, 2
        )
        self.particles[:, 2:] = last[:, self.nq + self.object_qvel_idx]

    def update(self, obs_y: float, obs_q: np.ndarray, robot_xy: np.ndarray) -> None:
        """Bayes update with one contact observation, resampling (with jitter) when the ESS collapses."""
        energy = contact_energy(obs_y, obs_q, robot_xy, self.xy, self.params)
        log_w = np.log(np.maximum(self.weights, 1e-300)) - energy
        log_w -= np.logaddexp.reduce(log_w)
        self.weights = np.exp(log_w)
        if self.ess() < 0.5 * self.num_particles:
            self.resample(robot_xy, jitter=0.5 * self.params.sigma_gap)

    def resample(self, robot_xy: np.ndarray, jitter: float) -> None:
        """Systematic resampling with roughening jitter."""
        idx = systematic_resample(self.weights, self.num_particles)
        parents = self.particles[idx].copy()
        xy = parents[:, :2] + jitter * np.random.randn(self.num_particles, 2)
        self._resample_away_from_robot(robot_xy, xy, jitter, parents[:, :2])
        parents[:, :2] = xy
        self.particles = parents
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

    def sample(self, num_samples: int) -> np.ndarray:
        """Draw `num_samples` particles (systematic, with replacement) for planning. Shape (num_samples, 4)."""
        return self.particles[systematic_resample(self.weights, num_samples)].copy()
