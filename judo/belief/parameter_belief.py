# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Particle belief over model parameters (e.g. a centre-of-mass offset), updated with the motion-prediction energy.

The state is observed; what is uncertain is a parameter of the MuJoCo model. Each particle is one parameter vector.
To score it, the particle's own copy of the model simulates one step from the observed pre-step state under the
executed control, and the prediction is compared with the observed post-step state (`dynamics_energy`).
"""

from collections.abc import Callable

import numpy as np
from mujoco import MjData, MjModel

from judo.belief.dynamics_energy import DynamicsEnergyParams, dynamics_energy
from judo.belief.particle_filter import systematic_resample
from judo.utils.mj_rollout_backend import MJRolloutBackend

ApplyParams = Callable[[list[MjModel], list[MjData], np.ndarray], None]


class ParameterBelief:
    """Weighted particles over a parameter vector theta of dimension d.

    Args:
        model: MuJoCo model of the scene, compiled so that every parameter value can be set at runtime.
        params: Motion-prediction energy parameters.
        apply_params: Sets parameters on a list of models (one row of thetas per model); must call mj_setConst.
        sample_prior: Draws n parameter vectors from the prior, shape (n, d).
        project: Maps parameter vectors back into the prior's support, shape (n, d) -> (n, d).
        features: Maps full states (..., nq + nv) to the object features compared by the energy (..., F).
        num_particles: Number of particles.
        jitter: Std of the Gaussian roughening noise added to resampled particles.
    """

    def __init__(
        self,
        model: MjModel,
        params: DynamicsEnergyParams,
        apply_params: ApplyParams,
        sample_prior: Callable[[int], np.ndarray],
        project: Callable[[np.ndarray], np.ndarray],
        features: Callable[[np.ndarray], np.ndarray],
        num_particles: int = 100,
        jitter: float = 0.0,
    ) -> None:
        """Initialize the belief (call `init_from_prior` before use)."""
        self.model = model
        self.params = params
        self.apply_params = apply_params
        self.sample_prior = sample_prior
        self.project = project
        self.features = features
        self.num_particles = num_particles
        self.jitter = jitter
        self.particles = np.zeros((num_particles, 0))
        self.weights = np.full(num_particles, 1.0 / num_particles)
        self._backend = MJRolloutBackend(model, num_threads=num_particles)
        self._applied: np.ndarray | None = None  # particles last written into the backend's models

    # ---------------------------------------------------------------- helpers
    def mean(self) -> np.ndarray:
        """Weighted mean parameter, shape (d,)."""
        return self.weights @ self.particles

    def cov(self) -> np.ndarray:
        """Weighted parameter covariance, shape (d, d)."""
        d = self.particles - self.mean()
        return (self.weights[:, None] * d).T @ d

    def ess(self) -> float:
        """Effective sample size."""
        return float(1.0 / np.sum(np.square(self.weights)))

    def _sync_models(self) -> None:
        """Write the particles into the per-particle models, only when they changed (mj_setConst is not free)."""
        if self._applied is None or not np.array_equal(self._applied, self.particles):
            # MJRolloutBackend keeps one model/data pair per thread; each thread simulates one particle.
            self.apply_params(self._backend._models, self._backend._datas, self.particles)
            self._applied = self.particles.copy()

    # ---------------------------------------------------------------- filter
    def init_from_prior(self) -> None:
        """Draw the particles from the prior with uniform weights."""
        self.particles = np.asarray(self.sample_prior(self.num_particles), dtype=float)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

    def predict(self, pre_state: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        """One physics step from the observed pre-step state under every particle's parameters.

        Args:
            pre_state: Observed full state (nq + nv,).
            ctrl: Executed control (nu,).

        Returns:
            Predicted post-step states, shape (N, nq + nv).
        """
        self._sync_models()
        x0 = np.tile(pre_state, (self.num_particles, 1))
        controls = np.tile(ctrl, (self.num_particles, 1, 1))
        states, _, _ = self._backend.rollout(x0, controls)
        return states[:, -1]

    def update(self, pre_state: np.ndarray, ctrl: np.ndarray, post_state: np.ndarray) -> None:
        """Bayes update with one observed transition, resampling (with jitter) when the ESS collapses."""
        predicted = self.predict(pre_state, ctrl)
        energy = dynamics_energy(self.features(post_state), self.features(predicted), self.params)
        log_w = np.log(np.maximum(self.weights, 1e-300)) - energy
        log_w -= np.logaddexp.reduce(log_w)
        self.weights = np.exp(log_w)
        if self.ess() < 0.5 * self.num_particles:
            self.resample()

    def resample(self) -> None:
        """Systematic resampling with roughening jitter, projected back into the prior's support."""
        idx = systematic_resample(self.weights, self.num_particles)
        particles = self.particles[idx] + self.jitter * np.random.randn(*self.particles.shape)
        self.particles = self.project(particles)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

    def sample(self, num_samples: int) -> np.ndarray:
        """Draw `num_samples` particles (systematic, with replacement) for planning. Shape (num_samples, d)."""
        return self.particles[systematic_resample(self.weights, num_samples)].copy()
