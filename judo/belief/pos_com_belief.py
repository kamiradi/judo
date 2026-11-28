# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Factored belief over an object's planar position and a model parameter (e.g. its centre of mass), with a gate.

b(z, theta) ~ b(z) b(theta):
  * Position particles z (`ParticleBelief`) are propagated with MuJoCo and updated with the contact energy. Each one is
    propagated with a parameter drawn from the parameter belief, so parameter uncertainty spreads the position belief.
  * Parameter particles theta (`ParameterBelief`) are updated with the motion-prediction energy, simulating one step
    from the observed pre-step state with the position belief's mean filled in for the object's position / velocity.

The gate: the parameter belief is only updated while the position belief is tight (sqrt(tr(cov)) < localized_std).
A position error and a parameter can explain the same motion (for a CoM: a sideways position error and a sideways CoM
offset give the same torque), so updating the parameter with a wrong plug-in position would fit a parameter that
compensates for the position error. `localized_std = inf` turns the gate off.

Why factored rather than one particle set over (z, theta): resampling on contact information would keep only the few
parameter values that happen to sit on the surviving position particles (depletion), unless the set is very large, and
every particle carries its own MjModel / MjData.
"""

from collections.abc import Callable

import numpy as np
from mujoco import MjData, MjModel

from judo.belief.parameter_belief import ParameterBelief
from judo.belief.particle_filter import ParticleBelief

ApplyParams = Callable[[list[MjModel], list[MjData], np.ndarray], None]


class PosComBelief:
    """A position belief and a parameter belief, coupled as described in the module docstring.

    Args:
        position: Particle belief over the object's planar state [x, y, vx, vy].
        com: Particle belief over the model parameter.
        apply_params: Sets parameters on a list of models (one row per model); must call mj_setConst.
        localized_std: Position std (m) below which the parameter belief is updated.
    """

    def __init__(
        self, position: ParticleBelief, com: ParameterBelief, apply_params: ApplyParams, localized_std: float
    ) -> None:
        """Initialize (call `init_from_prior` before use)."""
        self.pos = position
        self.com = com
        self.apply_params = apply_params
        self.localized_std = localized_std
        self.num_particles = position.num_particles  # the controller re-creates the belief when the config changes
        self.pos_coms = np.zeros((position.num_particles, 0))  # parameter used to propagate each position particle

    # ---------------------------------------------------------------- summaries
    def mean(self) -> np.ndarray:
        """Weighted mean object position (the primary latent), shape (2,)."""
        return self.pos.mean()

    def cov(self) -> np.ndarray:
        """Weighted object position covariance, shape (2, 2)."""
        return self.pos.cov()

    def pos_std(self) -> float:
        """Position spread sqrt(tr(cov)) (m)."""
        return float(np.sqrt(np.trace(self.pos.cov())))

    def com_mean(self) -> np.ndarray:
        """Weighted mean parameter, shape (d,)."""
        return self.com.mean()

    def com_cov(self) -> np.ndarray:
        """Weighted parameter covariance, shape (d, d)."""
        return self.com.cov()

    def com_std(self) -> float:
        """Parameter spread sqrt(tr(cov))."""
        return float(np.sqrt(np.trace(self.com.cov())))

    @property
    def localized(self) -> bool:
        """Whether the gate is open: the position belief is tight enough to update the parameter belief."""
        return self.pos_std() < self.localized_std

    def mean_object_state(self) -> np.ndarray:
        """Weighted mean of the position particles' full planar state [x, y, vx, vy]."""
        return self.pos.weights @ self.pos.particles

    # ---------------------------------------------------------------- filter
    def init_from_prior(self, mean: np.ndarray, sigma: float, robot_xy: np.ndarray) -> None:
        """Gaussian prior over position (as `ParticleBelief`), the parameter prior, and fresh parameter assignments."""
        self.pos.init_from_prior(mean, sigma, robot_xy)
        self.com.init_from_prior()
        self._assign_coms()

    def _assign_coms(self) -> None:
        """Give each position particle a parameter drawn from the parameter belief (one model per particle)."""
        self.pos_coms = self.com.sample(self.pos.num_particles)
        backend = self.pos._backend  # one model/data pair per position particle
        self.apply_params(backend._models, backend._datas, self.pos_coms)

    def step(
        self, pre_state: np.ndarray, ctrl: np.ndarray, obs_y: float, obs_q: np.ndarray, post_state: np.ndarray
    ) -> bool:
        """Filter one observed transition. Returns whether the parameter belief was updated (gate open).

        Args:
            pre_state: Observed pre-step state (nq + nv,); the object's x/y position and velocity entries are ignored.
            ctrl: Executed control (nu,).
            obs_y: Observed contact flag after the step.
            obs_q: Observed contact point after the step, shape (2,).
            post_state: Observed post-step state (nq + nv,), same convention as pre_state.
        """
        gate_open = self.localized  # decided before the step, from the belief the plug-in position comes from
        if gate_open:
            plug_in = self.pos.full_states(pre_state, self.mean_object_state()[None])[0]

        self.pos.propagate(pre_state, ctrl)
        self.pos.update(obs_y, obs_q, post_state[self.pos.robot_qpos_idx])

        if gate_open:
            self.com.update(plug_in, ctrl, post_state)
            self._assign_coms()
        return gate_open

    # ---------------------------------------------------------------- planning
    def sample(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw `num_samples` (position state, parameter) pairs, independently from the two marginals.

        Returns:
            Object states, shape (num_samples, 4), and parameters, shape (num_samples, d).
        """
        return self.pos.sample(num_samples), self.com.sample(num_samples)
