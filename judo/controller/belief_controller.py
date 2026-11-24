# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Belief-aware sampling-based MPC (ActiveContact-style).

Each optimization step:
  1. The belief b_t over the latent object state is updated from the window of contact observations sent by the
     simulator (perception energy), propagating particles with the known dynamics.
  2. M particles are drawn from b_t and shared across all K sampled control sequences (common random numbers).
  3. Every control sequence is rolled out once per particle (K * M rollouts) and scored with the task's
     belief-aware reward: task energy averaged over particles + lambda * expected information gain.
  4. The optimizer (e.g. MPPI) reweights the K control sequences exactly as in the fully observed case.

The task must provide `make_belief()`, `belief_reward(...)`, and publish `episode_id`, `prior_mean` and `obs_*` entries
in its sim metadata (see `judo.tasks.uncertain_cylinder_push`).
"""

from typing import Any

import numpy as np

from judo.app.structs import MujocoState
from judo.belief import ParticleBelief
from judo.controller.controller import Controller


class BeliefAwareController(Controller):
    """Controller that plans under a particle belief over the object state."""

    belief: ParticleBelief | None

    @property
    def num_plan_particles(self) -> int:
        """Number of belief particles each control sequence is evaluated under (M)."""
        return max(1, int(self.task.config.num_plan_particles))

    @property
    def num_rollout_threads(self) -> int:
        """K control sequences x M belief particles."""
        return self.optimizer_cfg.num_rollouts * self.num_plan_particles

    # ------------------------------------------------------------------ estimation
    def reset(self) -> None:
        """Reset controller and belief."""
        super().reset()
        self.belief = None
        self._episode_id: int | None = None
        self._last_obs_time = -np.inf
        self.plan_particles: np.ndarray | None = None
        self.plan_weights: np.ndarray | None = None

    def update_states(self, state_msg: MujocoState) -> None:
        """Update the robot state and run the belief filter on newly received observations."""
        super().update_states(state_msg)
        self.update_belief(self.system_metadata)

    def _init_belief(self, prior_mean: np.ndarray) -> None:
        self.belief = self.task.make_belief()
        robot_xy = self.current_state[self.belief.robot_qpos_idx]
        self.belief.init_from_prior(np.asarray(prior_mean, dtype=float), self.task.config.prior_sigma, robot_xy)
        self._last_obs_time = -np.inf

    def update_belief(self, metadata: dict[str, Any]) -> None:
        """Filter all observations in the sliding window that are newer than the last processed one."""
        if "episode_id" not in metadata:
            return
        cfg = self.task.config
        episode_id = int(metadata["episode_id"])
        if self.belief is None or episode_id != self._episode_id or self.belief.num_particles != cfg.num_particles:
            self._episode_id = episode_id
            self._init_belief(metadata["prior_mean"])
        assert self.belief is not None

        # pick up live config changes from the GUI
        self.belief.params = self.task.contact_params
        self.belief.process_noise = cfg.process_noise

        obs_time = np.asarray(metadata["obs_time"])
        for i in np.flatnonzero(obs_time > self._last_obs_time):
            self.belief.propagate(np.asarray(metadata["obs_pre_state"][i]), np.asarray(metadata["obs_ctrl"][i]))
            self.belief.update(
                float(metadata["obs_y"][i]),
                np.asarray(metadata["obs_q"][i]),
                np.asarray(metadata["obs_pusher_xy"][i]),
            )
        if len(obs_time) > 0:
            self._last_obs_time = max(self._last_obs_time, float(obs_time[-1]))

    def belief_viz(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Particle positions (N, 2) and weights (N,) for visualization."""
        if self.belief is None:
            return None
        return self.belief.xy.copy(), self.belief.weights.copy()

    # ------------------------------------------------------------------ planning (hooks into Controller.update_action)
    def _pre_optimization(self) -> None:
        """Draw the M plan particles shared by all control samples for this planning step."""
        super()._pre_optimization()
        if self.belief is None:
            # no observations received yet: fall back to the task's own prior
            self._init_belief(self.task.prior_mean)
        assert self.belief is not None
        self.plan_particles = self.belief.sample(self.num_plan_particles)
        self.plan_weights = np.full(self.num_plan_particles, 1.0 / self.num_plan_particles)

    def _rollout(self) -> None:
        """Roll out each of the K control sequences from each of the M plan particles."""
        assert self.belief is not None and self.plan_particles is not None
        num_particles = self.num_plan_particles
        x0 = self.belief.full_states(self.current_state, self.plan_particles)
        x0 = np.tile(x0, (self.optimizer_cfg.num_rollouts, 1))
        sim_controls = np.repeat(self.task.task_to_sim_ctrl(self.rollout_controls), num_particles, axis=0)
        self.belief_states, self.belief_sensors, _ = self.rollout_backend.rollout(x0, sim_controls)

    def _post_rollout(self) -> None:
        """Belief-aware rewards, one per control sequence."""
        num_particles = self.num_plan_particles
        self.rewards = self.task.belief_reward(
            self.belief_states,
            self.belief_sensors,
            np.repeat(self.rollout_controls, num_particles, axis=0),
            num_particles,
            self.plan_weights,
            self.system_metadata,
            belief_std=float(np.sqrt(np.trace(self.belief.cov()))) if self.belief is not None else None,
        )
        # expose one representative rollout per control sequence so traces etc. keep working unchanged
        num_rollouts = self.optimizer_cfg.num_rollouts
        self.states = self.belief_states.reshape(num_rollouts, num_particles, *self.belief_states.shape[1:])[:, 0]
        self.sensors = self.belief_sensors.reshape(num_rollouts, num_particles, *self.belief_sensors.shape[1:])[:, 0]
