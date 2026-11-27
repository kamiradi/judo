# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Belief-aware sampling-based MPC under uncertainty in a model parameter (e.g. a centre of mass).

The same planning scheme as `BeliefAwareController`, with particles that are model parameters instead of object
states:
  1. The belief over the parameter is updated from the window of observed transitions sent by the simulator, using
     the motion-prediction energy (each particle's own model predicts the next state).
  2. M parameter guesses are drawn from the belief and shared across all K sampled control sequences.
  3. Every control sequence is rolled out once per guess (K * M rollouts) from the measured state; rollout k*M + m
     uses a model carrying guess m. Rewards come from the task's belief-aware reward, exactly as for state beliefs.

The task must provide `make_belief()`, `belief_jitter`, `apply_params(models, datas, thetas)`, `belief_reward(...)`,
`belief_viz_positions(thetas, state)`, and publish `episode_id` and `obs_*` entries in its sim metadata
(see `judo.tasks.uncertain_com_cylinder_push`).
"""

from typing import Any

import numpy as np

from judo.belief import ParameterBelief
from judo.controller.belief_controller import BeliefAwareController


class ParamBeliefController(BeliefAwareController):
    """Controller that plans under a particle belief over a model parameter."""

    belief: ParameterBelief | None  # type: ignore[assignment]

    def reset(self) -> None:
        """Reset controller and belief."""
        super().reset()
        self._params_applied = False

    # ------------------------------------------------------------------ estimation
    def _init_belief(self, prior_mean: np.ndarray | None = None) -> None:
        self.belief = self.task.make_belief()
        self.belief.init_from_prior()
        self._last_obs_time = -np.inf

    def update_belief(self, metadata: dict[str, Any]) -> None:
        """Filter all observed transitions in the sliding window that are newer than the last processed one."""
        if "episode_id" not in metadata:
            return
        cfg = self.task.config
        episode_id = int(metadata["episode_id"])
        if self.belief is None or episode_id != self._episode_id or self.belief.num_particles != cfg.num_particles:
            self._episode_id = episode_id
            self._init_belief()
        assert self.belief is not None

        # pick up live config changes from the GUI
        self.belief.params = self.task.energy_params
        self.belief.jitter = self.task.belief_jitter

        obs_time = np.asarray(metadata["obs_time"])
        for i in np.flatnonzero(obs_time > self._last_obs_time):
            self.belief.update(
                np.asarray(metadata["obs_pre_state"][i]),
                np.asarray(metadata["obs_ctrl"][i]),
                np.asarray(metadata["obs_post_state"][i]),
            )
        if len(obs_time) > 0:
            self._last_obs_time = max(self._last_obs_time, float(obs_time[-1]))

    def belief_viz(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Guesses drawn where they would sit on the object in its current pose (N, 2), and their weights (N,)."""
        if self.belief is None:
            return None
        return self.task.belief_viz_positions(self.belief.particles, self.current_state), self.belief.weights.copy()

    # ------------------------------------------------------------------ planning (hooks into Controller.update_action)
    def _pre_optimization(self) -> None:
        """Draw the M parameter guesses shared by all control samples for this planning step."""
        # Skip BeliefAwareController._pre_optimization (it samples object states); keep the base Controller's setup.
        super(BeliefAwareController, self)._pre_optimization()
        if self.belief is None:
            self._init_belief()  # no observations received yet: plan under the prior
        assert self.belief is not None
        self.plan_particles = self.belief.sample(self.num_plan_particles)
        self.plan_weights = np.full(self.num_plan_particles, 1.0 / self.num_plan_particles)
        self._params_applied = False

    def _rollout(self) -> None:
        """Roll out each of the K control sequences under each of the M parameter guesses."""
        assert self.plan_particles is not None
        num_particles = self.num_plan_particles
        num_rollouts = self.optimizer_cfg.num_rollouts
        # Tasks whose reward depends on the guessed parameter (e.g. a predicted resting point) read the guesses here;
        # rollout k*M + m uses plan_params[m]. Set here rather than in _pre_optimization so that subclasses that
        # replace plan_particles (oracle, certainty-equivalent baselines) are seen by the reward too.
        self.task.plan_params = self.plan_particles
        if not self._params_applied:
            # MJRolloutBackend keeps one model/data pair per thread, so thread k*M + m can carry guess m.
            # (Accessing its model list is the only use of the backend's internals.)
            thetas = np.tile(self.plan_particles, (num_rollouts, 1))
            self.task.apply_params(self.rollout_backend._models, self.rollout_backend._datas, thetas)  # type: ignore[attr-defined]
            self._params_applied = True
        x0 = np.tile(self.current_state, (num_rollouts * num_particles, 1))
        sim_controls = np.repeat(self.task.task_to_sim_ctrl(self.rollout_controls), num_particles, axis=0)
        self.belief_states, self.belief_sensors, _ = self.rollout_backend.rollout(x0, sim_controls)
