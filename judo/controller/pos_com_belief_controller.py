# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Belief-aware sampling-based MPC under joint uncertainty in an object's position and a model parameter (its CoM).

The same planning scheme as `BeliefAwareController` and `ParamBeliefController`, with plan hypotheses that are
(position, CoM) pairs:
  1. The factored belief (`judo.belief.PosComBelief`) is updated from the window of observed transitions and contacts
     sent by the simulator: contact energy for the position, gated yaw-dynamics energy for the CoM.
  2. M position guesses and M CoM guesses are drawn from the two marginals and paired; the pairs are shared across all
     K sampled control sequences.
  3. Rollout k*M + m starts from position guess m and uses a model carrying CoM guess m. Rewards come from the task's
     belief-aware reward, which receives both belief spreads to gate its two information terms.

The task must provide `make_belief()`, `apply_params(models, datas, thetas)`, `belief_reward(..., belief_std, com_std)`,
`belief_viz_positions(thetas, state)`, and publish `episode_id`, `prior_mean` and `obs_*` entries in its sim metadata
(see `judo.tasks.uncertain_pos_com_cylinder_push`).
"""

from typing import Any

import numpy as np

from judo.belief import PosComBelief
from judo.controller.belief_controller import BeliefAwareController

# Where belief_viz parks the guesses of the belief that is not shown (far off-screen; the viewer needs a fixed count).
HIDDEN_XY = np.array([1e3, 1e3])


class PosComBeliefController(BeliefAwareController):
    """Controller that plans under a factored belief over an object's position and its CoM."""

    belief: PosComBelief | None  # type: ignore[assignment]

    def reset(self) -> None:
        """Reset controller and belief."""
        super().reset()
        self.plan_coms: np.ndarray | None = None
        self._params_applied = False

    # ------------------------------------------------------------------ estimation
    def _init_belief(self, prior_mean: np.ndarray) -> None:
        self.belief = self.task.make_belief()
        robot_xy = self.current_state[self.belief.pos.robot_qpos_idx]
        self.belief.init_from_prior(np.asarray(prior_mean, dtype=float), self.task.config.prior_sigma, robot_xy)
        self._last_obs_time = -np.inf

    def update_belief(self, metadata: dict[str, Any]) -> None:
        """Filter all observations in the sliding window that are newer than the last processed one."""
        if "episode_id" not in metadata:
            return
        cfg = self.task.config
        episode_id = int(metadata["episode_id"])
        if (
            self.belief is None
            or episode_id != self._episode_id
            or self.belief.pos.num_particles != cfg.num_particles
            or self.belief.com.num_particles != cfg.num_com_particles
        ):
            self._episode_id = episode_id
            self._init_belief(metadata["prior_mean"])
        assert self.belief is not None

        # pick up live config changes from the GUI
        self.belief.pos.params = self.task.contact_params
        self.belief.pos.process_noise = cfg.process_noise
        self.belief.com.params = self.task.energy_params
        self.belief.com.jitter = cfg.com_jitter
        self.belief.localized_std = cfg.localized_std

        obs_time = np.asarray(metadata["obs_time"])
        for i in np.flatnonzero(obs_time > self._last_obs_time):
            self._filter_step(metadata, int(i))
        if len(obs_time) > 0:
            self._last_obs_time = max(self._last_obs_time, float(obs_time[-1]))

    def _filter_step(self, metadata: dict[str, Any], i: int) -> None:
        """Filter observation i of the window (a hook for the oracle baselines)."""
        assert self.belief is not None
        self.belief.step(
            np.asarray(metadata["obs_pre_state"][i]),
            np.asarray(metadata["obs_ctrl"][i]),
            float(metadata["obs_y"][i]),
            np.asarray(metadata["obs_q"][i]),
            np.asarray(metadata["obs_post_state"][i]),
        )

    def belief_viz(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Cart-centre guesses until the cart is localized, then CoM guesses on the estimated cart.

        Always returns num_particles + num_com_particles points (positions (N, 2), weights (N,)); the set not shown is
        parked off-screen with zero weight, so the count stays fixed for the viewer and for recorded trajectories.
        """
        if self.belief is None:
            return None
        pos, com = self.belief.pos, self.belief.com
        if self.belief.localized:
            state = self.belief.pos.full_states(self.current_state, self.belief.mean_object_state()[None])[0]
            com_xy = self.task.belief_viz_positions(com.particles, state)
            xy = np.concatenate([np.tile(HIDDEN_XY, (pos.num_particles, 1)), com_xy])
            w = np.concatenate([np.zeros(pos.num_particles), com.weights])
        else:
            xy = np.concatenate([pos.xy, np.tile(HIDDEN_XY, (com.num_particles, 1))])
            w = np.concatenate([pos.weights, np.zeros(com.num_particles)])
        return xy, w

    # ------------------------------------------------------------------ planning (hooks into Controller.update_action)
    def _pre_optimization(self) -> None:
        """Draw the M (position, CoM) pairs shared by all control samples for this planning step."""
        # Skip BeliefAwareController._pre_optimization (it samples positions only); keep the base Controller's setup.
        super(BeliefAwareController, self)._pre_optimization()
        if self.belief is None:
            self._init_belief(self.task.prior_mean)  # no observations received yet: plan under the prior
        assert self.belief is not None
        self.plan_particles, self.plan_coms = self.belief.sample(self.num_plan_particles)
        self.plan_weights = np.full(self.num_plan_particles, 1.0 / self.num_plan_particles)
        self._params_applied = False

    def _rollout(self) -> None:
        """Roll out each of the K control sequences from each of the M (position, CoM) pairs."""
        assert self.belief is not None and self.plan_particles is not None and self.plan_coms is not None
        num_particles = self.num_plan_particles
        num_rollouts = self.optimizer_cfg.num_rollouts
        if not self._params_applied:
            # MJRolloutBackend keeps one model/data pair per thread, so thread k*M + m can carry CoM guess m.
            thetas = np.tile(self.plan_coms, (num_rollouts, 1))
            self.task.apply_params(self.rollout_backend._models, self.rollout_backend._datas, thetas)  # type: ignore[attr-defined]
            self._params_applied = True
        x0 = self.belief.pos.full_states(self.current_state, self.plan_particles)
        x0 = np.tile(x0, (num_rollouts, 1))
        sim_controls = np.repeat(self.task.task_to_sim_ctrl(self.rollout_controls), num_particles, axis=0)
        self.belief_states, self.belief_sensors, _ = self.rollout_backend.rollout(x0, sim_controls)

    def _post_rollout(self) -> None:
        """Belief-aware rewards, one per control sequence, with both belief spreads for the info-term gates."""
        assert self.belief is not None
        num_particles = self.num_plan_particles
        self.rewards = self.task.belief_reward(
            self.belief_states,
            self.belief_sensors,
            np.repeat(self.rollout_controls, num_particles, axis=0),
            num_particles,
            self.plan_weights,
            self.system_metadata,
            belief_std=self.belief.pos_std(),
            com_std=self.belief.com_std(),
        )
        # expose one representative rollout per control sequence so traces etc. keep working unchanged
        num_rollouts = self.optimizer_cfg.num_rollouts
        self.states = self.belief_states.reshape(num_rollouts, num_particles, *self.belief_states.shape[1:])[:, 0]
        self.sensors = self.belief_sensors.reshape(num_rollouts, num_particles, *self.belief_sensors.shape[1:])[:, 0]
