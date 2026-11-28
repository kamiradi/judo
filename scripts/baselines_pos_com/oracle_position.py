# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Partial oracle: knows the cart's position, filters the CoM. Isolates what the CoM uncertainty costs."""

from typing import Any

import numpy as np

from baselines_pos_com import TUNED
from baselines_pos_com.oracle import true_cart_state
from judo.controller.pos_com_belief_controller import PosComBeliefController

NAME = "oracle_position"
DESCRIPTION = "Knows the true cart position; filters the CoM (gate always open) and averages over CoM guesses"
OVERRIDES = dict(TUNED)


class OraclePositionController(PosComBeliefController):
    """Position particles are collapsed onto the true cart state before every filter step and every plan.

    The gate is then open from the start and the CoM update's plug-in position is the true one.
    """

    def _filter_step(self, metadata: dict[str, Any], i: int) -> None:
        assert self.belief is not None
        pos = self.belief.pos
        pos.particles[:] = np.asarray(metadata["obs_true_cart_pre"][i])
        pos.weights = np.full(pos.num_particles, 1.0 / pos.num_particles)
        super()._filter_step(metadata, i)

    def _pre_optimization(self) -> None:
        super()._pre_optimization()
        self.plan_particles = np.tile(true_cart_state(self), (self.num_plan_particles, 1))


CONTROLLER_CLS = OraclePositionController
