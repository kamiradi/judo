# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Upper bound: plan against the true cart state (perfect knowledge)."""

import numpy as np

from baselines import SINGLE_GUESS, TUNED
from judo.controller.belief_controller import BeliefAwareController

NAME = "oracle"
DESCRIPTION = "Knows the true cart position; plans against it (upper bound)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}

CART_QPOS = [2, 3]
CART_QVEL = [2, 3]


class OracleController(BeliefAwareController):
    """Replaces the sampled plan guess with the true cart state. The belief still runs, for logging only."""

    def _pre_optimization(self) -> None:
        super()._pre_optimization()
        nq = self.model.nq
        xy = self.current_state[CART_QPOS]
        vel = self.current_state[[nq + i for i in CART_QVEL]]
        self.plan_particles = np.concatenate([xy, vel])[None, :]


CONTROLLER_CLS = OracleController
