# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Partial oracle: knows the CoM, filters the cart's position. Isolates what the position uncertainty costs."""

import numpy as np

from baselines_pos_com import TUNED
from judo.controller.pos_com_belief_controller import PosComBeliefController

NAME = "oracle_com"
DESCRIPTION = "Knows the true CoM (filter and plans use it); filters the position and averages over position guesses"
OVERRIDES = dict(TUNED)


class OracleComController(PosComBeliefController):
    """Every CoM particle is the true CoM and is never updated, so position particles propagate with the true CoM."""

    def _init_belief(self, prior_mean: np.ndarray) -> None:
        super()._init_belief(prior_mean)
        true_com = self.system_metadata.get("true_com")
        if true_com is None:  # planning before the first observation: the belief is re-created when it arrives
            return
        com = self.belief.com  # type: ignore[union-attr]
        com.particles = np.tile(np.asarray(true_com, dtype=float), (com.num_particles, 1))
        com.weights = np.full(com.num_particles, 1.0 / com.num_particles)
        com.update = lambda *args, **kwargs: None  # type: ignore[method-assign]
        self.belief._assign_coms()  # type: ignore[union-attr]


CONTROLLER_CLS = OracleComController
