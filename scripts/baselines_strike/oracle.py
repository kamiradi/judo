# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Upper bound: plan with the true floor friction (perfect model)."""

import numpy as np

from baselines_strike import SINGLE_GUESS, TUNED
from judo.controller.param_belief_controller import ParamBeliefController

NAME = "oracle"
DESCRIPTION = "Knows the true friction; plans with it (upper bound)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}


class OracleController(ParamBeliefController):
    """Replaces the sampled plan guess with the simulator's true friction. The belief still runs, for logging only."""

    def _pre_optimization(self) -> None:
        super()._pre_optimization()
        true_log_mu = self.system_metadata.get("true_log_mu")
        if true_log_mu is not None:  # published by the sim from its first step on
            self.plan_particles = np.asarray(true_log_mu, dtype=float).reshape(1, 1)


CONTROLLER_CLS = OracleController
