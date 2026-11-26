# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Upper bound: plan with the true centre of mass (perfect model)."""

import numpy as np

from baselines_com import SINGLE_GUESS, TUNED
from judo.controller.param_belief_controller import ParamBeliefController

NAME = "oracle"
DESCRIPTION = "Knows the true CoM; plans with it (upper bound)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}


class OracleController(ParamBeliefController):
    """Replaces the sampled plan guess with the simulator's true CoM. The belief still runs, for logging only."""

    def _pre_optimization(self) -> None:
        super()._pre_optimization()
        true_com = self.system_metadata.get("true_com")
        if true_com is not None:  # published by the sim from its first step on
            self.plan_particles = np.asarray(true_com, dtype=float)[None, :]


CONTROLLER_CLS = OracleController
