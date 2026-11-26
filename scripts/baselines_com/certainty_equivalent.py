# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Standard MPPI + parameter estimator: plan with the belief's mean CoM as if it were the true one."""

import numpy as np

from baselines_com import SINGLE_GUESS, TUNED
from judo.controller.param_belief_controller import ParamBeliefController

NAME = "certainty_equivalent"
DESCRIPTION = "Same filter; plans with the mean CoM only (before any contact: the cart's centre)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}


class CertaintyEquivalentController(ParamBeliefController):
    """Replaces the sampled plan guess with the weighted mean of the CoM belief."""

    def _pre_optimization(self) -> None:
        super()._pre_optimization()
        self.plan_particles = np.asarray(self.belief.mean())[None, :]  # type: ignore[union-attr]


CONTROLLER_CLS = CertaintyEquivalentController
