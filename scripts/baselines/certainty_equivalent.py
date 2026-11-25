# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Standard MPPI + state estimator: plan against the belief mean as if it were the true cart."""

import numpy as np

from baselines import SINGLE_GUESS, TUNED
from judo.controller.belief_controller import BeliefAwareController

NAME = "certainty_equivalent"
DESCRIPTION = "Same filter; plans against the belief mean only (MPPI + state estimator)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}


class CertaintyEquivalentController(BeliefAwareController):
    """Replaces the sampled plan guess with the weighted mean of the belief (position and velocity)."""

    def _pre_optimization(self) -> None:
        super()._pre_optimization()
        mean_state = self.belief.weights @ self.belief.particles  # type: ignore[union-attr]
        self.plan_particles = np.asarray(mean_state)[None, :]


CONTROLLER_CLS = CertaintyEquivalentController
