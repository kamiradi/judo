# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Standard MPPI + estimators: plan with the mean cart state and the mean CoM as if they were the truth."""

import numpy as np

from baselines_pos_com import SINGLE_GUESS, TUNED
from judo.controller.pos_com_belief_controller import PosComBeliefController

NAME = "certainty_equivalent"
DESCRIPTION = "Same filters; plans with the mean cart state and the mean CoM only"
OVERRIDES = {**TUNED, **SINGLE_GUESS}


class CertaintyEquivalentController(PosComBeliefController):
    """Replaces the sampled plan pair with the beliefs' weighted means."""

    def _pre_optimization(self) -> None:
        super()._pre_optimization()
        self.plan_particles = self.belief.mean_object_state()[None, :]  # type: ignore[union-attr]
        self.plan_coms = np.asarray(self.belief.com_mean())[None, :]  # type: ignore[union-attr]


CONTROLLER_CLS = CertaintyEquivalentController
