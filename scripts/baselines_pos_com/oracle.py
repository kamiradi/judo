# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Upper bound: plan with the true cart state and the true centre of mass (perfect knowledge)."""

import numpy as np

from baselines_pos_com import SINGLE_GUESS, TUNED
from judo.controller.pos_com_belief_controller import PosComBeliefController
from judo.tasks.uncertain_pos_com_cylinder_push import CART_XY_IDX, CART_XY_VEL_IDX

NAME = "oracle"
DESCRIPTION = "Knows the true cart position and CoM; plans with them (upper bound)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}


def true_cart_state(ctrl: PosComBeliefController) -> np.ndarray:
    """True cart [x, y, vx, vy] from the measured state (which the stock controller overwrites with guesses)."""
    nq = ctrl.model.nq
    return np.concatenate([ctrl.current_state[CART_XY_IDX], ctrl.current_state[nq + CART_XY_VEL_IDX]])


class OracleController(PosComBeliefController):
    """Replaces the sampled plan pair with the true cart state and true CoM. The beliefs still run, for logging only."""

    def _pre_optimization(self) -> None:
        super()._pre_optimization()
        m = self.num_plan_particles
        self.plan_particles = np.tile(true_cart_state(self), (m, 1))
        true_com = self.system_metadata.get("true_com")
        if true_com is not None:  # published by the sim from its first step on
            self.plan_coms = np.tile(np.asarray(true_com, dtype=float), (m, 1))


CONTROLLER_CLS = OracleController
