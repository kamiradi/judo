# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Lower bound: the full method, but no observation ever updates either belief."""

from baselines_pos_com import TUNED
from judo.controller.pos_com_belief_controller import PosComBeliefController

NAME = "blind"
DESCRIPTION = "Full method, but touch and motion never update the beliefs (they stay at the priors; lower bound)"
OVERRIDES = dict(TUNED)


class BlindController(PosComBeliefController):
    """Position particles are still propagated with the executed controls, but neither belief is ever reweighted."""

    def _init_belief(self, prior_mean) -> None:  # noqa: ANN001
        super()._init_belief(prior_mean)
        self.belief.pos.update = lambda *args, **kwargs: None  # type: ignore[method-assign, union-attr]
        self.belief.com.update = lambda *args, **kwargs: None  # type: ignore[method-assign, union-attr]


CONTROLLER_CLS = BlindController
