# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Lower bound: the full method, but touch never updates the belief."""

from baselines import TUNED
from judo.controller.belief_controller import BeliefAwareController

NAME = "blind"
DESCRIPTION = "Full method, but the belief ignores touch (stays at the prior; lower bound)"
OVERRIDES = dict(TUNED)


class BlindController(BeliefAwareController):
    """Particles are still propagated with the executed controls, but never reweighted or resampled."""

    def _init_belief(self, prior_mean) -> None:  # noqa: ANN001
        super()._init_belief(prior_mean)
        self.belief.update = lambda *args, **kwargs: None  # type: ignore[method-assign, union-attr]


CONTROLLER_CLS = BlindController
