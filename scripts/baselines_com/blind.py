# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Lower bound: the full method, but observed motion never updates the CoM belief."""

from baselines_com import TUNED
from judo.controller.param_belief_controller import ParamBeliefController

NAME = "blind"
DESCRIPTION = "Full method, but the belief ignores observations (stays at the prior; lower bound)"
OVERRIDES = dict(TUNED)


class BlindController(ParamBeliefController):
    """The CoM guesses keep their prior weights for the whole episode, so planning averages over the prior."""

    def _init_belief(self, prior_mean=None) -> None:  # noqa: ANN001
        super()._init_belief(prior_mean)
        self.belief.update = lambda *args, **kwargs: None  # type: ignore[method-assign, union-attr]


CONTROLLER_CLS = BlindController
