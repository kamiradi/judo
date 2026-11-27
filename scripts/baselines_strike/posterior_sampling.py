# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Posterior-sampling MPPI: plan with one friction drawn from the belief, redrawn every planning step."""

from baselines_strike import SINGLE_GUESS, TUNED

NAME = "posterior_sampling"
DESCRIPTION = "Same filter; plans with one sampled friction, redrawn every step (no info term)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}  # with one guess the info term is off automatically
CONTROLLER_CLS = None
