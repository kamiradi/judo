# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Posterior-sampling MPPI: plan with one CoM drawn from the belief, redrawn every planning step."""

from baselines_com import SINGLE_GUESS, TUNED

NAME = "posterior_sampling"
DESCRIPTION = "Same filter; plans with one sampled CoM, redrawn every step (no info term)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}  # with one guess the info term is off automatically
CONTROLLER_CLS = None
