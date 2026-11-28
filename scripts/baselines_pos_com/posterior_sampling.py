# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Posterior-sampling MPPI: plan with one (position, CoM) pair drawn from the beliefs, redrawn every planning step."""

from baselines_pos_com import SINGLE_GUESS, TUNED

NAME = "posterior_sampling"
DESCRIPTION = "Same filters; plans with one sampled (position, CoM) pair, redrawn every step (no info term)"
OVERRIDES = {**TUNED, **SINGLE_GUESS}  # with one guess both info terms are off automatically
CONTROLLER_CLS = None
