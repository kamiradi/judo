# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Reference: the full method at the code defaults (the stage 1/2 baseline)."""

NAME = "baseline"  # the runner compares every ablation against the one named "baseline"
DESCRIPTION = "Full method at code defaults (30 particles, w_info=1)"
OVERRIDES: dict = {}
CONTROLLER_CLS = None
