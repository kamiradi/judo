# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""The full method: reward averaged over (position, CoM) guesses + gated contact and CoM information terms."""

from baselines_pos_com import TUNED

NAME = "ours"
DESCRIPTION = "Full method: averaged reward over the guesses + gated information terms"
OVERRIDES = dict(TUNED)
CONTROLLER_CLS = None
