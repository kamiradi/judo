# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""The full method: reward averaged over friction guesses + per-step information gain while the cart is reachable."""

from baselines_strike import TUNED

NAME = "ours"
DESCRIPTION = "Full method: averaged reward over the friction guesses + actionable, per-step information gain"
OVERRIDES = dict(TUNED)
CONTROLLER_CLS = None
