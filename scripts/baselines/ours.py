# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""The full method with stage-2 tuned settings: belief-averaged reward + energy-consistent information gain."""

from baselines import TUNED

NAME = "ours"
DESCRIPTION = "Full method, tuned: averaged reward over 15 guesses + information gain (w_info=10)"
OVERRIDES = dict(TUNED)
CONTROLLER_CLS = None
