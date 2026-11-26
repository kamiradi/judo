# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""The full method with the stage-2 tuned settings: reward averaged over CoM guesses + information gain."""

from baselines_com import TUNED

NAME = "ours"
DESCRIPTION = "Full method, tuned: averaged reward over the CoM guesses + information gain"
OVERRIDES = dict(TUNED)
CONTROLLER_CLS = None
