# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Expected-reward MPPI: average the task reward over friction guesses, without the information-gain term."""

from baselines_strike import TUNED

NAME = "no_info"
DESCRIPTION = "Same filter; averages the task reward over the friction guesses, no info term"
OVERRIDES = {**TUNED, "task.w_info": 0.0}
CONTROLLER_CLS = None
