# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Expected-reward MPPI: average the task reward over (position, CoM) guesses, without either information term."""

from baselines_pos_com import TUNED

NAME = "no_info"
DESCRIPTION = "Same filters; averages the task reward over the guesses, no info terms"
OVERRIDES = {**TUNED, "task.w_info_pos": 0.0, "task.w_info_com": 0.0}
CONTROLLER_CLS = None
