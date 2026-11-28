# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Method ablation: the full method without the yaw-dynamics (CoM) information term."""

from baselines_pos_com import TUNED

NAME = "no_com_info"
DESCRIPTION = "Full method without the CoM info term (keeps the contact info term)"
OVERRIDES = {**TUNED, "task.w_info_com": 0.0}
CONTROLLER_CLS = None
