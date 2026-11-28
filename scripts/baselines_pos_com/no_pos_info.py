# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Method ablation: the full method without the contact (position) information term."""

from baselines_pos_com import TUNED

NAME = "no_pos_info"
DESCRIPTION = "Full method without the contact info term (keeps the gated CoM info term)"
OVERRIDES = {**TUNED, "task.w_info_pos": 0.0}
CONTROLLER_CLS = None
