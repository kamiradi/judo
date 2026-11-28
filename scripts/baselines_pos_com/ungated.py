# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Method ablation: no gate, so the CoM is updated (and its info term is on) from the start.

The CoM update then uses the position belief's mean as the plug-in position, however uncertain it still is.
"""

from baselines_pos_com import TUNED

NAME = "ungated"
DESCRIPTION = "Full method without the gate: CoM updated from the start with the (uncertain) mean position"
OVERRIDES = {**TUNED, "task.localized_std": float("inf")}
CONTROLLER_CLS = None
