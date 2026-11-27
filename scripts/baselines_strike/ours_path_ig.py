# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Ours with the information term used in the position and CoM tasks: whole-path energies, every step counted.

Isolates the two changes made for this task (per-step increments, and counting only steps at which the cart is still
reachable). Path energies saturate for a strike as much as for a poke, so this variant has no reason to probe first.
"""

from baselines_strike import TUNED

NAME = "ours_path_ig"
DESCRIPTION = "Ours with the path-energy info term over the whole horizon (as in the position/CoM tasks)"
OVERRIDES = {**TUNED, "task.info_mode": "path", "task.info_reachable_only": False}
CONTROLLER_CLS = None
