# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Stage 3 baselines for the joint task (`--task pos_com` in scripts/ablate_uncertain_push.py), one module per baseline.

Same structure as scripts/baselines_com/. Each module defines:
    NAME            ablation name used in logs, plots and trajectory files
    DESCRIPTION     one line on what the baseline knows / does
    OVERRIDES       config overrides, same "task.* / controller.* / optimizer.*" keys as the ablation runner
    CONTROLLER_CLS  a PosComBeliefController subclass, or None for the stock controller

Shared rules, so the comparison is fair:
    * Every baseline except `blind` and the oracles uses the same filters on the same observations.
    * Every baseline uses the tuned settings (TUNED, to be picked from stage 2).
    * Compute is matched at ~480 rollouts per planning step: belief-averaged planners use K sequences x M guesses
      (TUNED), single-guess planners use 480 sequences x 1 guess.
    * Paired wins and losses are counted against "ours".

The ladder has three parts:
    bounds              oracle (knows position and CoM), oracle_position, oracle_com (knows one of the two), blind
    standard baselines  certainty_equivalent, posterior_sampling, no_info
    method ablations    no_pos_info, no_com_info (drop one information term), ungated (CoM updated from the start)
"""

# The tuned method: EMPTY until stage 2, so every baseline runs the code defaults for now (the implementation check).
# Set from POS_COM stage 2 (out/ablations/<run>), with a comment per setting as in scripts/baselines_com/__init__.py.
TUNED: dict = {}

# Single-guess planners get the same rollout budget as the belief-averaged ones (K x M = 48 x 10 at the defaults).
SINGLE_GUESS = {"task.num_plan_particles": 1, "optimizer.num_rollouts": 480}

# Ladder order: bounds first, then one ingredient added at a time, then the method ablations and the method.
ORDER = [
    "oracle",
    "oracle_position",
    "oracle_com",
    "blind",
    "certainty_equivalent",
    "posterior_sampling",
    "no_info",
    "no_pos_info",
    "no_com_info",
    "ungated",
    "ours",
]
