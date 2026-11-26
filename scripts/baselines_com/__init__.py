# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Stage 3 baselines for the CoM task (`--task com` in scripts/ablate_uncertain_push.py), one module per baseline.

Same structure as scripts/baselines/. Each module defines:
    NAME            ablation name used in logs, plots and trajectory files
    DESCRIPTION     one line on what the baseline knows / does
    OVERRIDES       config overrides, same "task.* / controller.* / optimizer.*" keys as the ablation runner
    CONTROLLER_CLS  a ParamBeliefController subclass, or None for the stock CoM-belief controller

Shared rules, so the comparison is fair:
    * Every baseline except `blind` uses the same filter on the same observed transitions.
    * Every baseline uses the tuned filter settings (TUNED, picked from stage 2).
    * Compute is matched at ~480 rollouts per planning step: belief-averaged planners use 32 sequences x 15 guesses,
      single-guess planners use 480 sequences x 1 guess.
    * There is no "defaults" entry: the code defaults are compared in stages 1 and 2. Paired wins and losses are
      counted against "ours".
"""

# The tuned method, picked from the CoM stage 2 results. Empty until stage 2 has run.
TUNED: dict = {}

# Single-guess planners get the same rollout budget as 32 sequences x 15 guesses.
SINGLE_GUESS = {"task.num_plan_particles": 1, "optimizer.num_rollouts": 480}

# Ladder order: bounds first, then one ingredient added at a time, then the method.
ORDER = [
    "oracle",
    "blind",
    "certainty_equivalent",
    "posterior_sampling",
    "no_info",
    "ours",
]
