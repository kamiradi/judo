# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Stage 3 baselines for scripts/ablate_uncertain_push.py, one module per baseline.

Each module defines:
    NAME            ablation name used in logs, plots and trajectory files
    DESCRIPTION     one line on what the baseline knows / does
    OVERRIDES       config overrides, same "task.* / controller.* / optimizer.*" keys as the ablation runner
    CONTROLLER_CLS  a BeliefAwareController subclass, or None for the stock belief-aware controller

Shared rules, so the comparison is fair:
    * Every baseline except `blind` uses the same particle filter on the same touch observations.
    * Every baseline uses the tuned filter size (num_particles=100).
    * Compute is matched at ~480 rollouts per planning step: belief-averaged planners use 32 sequences x 15 guesses,
      single-guess planners use 480 sequences x 1 guess.
"""

# The tuned method from stage 2 (one of the configs that solved 19/20 seeds).
TUNED = {"task.num_particles": 100, "task.w_info": 10.0}

# Single-guess planners get the same rollout budget as 32 sequences x 15 guesses.
SINGLE_GUESS = {"task.num_plan_particles": 1, "optimizer.num_rollouts": 480}

# Ladder order: bounds first, then one ingredient added at a time, then the method.
ORDER = [
    "defaults",
    "oracle",
    "blind",
    "certainty_equivalent",
    "posterior_sampling",
    "no_info",
    "ours",
]
