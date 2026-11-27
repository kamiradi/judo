# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Stage 3 baselines for the strike task (`--task strike` in scripts/ablate_uncertain_push.py), one module per baseline.

Same structure as scripts/baselines_com/. Each module defines:
    NAME            ablation name used in logs, plots and trajectory files
    DESCRIPTION     one line on what the baseline knows / does
    OVERRIDES       config overrides, same "task.* / controller.* / optimizer.*" keys as the ablation runner
    CONTROLLER_CLS  a ParamBeliefController subclass, or None for the stock parameter-belief controller

Shared rules, so the comparison is fair:
    * Every baseline except `blind` uses the same filter on the same observed transitions.
    * Every baseline uses the tuned settings (TUNED, picked from stage 2).
    * Single-guess planners get as many rollouts as ours (K sequences x M guesses -> K * M sequences x 1 guess).
    * There is no "defaults" entry: the code defaults are compared in stages 1 and 2. Paired wins and losses are
      counted against "ours".
"""

# The tuned method, picked from stage 2. Empty until stage 2 has run (the code defaults are used meanwhile).
TUNED: dict = {}

# Code defaults of the rollout split (judo/optimizers/overrides.py, UncertainFrictionStrikeConfig); the runner's
# check_baseline() keeps STRIKE_BASELINE equal to them.
DEFAULT_NUM_ROLLOUTS = 96
DEFAULT_NUM_PLAN_PARTICLES = 5

# Single-guess planners get the same number of rollouts as ours.
_K = TUNED.get("optimizer.num_rollouts", DEFAULT_NUM_ROLLOUTS)
_M = TUNED.get("task.num_plan_particles", DEFAULT_NUM_PLAN_PARTICLES)
SINGLE_GUESS = {"task.num_plan_particles": 1, "optimizer.num_rollouts": _K * _M}

# Ladder order: bounds first, then one ingredient added at a time, then the method and its information-term ablation.
ORDER = [
    "oracle",
    "blind",
    "certainty_equivalent",
    "posterior_sampling",
    "no_info",
    "ours_path_ig",
    "ours",
]
