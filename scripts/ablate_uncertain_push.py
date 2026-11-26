# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Headless ablations for the uncertain cylinder push tasks: which parameters raise the success rate?

Two tasks, chosen with --task:
    position  `uncertain_cylinder_push`      the cart's position is unknown (state uncertainty)
    com       `uncertain_com_cylinder_push`  the cart's centre of mass is unknown (dynamics uncertainty)

Read this file top to bottom:
    BASELINE / COM_BASELINE  - the reference settings every ablation is compared against.
    STAGE_1 / COM_STAGE_1    - one parameter changed at a time from the baseline.
    STAGE_2 / COM_STAGE_2    - a small grid over the parameters that mattered most in stage 1 (edit after stage 1).
    STAGE_3 / COM_STAGE_3    - the tuned method against baselines (scripts/baselines/, scripts/baselines_com/).
    TASKS         - everything task-specific: which latent quantity the belief tracks, what counts as a hard start.
    run_episode() - one closed-loop episode: sim + belief-aware MPPI, exactly as in the live app's loop.
    main()        - runs every (ablation, seed) pair and writes the logs described below.

Usage (from the repo root, inside the pixi shell):
    python scripts/ablate_uncertain_push.py --stage 1                      # position task, 25 ablations x 10 seeds
    python scripts/ablate_uncertain_push.py --stage 1 --only baseline --only w_info=10.0 --seeds 30
    python scripts/ablate_uncertain_push.py --stage 2 --seeds 20           # 18 grid cells x 20 seeds
    python scripts/ablate_uncertain_push.py --task com --stage 1           # CoM task

Output, in out/ablations/<timestamp>_<task>_stage<N>/:
    run_config.json          baseline, ablations, seeds, duration, git commit
    episodes.csv             one row per (ablation, seed): overrides + metrics (the raw data)
    summary.csv              one row per ablation: success rate +- std error, delta and paired wins/losses vs baseline
    trajectories/<name>.h5   every episode of that ablation, for replay with scripts/replay_uncertain_push.py

Seeds are paired: seed s gives the same initial pusher/cart positions and the same prior noise (position task) or
true centre of mass (CoM task) in every ablation, so "w_info=10 won on seed 3 where baseline failed" is a
like-for-like comparison.

Caveats:
    * Fidelity: here the controller plans synchronously every `steps_per_plan` sim steps (25 Hz by default) with no
      planning latency. The live app plans asynchronously at 20 Hz with real latency. Rankings should transfer,
      absolute success rates may not.
    * Statistics: with 10 seeds the standard error of a success rate is up to +-16 percentage points, so differences
      below ~30 points from a single ablation are not conclusive. Rerun finalists with more seeds (--seeds 30 --only).
"""

import argparse
import csv
import importlib
import itertools
import json
import multiprocessing
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

OPTIMIZER = "mppi"
SIM_DT = 0.02  # <option timestep> of judo/models/xml/cylinder_push.xml, shared by both tasks
REPO_ROOT = Path(__file__).resolve().parents[1]

# ============================================================================================================== #
# Ablations                                                                                                      #
#                                                                                                                #
# An ablation is a name plus a dict of overrides. Keys say which config the field lives in:                      #
#     "task.<field>"        UncertainCylinderPushConfig  (judo/tasks/uncertain_cylinder_push.py)                 #
#     "controller.<field>"  ControllerConfig              (judo/controller/controller.py)                        #
#     "optimizer.<field>"   MPPIConfig                    (judo/optimizers/mppi.py)                              #
# Anything not overridden keeps the value the task/controller would use in the live app.                         #
# ============================================================================================================== #

# The defaults as of this commit, written out so the reference point is visible. These are NOT applied as
# overrides (the code defaults are used); `check_baseline()` asserts they still match the code.
BASELINE = {
    "task.w_info": 1.0,
    "task.num_plan_particles": 15,
    "task.num_particles": 30,
    "task.info_gate_std": 0.05,
    "task.info_subsample": 5,
    "task.prior_sigma": 0.5,
    "task.process_noise": 0.002,
    "task.mismatch_energy": 5.0,
    "task.p_outlier": 0.05,
    "task.w_pusher_proximity": 0.5,
    "controller.horizon": 1.0,
    "optimizer.num_rollouts": 32,
    "optimizer.temperature": 0.05,
}


def sweep(key: str, values: list) -> list[tuple[str, dict]]:
    """One ablation per value of a single parameter, named e.g. 'w_info=10'."""
    short = key.split(".", 1)[1]
    return [(f"{short}={v}", {key: v}) for v in values]


def grid(axes: dict[str, list]) -> list[tuple[str, dict]]:
    """Every combination of the given parameter values, named e.g. 'w_info=10,horizon=1.5'."""
    keys = list(axes)
    ablations = []
    for values in itertools.product(*axes.values()):
        overrides = dict(zip(keys, values, strict=True))
        name = ",".join(f"{k.split('.', 1)[1]}={v}" for k, v in overrides.items())
        ablations.append((name, overrides))
    return ablations


# Stage 1: one parameter at a time. The comment says which failure mechanism each sweep probes.
STAGE_1 = [
    ("baseline", {}),
    *sweep("task.w_info", [0.0, 5.0, 10.0, 25.0, 50.0]),  # info term too weak relative to the task term
    *sweep("controller.horizon", [1.5, 2.0]),  # info term is 0 when no rollout reaches a hypothesis
    *sweep("task.info_gate_std", [0.0, 0.1]),  # when the info term switches off
    *sweep("task.num_plan_particles", [1, 4, 32]),  # averaging over hypotheses / dithering between them
    *sweep("task.w_pusher_proximity", [0.1, 2.0]),  # pull toward the particle-averaged pusher target
    *sweep("task.num_particles", [100, 300]),  # particle depletion -> belief locks onto a phantom cart
    *sweep("task.process_noise", [0.0, 0.01]),  # ability of the belief to recover after collapsing
    *sweep("task.mismatch_energy", [2.0, 10.0]),  # how strongly a miss rules out a touched hypothesis
    *sweep("task.p_outlier", [0.01, 0.2]),  # cap on how much one observation can move the belief
    *sweep("optimizer.temperature", [0.01, 0.5]),  # how greedily MPPI picks among the sampled actions
]

# Stage 2: a grid over the parameters that raised success most in stage 1 without breaking the live time budget.
# Each axis includes its baseline value so the grid also contains the single-parameter effects. Left out on purpose:
#   controller.horizon, num_plan_particles (larger) - planning time exceeds the live app's 50 ms tick.
#   optimizer.temperature                          - no value improved on the baseline.
# Run with more seeds than stage 1: single-parameter winners tend to tie, and few seeds cannot separate them.
STAGE_2 = [
    ("baseline", {}),
    *[
        (name, overrides)
        for name, overrides in grid(
            {
                "task.w_info": [1.0, 10.0, 25.0],
                "task.w_pusher_proximity": [0.1, 0.5],
                "task.num_particles": [30, 100, 300],
            }
        )
        if any(BASELINE[k] != v for k, v in overrides.items())  # the all-baseline cell is "baseline" above
    ],
]


def baseline_ablations(package_name: str = "baselines") -> list[tuple[str, dict]]:
    """Stage 3: one ablation per module in scripts/<package_name>/, in ladder order (see its __init__.py).

    A baseline with its own controller class carries the key "controller_cls" (the module name, so it can be sent to
    worker processes); run_episode() passes that class to make_controller.
    """
    # scripts/baselines*/ are importable because this script lives in scripts/
    package = importlib.import_module(package_name)

    ablations = []
    for module_name in package.ORDER:
        module = importlib.import_module(f"{package_name}.{module_name}")
        overrides = dict(module.OVERRIDES)
        if module.CONTROLLER_CLS is not None:
            overrides["controller_cls"] = module.__name__
        ablations.append((module.NAME, overrides))
    return ablations


# Stage 3: the method against simpler baselines that get the same touch data and the same rollout budget.
STAGE_3 = baseline_ablations()

STAGES = {1: STAGE_1, 2: STAGE_2, 3: STAGE_3}


# ---------------------------------------------------------------------------------------------------------------- #
# CoM task: `uncertain_com_cylinder_push` (judo/tasks/uncertain_com_cylinder_push.py). The cart's pose is observed;  #
# its centre of mass is not. The goal is a position and an angle.                                                  #
# ---------------------------------------------------------------------------------------------------------------- #

COM_BASELINE = {
    "task.w_info": 1.0,
    "task.num_plan_particles": 15,
    "task.num_particles": 30,
    "task.info_gate_std": 0.02,
    "task.info_subsample": 5,
    "task.com_radius": 0.15,
    "task.com_jitter": 0.003,
    "task.sigma_pos": 0.005,
    "task.sigma_vel": 0.05,
    "task.p_outlier": 0.05,
    "task.w_pusher_proximity": 0.5,
    "task.w_cart_yaw": 1.0,
    "task.yaw_tol": 0.1,
    "controller.horizon": 1.0,
    "optimizer.num_rollouts": 32,
    "optimizer.temperature": 0.05,
}

# Stage 1: one parameter at a time. The prior (com_radius) and the success test (yaw_tol) define the experiment and
# are not tuned.
COM_STAGE_1 = [
    ("baseline", {}),
    *sweep("task.w_info", [0.0, 10.0, 25.0]),  # info term too weak / too strong relative to the task term
    *sweep("task.num_particles", [50, 100, 300]),  # depletion: 30 guesses are ~5 cm apart over the prior disk
    *sweep("task.num_plan_particles", [5, 15, 30]),  # averaging over guesses vs planning time (15 = baseline)
    *sweep("task.sigma_vel", [0.02, 0.15]),  # how sharply one observed transition separates guesses
    *sweep("task.p_outlier", [0.01, 0.2]),  # cap on how much one transition can move the belief
    *sweep("task.com_jitter", [0.0, 0.01]),  # ability of resampled guesses to move towards the truth
    *sweep("task.info_gate_std", [0.0, 0.05]),  # when the info term switches off
    *sweep("task.w_pusher_proximity", [0.1, 2.0]),  # pull to the point behind the cart vs off-centre pushes
    *sweep("task.w_cart_yaw", [0.5, 2.0]),  # angle term vs position terms
    *sweep("optimizer.temperature", [0.01, 0.2]),  # how greedily MPPI picks among the sampled actions
    *sweep("controller.horizon", [0.5, 1.5]),  # how far ahead a push's effect on the angle is seen
]

# Stage 2: a grid over the parameters that raised success in stage 1, crossed with how the ~480-rollout budget is
# split between control sequences (K) and CoM guesses (M). Reasoning:
#   num_particles, horizon, w_pusher_proximity - the stage 1 settings that won hard starts (large initial angle error).
#   w_info                                     - kept in to test whether the info term helps once filter and control
#                                                are better (at the defaults it changed nothing, and 10+ hurt).
#   K x M                                      - single-guess planners search 15x more control sequences on the same
#                                                budget; this tests whether fewer guesses and more sequences help ours.
# Left out: temperature (its stage 1 gain was within noise), the filter's energy settings (no effect).
# Run with 20 seeds: only seeds with a large initial angle error separate the settings.
COM_ROLLOUT_SPLITS = {"32x15": (32, 15), "96x5": (96, 5)}  # K x M, both ~480 rollouts per planning step
COM_STAGE_2 = [
    ("baseline", {}),
    *[
        (
            f"{name},KxM={split}",
            {**overrides, "optimizer.num_rollouts": k, "task.num_plan_particles": m},
        )
        for name, overrides in grid(
            {
                "task.num_particles": [100, 300],
                "controller.horizon": [1.0, 1.5],
                "task.w_pusher_proximity": [0.5, 2.0],
                "task.w_info": [0.0, 1.0, 5.0],
            }
        )
        for split, (k, m) in COM_ROLLOUT_SPLITS.items()
    ],
]

# Stage 3: the baseline ladder in scripts/baselines_com/ (no "defaults" entry: stages 1-2 already compare them).
COM_STAGES = {1: COM_STAGE_1, 2: COM_STAGE_2, 3: baseline_ablations("baselines_com")}


# ---------------------------------------------------------------------------------------------------------------- #
# Task-specific pieces. The belief error is |belief mean - latent truth| in both tasks (metres).                    #
# ---------------------------------------------------------------------------------------------------------------- #


def _cart_xy(sim) -> np.ndarray:  # noqa: ANN001
    return np.asarray(sim.task.data.qpos[2:4])


def _true_com(sim) -> np.ndarray:  # noqa: ANN001
    return np.asarray(sim.task.true_com)


@dataclass(frozen=True)
class TaskSetup:
    """What differs between the tasks."""

    task: str  # registered task name
    baseline: dict
    stages: dict[int, list]
    latent: Callable  # sim -> the quantity the belief tracks (true cart xy / true CoM offset)
    prior_center: Callable  # sim -> centre of the prior at t=0 (prior mean / zero offset)
    reference: Callable  # sim -> xy of the marker drawn in replay (prior mean / true CoM on the cart)
    hard_threshold: float  # prior error above which a start counts as hard (half the seeds, roughly)


TASKS = {
    "position": TaskSetup(
        task="uncertain_cylinder_push",
        baseline=BASELINE,
        stages=STAGES,
        latent=_cart_xy,
        prior_center=lambda sim: np.asarray(sim.task.prior_mean),
        reference=lambda sim: np.asarray(sim.task.prior_mean),
        hard_threshold=BASELINE["task.prior_sigma"],
    ),
    "com": TaskSetup(
        task="uncertain_com_cylinder_push",
        baseline=COM_BASELINE,
        stages=COM_STAGES,
        latent=_true_com,
        prior_center=lambda sim: np.zeros(2),
        reference=lambda sim: sim.task.belief_viz_positions(
            _true_com(sim)[None], np.concatenate([sim.task.data.qpos, sim.task.data.qvel])
        )[0],
        # median distance of a uniform draw over the prior disk from its centre
        hard_threshold=COM_BASELINE["task.com_radius"] / np.sqrt(2),
    ),
}


# ============================================================================================================== #
# One episode                                                                                                    #
# ============================================================================================================== #


@dataclass
class Episode:
    """Metrics (and optionally the full trajectory) of one closed-loop episode."""

    ablation: str
    seed: int
    success: bool
    success_time: float  # first time the task's success test passes (nan if never)
    first_contact_time: float  # first time pusher and cart touch (nan if never)
    contact_fraction: float  # fraction of sim steps with pusher-cart contact
    mean_belief_err: float  # mean over plan steps of |belief mean - truth| (cart position / CoM offset, metres)
    final_belief_err: float
    final_belief_std: float  # sqrt(trace(belief covariance)) at the end
    mean_plan_ms: float  # mean wall time of one update_action call
    prior_err: float  # |prior centre - truth| at t=0: how wrong the starting guess is
    identify_time: float  # first plan step at which the belief std drops below info_gate_std (nan if never)
    final_pos_err: float  # |cart - goal| at the end
    final_yaw_err: float  # |cart angle - goal angle| at the end (nan for tasks without an angle goal)
    belief_err_trace: list[float] = field(default_factory=list, repr=False)  # |belief mean - truth| per plan step
    # trajectory, only kept when recording
    qpos: np.ndarray | None = field(default=None, repr=False)  # (T, nq)
    belief_xy: np.ndarray | None = field(default=None, repr=False)  # (T, N, 2) from belief_viz(), held between plans
    belief_w: np.ndarray | None = field(default=None, repr=False)  # (T, N)
    prior_mean: np.ndarray | None = field(default=None, repr=False)  # (2,), position task only
    reference_xy: np.ndarray | None = field(default=None, repr=False)  # (T, 2) replay marker (TaskSetup.reference)
    goal_pos: np.ndarray | None = field(default=None, repr=False)  # (2,)

    METRICS = (
        "success",
        "success_time",
        "first_contact_time",
        "contact_fraction",
        "mean_belief_err",
        "final_belief_err",
        "final_belief_std",
        "mean_plan_ms",
        "prior_err",
        "identify_time",
        "final_pos_err",
        "final_yaw_err",
    )


def apply_overrides(sim, ctrl, overrides: dict) -> None:  # noqa: ANN001
    """Set each override on the right config object.

    Task overrides go to both the sim's and the controller's task, as the GUI does, because some task fields
    (obs_noise, p_flip, obs_window, prior_sigma) are read on the sim side.
    """
    for key, value in overrides.items():
        scope, name = key.split(".", 1)
        if scope == "task":
            targets = [ctrl.task.config, sim.task.config]
        elif scope == "controller":
            targets = [ctrl.controller_cfg]
        elif scope == "optimizer":
            targets = [ctrl.optimizer_cfg]
        else:
            raise ValueError(f"Unknown override scope in {key!r} (use task./controller./optimizer.)")
        for target in targets:
            assert hasattr(target, name), f"{type(target).__name__} has no field {name!r} (override {key!r})"
            setattr(target, name, value)


def in_contact(task) -> bool:  # noqa: ANN001
    """True contact between pusher and cart in the simulator (independent of observation noise)."""
    data = task.data
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} == {task.pusher_geom_id, task.cart_geom_id}:
            return True
    return False


def run_episode(
    task_key: str, ablation: str, overrides: dict, seed: int, duration: float, steps_per_plan: int, record: bool
) -> Episode:
    """Run one closed-loop episode: the sim steps physics, the controller re-plans every `steps_per_plan` steps."""
    import warnings  # noqa: PLC0415

    from judo.controller import make_controller  # noqa: PLC0415
    from judo.simulation.mj_simulation import MJSimulation  # noqa: PLC0415

    warnings.filterwarnings("ignore")
    setup = TASKS[task_key]

    overrides = dict(overrides)
    controller_module = overrides.pop("controller_cls", None)
    controller_cls = importlib.import_module(controller_module).CONTROLLER_CLS if controller_module else None

    np.random.seed(seed)
    sim = MJSimulation(init_task=setup.task)
    ctrl = make_controller(setup.task, OPTIMIZER, controller_cls=controller_cls)
    apply_overrides(sim, ctrl, overrides)

    # Re-seed and reset so the scene and the prior noise / true CoM depend only on the seed (identical across
    # ablations), and so prior overrides (prior_sigma, com_radius) are used when drawing them.
    np.random.seed(seed)
    sim.task.reset()
    ctrl.reset()

    prior_err = float(np.linalg.norm(setup.prior_center(sim) - setup.latent(sim)))
    num_steps = int(round(duration / sim.timestep))
    qpos_log, xy_log, w_log, ref_log = [], [], [], []
    errors, plan_ms = [], []
    contacts = 0
    first_contact_time = success_time = identify_time = float("nan")

    for step in range(num_steps):
        t = sim.task.data.time
        if step % steps_per_plan == 0:
            ctrl.update_states(sim.sim_state)
            start = time.perf_counter()
            ctrl.update_action()
            plan_ms.append(1e3 * (time.perf_counter() - start))
            errors.append(float(np.linalg.norm(ctrl.belief.mean() - setup.latent(sim))))
            belief_std = float(np.sqrt(np.trace(ctrl.belief.cov())))
            if np.isnan(identify_time) and belief_std < ctrl.task.config.info_gate_std:
                identify_time = t
        if record:
            xy, w = ctrl.belief_viz()
            qpos_log.append(sim.task.data.qpos.copy())
            xy_log.append(xy.astype(np.float32))
            w_log.append(w.astype(np.float32))
            ref_log.append(setup.reference(sim).astype(np.float32))

        sim.step(ctrl.action(t))

        if in_contact(sim.task):
            contacts += 1
            if np.isnan(first_contact_time):
                first_contact_time = sim.task.data.time
        if np.isnan(success_time) and sim.task.success(sim.task.model, sim.task.data):
            success_time = sim.task.data.time

    yaw_error = getattr(sim.task, "yaw_error", None)
    return Episode(
        ablation=ablation,
        seed=seed,
        success=not np.isnan(success_time),
        success_time=success_time,
        first_contact_time=first_contact_time,
        contact_fraction=contacts / num_steps,
        mean_belief_err=float(np.mean(errors)),
        final_belief_err=errors[-1],
        final_belief_std=float(np.sqrt(np.trace(ctrl.belief.cov()))),
        mean_plan_ms=float(np.mean(plan_ms)),
        prior_err=prior_err,
        identify_time=identify_time,
        belief_err_trace=errors,
        final_pos_err=float(np.linalg.norm(sim.task.data.qpos[2:4] - sim.task.config.goal_pos[:2])),
        final_yaw_err=yaw_error(sim.task.data) if yaw_error is not None else float("nan"),
        qpos=np.array(qpos_log) if record else None,
        belief_xy=np.array(xy_log) if record else None,
        belief_w=np.array(w_log) if record else None,
        prior_mean=np.asarray(sim.task.prior_mean, dtype=float) if hasattr(sim.task, "prior_mean") else None,
        reference_xy=np.array(ref_log) if record else None,
        goal_pos=np.asarray(sim.task.config.goal_pos[:2], dtype=float),
    )


# ============================================================================================================== #
# Logging                                                                                                        #
# ============================================================================================================== #


def write_episodes_csv(path: Path, episodes: list[Episode], ablations: dict[str, dict]) -> None:
    """One row per (ablation, seed) with the overrides spelled out and all metrics."""
    override_keys = sorted({k for o in ablations.values() for k in o})
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ablation", "seed", *override_keys, *Episode.METRICS])
        for ep in sorted(episodes, key=lambda e: (e.ablation, e.seed)):
            overrides = ablations[ep.ablation]
            writer.writerow(
                [ep.ablation, ep.seed, *[overrides.get(k, "") for k in override_keys]]
                + [getattr(ep, m) for m in Episode.METRICS]
            )


def reference_name(ablation_names: list[str]) -> str | None:
    """The ablation others are paired against: "baseline", or "ours" in a run without one (CoM stage 3)."""
    return next((n for n in ("baseline", "ours") if n in ablation_names), None)


def summarize(episodes: list[Episode], ablation_names: list[str]) -> list[dict]:
    """Per-ablation success rate with standard error, plus a paired comparison against the reference on equal seeds."""
    by_name = {name: {ep.seed: ep for ep in episodes if ep.ablation == name} for name in ablation_names}
    ref = reference_name(ablation_names)
    base = by_name.get(ref, {}) if ref else {}
    rows = []
    for name in ablation_names:
        eps = by_name[name]
        if not eps:
            continue
        succ = np.array([e.success for e in eps.values()], dtype=float)
        p, n = succ.mean(), len(succ)
        shared = sorted(set(eps) & set(base))
        wins = sum(eps[s].success and not base[s].success for s in shared)
        losses = sum(base[s].success and not eps[s].success for s in shared)
        base_rate = np.mean([base[s].success for s in shared]) if shared else float("nan")
        rows.append(
            {
                "ablation": name,
                "n": n,
                "success_rate": p,
                "std_err": float(np.sqrt(p * (1 - p) / n)),
                "compared_to": ref or "",
                "delta_vs_baseline": p - base_rate if name != ref else 0.0,
                "paired_wins": wins,
                "paired_losses": losses,
                "mean_success_time": float(np.nanmean([e.success_time for e in eps.values()]))
                if succ.any()
                else float("nan"),
                "mean_first_contact": float(np.nanmean([e.first_contact_time for e in eps.values()]))
                if any(not np.isnan(e.first_contact_time) for e in eps.values())
                else float("nan"),
                "never_touched": int(sum(np.isnan(e.first_contact_time) for e in eps.values())),
                "mean_belief_err": float(np.mean([e.mean_belief_err for e in eps.values()])),
                "mean_plan_ms": float(np.mean([e.mean_plan_ms for e in eps.values()])),
            }
        )
    return rows


def write_belief_error(path: Path, episodes: list[Episode], plan_dt: float) -> None:
    """Belief error at every plan step of every episode: {"plan_dt": s, "traces": {ablation: {seed: [m, ...]}}}."""
    traces: dict[str, dict[str, list[float]]] = {}
    for ep in sorted(episodes, key=lambda e: (e.ablation, e.seed)):
        traces.setdefault(ep.ablation, {})[str(ep.seed)] = [round(e, 5) for e in ep.belief_err_trace]
    path.write_text(json.dumps({"plan_dt": plan_dt, "traces": traces}))


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    """Write the per-ablation summary."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    """Print the summary as a table sorted by success rate."""
    header = (
        f"{'ablation':42s} {'success':>13s} {'delta':>7s} {'win/loss':>9s} "
        f"{'t_success':>9s} {'t_contact':>9s} {'no_touch':>8s} {'belief_err':>10s} {'plan_ms':>8s}"
    )
    print("\n" + header + "\n" + "-" * len(header))
    for r in sorted(rows, key=lambda r: -r["success_rate"]):
        print(
            f"{r['ablation']:42s} {r['success_rate']:6.0%} +-{r['std_err']:4.0%} {r['delta_vs_baseline']:+7.0%} "
            f"{r['paired_wins']:>4d}/{r['paired_losses']:<4d} {r['mean_success_time']:9.2f} "
            f"{r['mean_first_contact']:9.2f} {r['never_touched']:>5d}/{r['n']:<2d} "
            f"{r['mean_belief_err']:10.3f} {r['mean_plan_ms']:8.1f}"
        )


def print_top_movers(rows: list[dict], ablations: dict[str, dict]) -> None:
    """For single-parameter ablations, per parameter: its best value and its worst value vs baseline.

    Helpful and harmful effects are listed separately. Ranking by |delta| alone would promote a parameter whose
    only notable value is one that breaks the controller (e.g. a very high MPPI temperature).
    Ties in delta are broken by paired wins minus losses.
    """
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        overrides = ablations[r["ablation"]]
        if len(overrides) == 1 and not np.isnan(r["delta_vs_baseline"]):
            by_key.setdefault(next(iter(overrides)), []).append(r)
    if not by_key:
        return

    def score(r: dict) -> tuple[float, int]:
        return (r["delta_vs_baseline"], r["paired_wins"] - r["paired_losses"])

    def line(key: str, r: dict) -> str:
        return (
            f"  {key:32s} {r['delta_vs_baseline']:+5.0%}  wins/losses {r['paired_wins']}/{r['paired_losses']}  "
            f"plan {r['mean_plan_ms']:5.1f} ms  ({r['ablation']})"
        )

    best = {k: max(rs, key=score) for k, rs in by_key.items()}
    worst = {k: min(rs, key=score) for k, rs in by_key.items()}
    helped = sorted((k for k in best if score(best[k]) > (0, 0)), key=lambda k: score(best[k]), reverse=True)
    hurt = sorted((k for k in worst if score(worst[k]) < (0, 0)), key=lambda k: score(worst[k]))

    print("\nParameters that RAISED success (best value of each) - candidates for STAGE_2:")
    print("\n".join(line(k, best[k]) for k in helped) or "  (none)")
    print("\nParameters that LOWERED success (worst value of each) - ranges to avoid:")
    print("\n".join(line(k, worst[k]) for k in hurt) or "  (none)")
    print("\nCheck plan ms against the live app's budget (20 Hz -> 50 ms) before carrying a setting forward.")


def write_trajectories(path: Path, task: str, episodes: list[Episode], overrides: dict) -> None:
    """Write all episodes of one ablation.

    The layout is compatible with run_mpc/visualize_trajectories.py (qpos, trajectory_length, goal_pos, attrs['task'])
    and adds the belief for scripts/replay_uncertain_push.py.
    """
    episodes = sorted(episodes, key=lambda e: e.seed)
    gz = {"compression": "gzip", "compression_opts": 4}
    with h5py.File(path, "w") as f:
        f.attrs["task"] = task
        f.attrs["ablation"] = episodes[0].ablation
        f.attrs["overrides"] = json.dumps(overrides)
        f.create_dataset("qpos", data=np.stack([e.qpos for e in episodes]), **gz)
        f.create_dataset("trajectory_length", data=np.array([len(e.qpos) for e in episodes]))
        f.create_dataset("goal_pos", data=np.stack([e.goal_pos for e in episodes]))
        if episodes[0].prior_mean is not None:
            f.create_dataset("prior_mean", data=np.stack([e.prior_mean for e in episodes]))
        f.create_dataset("reference_xy", data=np.stack([e.reference_xy for e in episodes]), **gz)
        f.create_dataset("belief_xy", data=np.stack([e.belief_xy for e in episodes]), **gz)
        f.create_dataset("belief_w", data=np.stack([e.belief_w for e in episodes]), **gz)
        for m in Episode.METRICS:
            f.create_dataset(m, data=np.array([getattr(e, m) for e in episodes], dtype=float))
        f.create_dataset("seed", data=np.array([e.seed for e in episodes]))


def git_info() -> dict:
    """Commit hash and whether the working tree has uncommitted changes, for reproducibility."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def check_baseline(setup: TaskSetup) -> None:
    """Fail early if the task's BASELINE no longer matches the defaults in code (so the table above stays truthful)."""
    from judo.controller import make_controller  # noqa: PLC0415

    ctrl = make_controller(setup.task, OPTIMIZER)
    actual = {"task": ctrl.task.config, "controller": ctrl.controller_cfg, "optimizer": ctrl.optimizer_cfg}
    for key, expected in setup.baseline.items():
        scope, name = key.split(".", 1)
        value = getattr(actual[scope], name)
        assert np.isclose(value, expected), f"BASELINE[{key!r}] = {expected} but the code default is {value}"


# ============================================================================================================== #
# Main                                                                                                           #
# ============================================================================================================== #


def main() -> None:
    """Run every (ablation, seed) pair in parallel and write the logs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="position", help="which uncertain push task")
    parser.add_argument("--stage", type=int, choices=sorted(STAGES), default=1)
    parser.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1 for every ablation")
    parser.add_argument("--duration", type=float, default=15.0, help="episode length in sim seconds")
    parser.add_argument("--steps-per-plan", type=int, default=2, help=f"re-plan every N sim steps (dt = {SIM_DT} s)")
    parser.add_argument("--workers", type=int, default=8, help="episodes run in parallel")
    parser.add_argument("--only", action="append", default=None, help="ablation name to run (repeatable)")
    parser.add_argument("--no-trajectories", action="store_true", help="skip writing the .h5 replay files")
    args = parser.parse_args()

    setup = TASKS[args.task]
    check_baseline(setup)
    ablations = dict(setup.stages[args.stage])
    assert ablations, f"Stage {args.stage} of the {args.task} task is empty: fill it in after the previous stage."
    if args.only:
        unknown = [n for n in args.only if n not in ablations]
        assert not unknown, f"Unknown ablations {unknown}; available: {list(ablations)}"
        ablations = {n: ablations[n] for n in args.only}
    seeds = list(range(args.seeds))
    record = not args.no_trajectories

    out_dir = REPO_ROOT / "out" / "ablations" / f"{datetime.now():%Y%m%d-%H%M%S}_{args.task}_stage{args.stage}"
    (out_dir / "trajectories").mkdir(parents=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "task": setup.task,
                "task_key": args.task,
                "hard_threshold": setup.hard_threshold,
                "optimizer": OPTIMIZER,
                "stage": args.stage,
                "seeds": seeds,
                "duration": args.duration,
                "steps_per_plan": args.steps_per_plan,
                "baseline": setup.baseline,
                "ablations": ablations,
                "git": git_info(),
            },
            f,
            indent=2,
        )
    print(f"{len(ablations)} ablations x {len(seeds)} seeds -> {out_dir}")

    jobs = [(name, overrides, seed) for name, overrides in ablations.items() for seed in seeds]
    episodes: list[Episode] = []
    start = time.time()
    ctx = multiprocessing.get_context("spawn")  # fresh interpreter per worker: no state shared between episodes
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(run_episode, args.task, name, overrides, seed, args.duration, args.steps_per_plan, record): (
                name,
                seed,
            )
            for name, overrides, seed in jobs
        }
        for i, future in enumerate(as_completed(futures), start=1):
            ep = future.result()
            episodes.append(ep)
            print(
                f"[{i:4d}/{len(jobs)} {time.time() - start:6.0f}s] {ep.ablation:40s} seed={ep.seed:<3d} "
                f"{'SUCCESS' if ep.success else 'fail   '} t_contact={ep.first_contact_time:5.2f} "
                f"belief_err={ep.final_belief_err:.3f}",
                flush=True,
            )

    write_episodes_csv(out_dir / "episodes.csv", episodes, ablations)
    write_belief_error(out_dir / "belief_error.json", episodes, args.steps_per_plan * SIM_DT)
    rows = summarize(episodes, list(ablations))
    write_summary_csv(out_dir / "summary.csv", rows)
    if record:
        for name, overrides in ablations.items():
            write_trajectories(
                out_dir / "trajectories" / f"{name}.h5",
                setup.task,
                [e for e in episodes if e.ablation == name],
                overrides,
            )

    print_summary(rows)
    if args.stage == 1:
        print_top_movers(rows, ablations)
    print(f"\nLogs: {out_dir}")
    print(f"Replay: python scripts/replay_uncertain_push.py --file {out_dir}/trajectories/<ablation>.h5")


if __name__ == "__main__":
    main()
