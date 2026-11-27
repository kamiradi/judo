# Dynamics-uncertainty experiments: options and parked plans

The CoM task (unknown **centre of mass**, goal **position and angle**, `judo/tasks/uncertain_com_cylinder_push.py`)
is done; its results are in `baselines.md`. The active plan is B (poke and strike, with unknown floor friction rather
than mass). Plan A is parked.

## Probe findings that apply to every option
These come from in-memory MuJoCo 3.6 simulations.

- **In `cylinder_push.xml` the cart's mass distribution can't matter.** The cart only slides in x and y, and it floats
  5 cm above the floor, so there is no floor friction.
- **Changing `body_mass` at runtime does nothing until `mujoco.mj_setConst(model, data)` is called.** The joint-space
  mass matrix is cached in `dof_M0`.
- **Changing `body_ipos` (the CoM) at runtime is silently ignored if the model was compiled with the CoM at the body
  origin.**
  - The compiler then marks the body as "simple" and leaves the slide–yaw coupling out of the mass matrix's sparsity
    pattern. `mj_setConst` cannot fix that.
  - The fix is to compile the base model with a small (1 mm) CoM offset.
- **`mj_setConst` costs about 0.5 ms for 480 models**, so setting per-rollout parameters every planning step is cheap.
- **`MJRolloutBackend` keeps one `MjModel`/`MjData` per rollout thread, and `mujoco.rollout` accepts a list of
  models.** Each rollout can therefore get its own parameters without changing judo's core.

## Parked plan A: unknown mass, push to goal
- **Unknown parameter:** θ = log cart mass, uniform on [0.25, 4] kg, using today's scene.
  - The sim draws the true mass from the same prior.
  - The pusher and cart states are observed with noise.
- **Physics probe:** after the same push and release, the cart coasts 0.05 / 0.09 / 0.19 / 0.34 / 0.53 m at
  0.25 / 0.5 / 1 / 2 / 4 kg. With viscous joint damping b = 4 this follows d = m·v/b.
- **Success:** the cart is *settled*, within 0.3 m of the goal and slower than 0.05 m/s, within 15 s. Passing through
  the goal no longer counts.
  - **Metrics:** time to settle, number of overshoots (entering the goal radius and leaving it without settling), and
    time to identify the mass.
  - **Tuned settings:** chosen by success, with time to settle as the tie-break.
  - **Why the metric doesn't bias behaviour:** the controller never sees the success test. How much it probes is set
    by `w_info` and the gate, through the reward.
- **Code:** reuses the generic parameter belief, the dynamics energy and the parameter-belief controller written for
  the CoM task, with d = 1.
  - Visualisation: a non-colliding ruler beside the arena, ticked at 0.25/0.5/1/2/4 kg, with the mass guesses drawn as
    weighted discs on it.
- **Weakness:** feedback can correct a wrong mass, so gaps between methods would show mostly as time and overshoots,
  not as failures.

## Active plan B: poke and strike with unknown floor friction (branch `calum/poke-strike`)
The CoM task didn't separate the methods (see `baselines.md`): any push toward the goal also identified the CoM, and
feedback corrected the rest. This task is built so that the informative action (a gentle poke) differs from the task
action (a strike), and a wrong estimate at release cannot be corrected. Task: `judo/tasks/uncertain_friction_strike.py`.
Related work: Aoyama et al., "Poke and Strike: Learning Task-Informed Exploration Policies", CoRL 2025
(arXiv 2509.00178), which learns the exploration policy with RL and uses a square goal region.

- **Unknown parameter: floor friction, not mass.** θ = log μ, log-uniform on [0.1, 0.4].
  - A struck cart coasts v²/(2μg), so a 20% error in μ is a 20% error in distance.
  - The needed release speed varies only 2× over the prior; mass under viscous damping needed 16× and a long coast.
  - Changing friction per model is just `geom_friction` (floor `priority` 1), with no `mj_setConst`.
- **Scene and layout.**
  - The pusher's centre is limited by joint ranges to x ∈ [reach_x − 2, reach_x], |y| ≤ 1.5. The area its body
    covers is drawn on the floor, with a red front edge.
  - The cart starts fully inside that area, giving 0.5–0.8 m of push travel.
  - The goal is a 1 m × 1 m square (2 × 2 floor grid squares), 1.5–3 m beyond the reach.
  - Success: the cart at rest with its centre inside the square.
- **Reward.** Each friction guess predicts the resting point x + v‖v‖/(2μg), checked to within ~2.5% of the
  simulated coast after a real strike.
- **Change to the information term (opt-in, `expected_info_gain_dynamics`).** Path energies saturate for a poke and a
  strike alike, so they can't prefer probing. Two changes fix that:
  - `mode="increment"`: compare per-step velocity changes, which is what the filter learns from one transition;
  - `mask`: count only steps while the cart is still reachable.

  Probe, 5 guesses (maximum log 5 = 1.61):

  | Action | Path energy | Increments, reachable steps only (σ_v 0.05 / 0.02) |
  |---|---|---|
  | Hard poke | 1.59 | 0.70 / 1.32 |
  | Strike, target 3 m | 1.61 | 1.31 / 1.56 |
  | Strike, target 6 m | 1.61 | 0.45 / 1.06 |
- **Calibration with scripted policies** (30 layouts, final layout, 1 m square goal): oracle strike 29/30, mean-μ
  strike 6/30, one poke (moves the cart ~5 cm) then strike 30/30. After the poke the log-μ error is ~3%. Velocity
  noise 0.005 m/s and σ_vel 0.01.
- **Quick MPPI probe** (12 seeds, untuned):

  | Config | Success | μ error at release |
  |---|---|---|
  | No-info | 4/12 | 18% (some passive learning during the long wind-up) |
  | Ours, `w_info` 5, K 96 × M 5 | 6/12 | 1.4% |
  | Ours, `w_info` 5, K 480 × M 5 | 9/12 | 2% |
  | No-info, K 480 × M 5 | 4/12 | 19% |

  Once μ is known, the strike's precision depends on K. Per your direction, M stays at 5–15 and K is varied.
- **Known artefact.** At the 20 ms timestep the sliding cart hops (up to ~6 cm at μ 0.4) from the soft floor friction.
  The sim and the rollouts share it, so planning is unaffected.
- **Deeper start (current):** the cart starts 0.6–1.0 m behind the reach line (1.1–1.5 m of push travel), so a poke
  doesn't push it out of reach. But no-info then learns μ during the long wind-up: at K 480 × M 5, no-info 8/12
  (μ error at release 2.6%) against ours 7/12 (2.7%).
- **Contact penalty `w_contact`** (option B, default 0): a cost on the fraction of the horizon with the pusher
  touching the cart. Probe, K 480 × M 5, 12 seeds (success, |log μ error| at release):

  | `w_contact` | No-info | Ours (`w_info` 5) |
  |---|---|---|
  | 0 | 8/12, 2.6% | 7/12, 2.7% |
  | 2 | 5/12, 9.9% | 2/12, 2.3% |
  | 5 | 3/12, 7.7% | 3/12, 5.8% |
  | 10 | 2/12, 11.6% | 5/12, 9.6% |

  - The penalty stops no-info's free identification.
  - It also stops ours probing at `w_contact` ≥ 5: one touch before release.
  - At `w_contact` 2, ours probes (2.3%) but strikes badly (median miss 0.89 m).
  - Next to try:
    - a larger `w_info` alongside `w_contact`;
    - penalising contact only while the cart moves, so pokes stay cheap;
    - better strike precision after probing (K, σ, temperature, horizon), then the gate and stage 1.
- **The alternative option C, not tried:** gate the task-loss energy while the belief is wide.

### Stage 1 results (defaults `w_contact` 2, `w_info` 5, K 480 × M 5; commit 52db609; 10 seeds)
Run in three parts and merged into `out/ablations/20261127-strike_stage1_combined` (table in `summary.md`, figures
in `figures/`):
- `20261127-020043`: the reward and information configs. It was OOM-killed with `info_gate_std=0.1` at 6 of 10
  seeds.
- `20261127-103405`: the baseline and the search configs.
- `20261127-153950`: M 10 and M 15.

The baseline ran in two of those parts and was identical seed for seed, so episodes are deterministic given the seed.

| Config | Success | Median miss (m) | log μ error at release (median) |
|---|---|---|---|
| `num_plan_particles` 15 | 8/10 (W/L 6/0) | 0.37 | 0.013 |
| `w_contact` 0 | 7/10 (6/1) | 0.49 | 0.018 |
| `w_contact` 1 | 6/10 | 0.57 | 0.017 |
| `horizon` 0.6 | 6/10 | 0.44 | 0.083 |
| `w_info` 0 (no-info) | 5/10 (3/0) | 0.50 | 0.071 |
| `w_info` 10 / `temperature` 0.2 / `w_pusher_proximity` 0.5 | 5/10 each | 0.47–0.63 | 0.012–0.029 |
| `sigma` 3, `horizon` 1.5, `w_rest` 3 / 5 | 4/10 each | 0.58–0.92 | 0.013–0.052 |
| K 96 / K 192 | 3/10, 2/10 | 0.93, 0.79 | 0.10, 0.13 |
| **baseline** | **2/10** | **0.78** | **0.012** |
| M 10, `w_info` 25, `w_rest` 0.5 | 2/10 each | 0.99–1.71 | 0.007–0.016 |
| `sigma` 1, `w_pusher_proximity` 1 | 1/10 each | 1.06, 0.88 | 0.04, 0.02 |

- **Noise dominates at 10 seeds.**
  - The sweeps aren't monotonic: M 5/10/15 gives 2/2/8 of 10, and K 96/192/480 gives 3/2/2.
  - The baseline sits in the bottom third; the median config is 4/10, and the pooled rate is 96/266 = 36%.
  - Only M 15 passes a paired sign test (6/0, p ≈ 0.03), and it doesn't survive correction for 26 comparisons.
  - M 15 at K 480 is also 3× the rollouts of the baseline, so it can't yet be credited to the number of guesses.
- **The information term works, but this layout barely needs it.**
  - `w_info` 5 cuts the median log-μ error at release from 0.071 (no-info) to 0.012.
  - Yet no-info succeeds more often (5/10 against 2/10; W/L 3/0, not significant).
  - The passive identification no-info gets, ~7% in μ, is a coast error of ~0.15–0.25 m, inside the ±0.5 m square.
- **Strike precision is the bottleneck, not the belief.**
  - Over all 264 released episodes, success is 49% with log-μ error < 0.02, 32–34% at 0.02–0.1, and 12–19% above 0.1.
  - So the belief matters, but half the strikes miss even with μ known to 2%.
  - The earlier miss diagnostic showed why: the planner releases at the wrong speed (short at ~0.76×, long at
    ~1.23×), while the physics and the belief are fine.
- **`w_contact`:** 0 and 1 beat 2 and 5, with the release error still ~1.7%. The penalty costs more strike quality
  than it buys separation.
- **Search:**
  - Shorter horizon (0.6 s), hotter temperature (0.2) and wider noise (σ 3) help.
  - σ 1 hurts (1/10).
  - K 96 and K 192 match K 480 on success but identify less before release.
- **Plan times aren't comparable across the three parts:** they ran with different worker counts (12 against 4).

**Proposed stage 2** (not yet in the runner):
- 20 seeds;
- `w_contact` {0, 1, 2} × M {5, 15} × horizon {0.6, 1.0} × `w_info` {0, 5}, so no-info and ours are paired in every
  cell: 24 cells, 480 episodes;
- run the M 15 cells with ≤ 4 workers (~9 GB each).

**Open decision:** this layout's goal is loose enough that passive identification nearly suffices. To make
information matter, either shrink the square to 0.5 m (one grid square) or move the goals farther out, so the coast
error from ~7% in μ exceeds the tolerance.

## Later: joint belief over position and CoM (compute budget)
A joint belief over (cart position, CoM offset) needs particles that cover both. Planning can either sample joint
guesses or cross position guesses with CoM guesses. Both cost more rollouts per planning step for the same coverage.

- **Planning budget.** Under a fixed budget (~480 rollouts per step, the live app's 50 ms at 20 Hz), the joint task
  will have fewer guesses per kind and/or fewer control sequences than either single task. Some loss of performance
  should be expected and reported.
- **What to carry over.** The position task's stage 2 and the CoM task's stage 2 (which crosses sequences K with
  guesses M at a fixed budget) say where each task is least sensitive. Use them to pick the split.
- **Filter cost.** The filter's cost also grows: the CoM filter simulates one step per particle per observation, and
  stage 1 showed it needs ~100–300 particles.
