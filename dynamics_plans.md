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
