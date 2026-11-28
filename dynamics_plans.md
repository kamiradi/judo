# Dynamics-uncertainty experiments: options and parked plans

The active plan is an unknown **centre of mass** (CoM) with a goal **position and angle**. It is implemented in
`judo/tasks/uncertain_com_cylinder_push.py`. The two plans below are parked, kept for later.

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

## Parked plan B: unknown mass, poke and strike (Aoyama-style)
Everything in A, plus a task subclass that changes the following.

- **Pusher reach:** limited with joint ranges, so the goal is out of reach. Once released, the cart cannot be corrected.
- **Cart damping:** lowered to about 1, and the pusher's actuator gain raised, so strikes can travel far enough.
- **Reset:** the cart starts within reach and the goal is outside it.
- **Reward:** scores the predicted resting point x_T + (m/b)·v_T, which is exact for a freely coasting viscous slider.
  - It is needed because the coast takes about m/b seconds, longer than the 1 s horizon.
  - Each guess uses its own mass, so per-guess masses must be passed to `belief_reward`.
  - The pusher-proximity term is dropped or reduced.
- **Success:** the cart comes to rest inside the goal radius.
- **Calibration:** the prior must narrow to about 0.5–2 kg. The required release speed v = b·d/m varies 16× over
  0.25–4 kg, so the lightest carts would need impossible speeds.
- **Optional extra baseline:** a scripted poke, a least-squares mass estimate, then a computed strike. This is a
  hand-designed "system ID, then act" competitor.
- **Cost:** about 30–50% more code than A, with higher calibration risk.
- **Upside:** a wrong mass at release means failure, so this option should give the clearest gap between planning
  without the information term and the full method.

## Joint task: unknown position and CoM (`uncertain_pos_com_cylinder_push`), implemented
The goal is the CoM task's goal pose. The latent is both the cart's position (as in the position task) and its CoM
offset (as in the CoM task). Run it with `--task pos_com`; the baselines are in `scripts/baselines_pos_com/`.

**Design choices** (alternatives in brackets):
- **What the cart reveals:** touch plus an on-board IMU (yaw and yaw rate, noisy). Cart x and y are never observed.
  [Touch alone can't give the goal angle: the cart is a disc. Observing the full pose would remove the position
  uncertainty.]
- **Touch sensor:** fires when the true surface gap at the post-step state is within `contact_tol` (5 mm), and reports
  the point on the pusher's rim facing the cart (plus noise). This is the filter's own observation model applied to the
  true state. MuJoCo's contact list was unusable here: it is one step stale after `mj_step`, and with the cart on a
  frictional floor the push is a series of kicks that chatter by a few mm, so the list dropped out at real penetrations.
  Each false "no contact" penalised exactly the hypotheses in contact, and the filter lost the cart.
- **Factored belief** b(z, θ) ≈ b(z)·b(θ) (`judo/belief/pos_com_belief.py`):
  - position particles with the contact energy
  - CoM particles with the dynamics energy on the yaw features only
  - each position particle is propagated with a CoM drawn from the CoM belief
  - the CoM update uses the position mean as its plug-in pre-step position

  [One particle set over (z, θ) would deplete the CoM dimension on every contact resample, unless it had ~1000+
  particles, each with its own MjModel.]
- **Gate:** the CoM filter updates only while position std < `localized_std` (0.03 m). It is not latched. A sideways
  position error and a sideways CoM offset give the same torque, so an ungated update fits a CoM that compensates for
  the position error. `localized_std = inf` is the `ungated` ablation.
- **Info terms:** `w_info_pos`·IG(contact) is on while the position std > `info_gate_std`. `w_info_com`·IG(yaw
  dynamics) is on only while localized and while the CoM std > `com_info_gate_std`. So the planner first localizes,
  then identifies, and values only information the filter will use.
- **Plan hypotheses:** M (position, CoM) pairs drawn independently from the two marginals. K×M = 48×10 at the
  defaults, between the position task's 32×15 and the CoM task's 96×5.
- **What a single straight push identifies:** only the CoM's offset across the push line (from the turning). The
  offset along the line needs pushes in other directions, which the information term should value.

**Staging:**
- **Gate (implementation check only):** 1 seed each of ours, oracle and blind.
- **Stage 1:** `POS_COM_STAGE_1`, one parameter at a time.
- **Stage 2:** a grid over the stage 1 movers × K×M. Fill in `POS_COM_STAGE_2` and then `TUNED` in
  `scripts/baselines_pos_com/__init__.py`.
- **Stage 3:** 11 entries:
  - **bounds:** oracle, oracle_position, oracle_com, blind
  - **standard baselines:** certainty_equivalent, posterior_sampling, no_info
  - **method ablations:** no_pos_info, no_com_info, ungated
  - **ours**

**Known failure mode (seen while checking the implementation):** each push is a series of kicks that the particles'
own simulations don't reproduce exactly. The position belief can collapse a few cm off and then lose a cart that slides
away between kicks: a "phantom cart", as in the position task. Process noise 0.005/0.01 did not clearly help on 6
seeds, and stage 1 sweeps it.

**Expectation:** separation will most likely come from the position part (hard starts), as in the position task. The
CoM part may add little, as in the CoM task. Under the fixed ~480-rollout budget some loss against the single-latent
tasks is expected and accepted.
