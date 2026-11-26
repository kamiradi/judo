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
