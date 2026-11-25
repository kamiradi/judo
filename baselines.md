# Baselines for `uncertain_cylinder_push`: why this method?

These are open notes, not decisions. They answer the question "why this method over an MPPI controller with an
information-gain term in the cost objective?"

## What the method is

The method is `judo/controller/belief_controller.py` together with `belief_reward` in
`judo/tasks/uncertain_cylinder_push.py`. It is itself MPPI with an information-gain term. So the argument has to be
about which of its specific ingredients earn their place:

1. **Planning over belief particles.** Each of the K candidate control sequences is rolled out once against each of M
   sampled cart positions (cart guesses), and the task reward is averaged over them. The alternative is planning
   against a single point estimate.
2. **A particle belief updated by the contact energy, including from misses.** It can represent "the cart is not
   here", such as the ring of guesses around the pusher seen in seed 9. A Gaussian belief cannot represent that.
3. **An information term computed with the same energy the filter uses.** The planner only values observations the
   filter can actually use. The alternative is a generic proxy, such as how much a Gaussian's covariance shrinks, or
   an exploration bonus.

## Principles for a fair comparison

- **Every baseline gets the same touch observations.** Switching off the belief update entirely (no perception) is a
  useful lower bound, but on its own it is a strawman: the competitor methods also use touch.
- **Equal planning compute.** At the defaults, the full method runs 32 candidates × 15 guesses = 480 rollouts per
  planning step. A single-guess baseline should get the same number of rollouts, e.g. 480 MPPI candidates.
- **Equal tuning effort**, and many more seeds than 10–20. Stratify results by how far the prior is from the true cart:
  the benefit should show up on the hard seeds (large prior error), not the easy ones.

## Baseline ladder: one ingredient added at a time

| ID | Baseline | Cart knowledge | Implemented in `scripts/baselines/` |
|---|---|---|---|
| B0 | Oracle (upper bound) | Plans against the true cart state (`current_state` already contains it) | `oracle.py` (controller subclass) |
| B1 | Blind (lower bound) | Belief propagated but never updated by touch | `blind.py` (controller subclass) |
| B2 | Certainty-equivalent MPPI | Same particle filter; plans against the belief mean only | `certainty_equivalent.py` (controller subclass) |
| B3 | Posterior-sampling MPPI | Same filter; one sampled guess per planning step | `posterior_sampling.py` (`task.num_plan_particles=1`) |
| B4 | Expected-reward MPPI | Same filter; averages over M guesses, no information term | `no_info.py` (`task.w_info=0`) |
| B5 | MPPI + generic information term | See the open options below | Not yet |
| B6 | Full method, tuned | | `ours.py` |

Each baseline is one module with its overrides and, where needed, a `BeliefAwareController` subclass passed to
`make_controller(controller_cls=...)`, so judo itself is unchanged. Run them with
`python scripts/ablate_uncertain_push.py --stage 3` and plot with `scripts/plot_ablations.py`.
Single-guess baselines (B0, B2, B3) get 480 MPPI sequences to match the 32 × 15 rollouts of the belief-averaged ones.

## Open decision 1: which competitor B5 is

- **Gaussian belief + information term.** Track the cart with a Gaussian (Kalman-style) belief updated from touch.
  Plan with MPPI against the belief mean, plus a bonus for actions predicted to shrink its covariance. This is the
  standard "MPPI + info gain" design and tests ingredients 2 and 3 together. It needs the most code.
- **Point estimate + exploration bonus.** The same particle filter, but plan only against its mean. Add a generic bonus
  for moving towards high-uncertainty regions, with no contact model in the bonus. Less code; it tests ingredient 3
  but not 2.
- **Defer.** Run B0–B4 and B6 first, then decide on B5 from those results. (B0–B4 and B6 now run as stage 3.)

## Open decision 2: how to keep compute equal

- **Match rollouts per planning step.** Deterministic and independent of the machine.
- **Match wall-clock planning time.** Closer to the live app (50 ms per step at 20 Hz), but noisy under the parallel
  ablation runner: stage 2 timings were inflated by CPU contention.
- **Both.** The main comparison with rollouts matched, plus a small extra run with each method at its default settings.

## Evidence so far

Logs are in `out/ablations/`; this folder is gitignored, so it exists only locally.

- **Stage 1, 10 seeds:**
  - Baseline (defaults): 60%.
  - B4, `w_info=0`: 60%.
  - B3, `num_plan_particles=1`: 80%.
  - `w_info` 10 or 25: 90%.

  At the default settings, the full method does not yet clearly beat the simpler baselines. 10 seeds cannot separate
  80% from 90%.
- **Stage 2, 20 seeds:**
  - Raising `num_particles` to 100 gives 95%.
  - Seed 9 fails in every config. It has the largest prior error (1.25 m), and the pusher stalls in the centre of a
    ring of guesses it has already ruled out.
- **Stage 3, 30 seeds** (baselines, rollouts matched; run folder `20261125-143306_stage3`):
  - Ours (tuned) 90%, oracle 100%, blind 0%; certainty equivalent and posterior sampling 77%; no info term 70%;
    defaults 50%.
  - Ours never fails a seed that a non-oracle baseline solves. Against no info term it wins 6 seeds and loses none.
  - The gap is on hard starts (prior error > 0.5 m): ours 79%, the other non-oracle baselines 43–57%.
  - Caveats: the tuned settings were chosen on seeds 0–19; baselines other than ours were not tuned; at 30 seeds the
    gap to certainty equivalent and posterior sampling (4 wins, 0 losses each) is not significant on its own.

## Next step (agreed): soft-min pusher target

In `belief_reward`, the averaged task reward includes a quadratic pull of the pusher towards each guess's push
position. Averaged over a ring of guesses, that pull is strongest at the ring's centre, which the filter has already
ruled out. The fix is to replace the average with a weighted soft minimum (log-sum-exp) over guesses, so that being
near any likely guess scores well.

- Add it behind a task flag, off by default, with a temperature parameter.
- Ablate it on seed 9, and check that it does not cost the seeds that already succeed.

The other options considered were committing to one guess until it is ruled out (likely slow), and adding an
end-of-horizon reward for moving towards the remaining belief.
