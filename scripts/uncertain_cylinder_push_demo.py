# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Headless comparison of belief-aware MPPI with and without the information energy on uncertain_cylinder_push.

Usage:
    python scripts/uncertain_cylinder_push_demo.py --seeds 5 --w-info 0 10
"""

import argparse

import numpy as np

from judo.controller import make_controller
from judo.simulation.mj_simulation import MJSimulation

TASK = "uncertain_cylinder_push"


def run_episode(seed: int, w_info: float, duration: float, steps_per_plan: int) -> dict:
    """Run one closed-loop episode and return summary metrics."""
    np.random.seed(seed)
    sim = MJSimulation(init_task=TASK)
    ctrl = make_controller(TASK, "mppi")
    ctrl.task.config.w_info = w_info
    ctrl.update_states(sim.sim_state)

    num_steps = int(duration / sim.timestep)
    first_contact, success_time = None, None
    errors, stds = [], []
    for step in range(num_steps):
        if step % steps_per_plan == 0:
            ctrl.update_states(sim.sim_state)
            ctrl.update_action()
            belief = ctrl.belief
            errors.append(float(np.linalg.norm(belief.mean() - sim.task.data.qpos[2:4])))
            stds.append(float(np.sqrt(np.trace(belief.cov()))))
        sim.step(ctrl.action(sim.task.data.time))
        t = sim.task.data.time
        if first_contact is None and sim.task._obs_buffer and sim.task._obs_buffer[-1][4] > 0.5:
            first_contact = t
        if success_time is None and sim.task.success(sim.task.model, sim.task.data):
            success_time = t
    return {
        "first_contact": first_contact,
        "success_time": success_time,
        "final_err": errors[-1],
        "final_std": stds[-1],
        "mean_err": float(np.mean(errors)),
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--w-info", type=float, nargs="+", default=[0.0, 10.0])
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--steps-per-plan", type=int, default=2)
    args = parser.parse_args()

    for w_info in args.w_info:
        results = [run_episode(s, w_info, args.duration, args.steps_per_plan) for s in range(args.seeds)]
        for s, r in enumerate(results):
            print(f"w_info={w_info:5.1f} seed={s} {r}")
        succ = [r["success_time"] for r in results if r["success_time"] is not None]
        contact = [r["first_contact"] for r in results if r["first_contact"] is not None]
        print(
            f"== w_info={w_info}: success {len(succ)}/{len(results)}"
            f" (mean t={np.mean(succ) if succ else float('nan'):.2f}s),"
            f" first contact mean t={np.mean(contact) if contact else float('nan'):.2f}s,"
            f" mean belief err={np.mean([r['mean_err'] for r in results]):.3f},"
            f" final err={np.mean([r['final_err'] for r in results]):.3f}"
        )


if __name__ == "__main__":
    main()
