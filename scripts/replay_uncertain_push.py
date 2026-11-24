# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Replay episodes saved by scripts/ablate_uncertain_push.py, with the controller's belief drawn as in the live GUI.

The true pusher and cart are rendered from the recorded qpos; the belief particles are drawn as transparent copies of
the cart (opacity = particle weight), using the same rendering as the live app. No dora is involved.

Usage (from the repo root, inside the pixi shell):
    python scripts/replay_uncertain_push.py --file out/ablations/<run>/trajectories/baseline.h5
Then open http://localhost:8080.
"""

import argparse
import json
import time

import h5py
import mujoco
from viser import ViserServer

from judo.tasks import get_registered_tasks
from judo.visualizers.model import ViserMjModel


def main() -> None:
    """Load the file and serve the replay GUI until Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True, help="trajectories/<ablation>.h5 written by ablate_uncertain_push.py")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Load everything into memory (a few MB per ablation).
    with h5py.File(args.file, "r") as f:
        task_name = str(f.attrs["task"])
        title = f"{f.attrs['ablation']}  {json.loads(f.attrs['overrides'])}"
        qpos = f["qpos"][()]  # (S, T, nq)
        belief_xy = f["belief_xy"][()]  # (S, T, N, 2)
        belief_w = f["belief_w"][()]  # (S, T, N)
        goal_pos = f["goal_pos"][()]  # (S, 2)
        prior_mean = f["prior_mean"][()]  # (S, 2)
        seeds = f["seed"][()]
        metrics = {k: f[k][()] for k in ("success", "success_time", "first_contact_time", "final_belief_err")}
    num_episodes, num_steps = qpos.shape[:2]

    task = get_registered_tasks()[task_name].task_type()
    dt = task.model.opt.timestep
    server = ViserServer(port=args.port)
    viser_model = ViserMjModel(server, task.spec)
    belief_geom = getattr(task, "belief_geom_name", "cart")

    goal_marker = server.scene.add_icosphere("/goal", radius=0.08, color=(0, 200, 0), position=(0.0, 0.0, 0.0))
    prior_marker = server.scene.add_icosphere("/prior_mean", radius=0.05, color=(255, 140, 0), position=(0, 0, 0))

    server.gui.add_markdown(f"**{title}**")
    episode_slider = server.gui.add_slider("Episode", min=0, max=num_episodes - 1, step=1, initial_value=0)
    info = server.gui.add_markdown("")
    step_slider = server.gui.add_slider("Timestep", min=0, max=num_steps - 1, step=1, initial_value=0)
    time_display = server.gui.add_number("Time (s)", initial_value=0.0, disabled=True)
    play_button = server.gui.add_button("Play")
    speed = server.gui.add_slider("Playback speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    playing = False

    def show_frame() -> None:
        i, t = episode_slider.value, step_slider.value
        task.data.qpos[:] = qpos[i, t]
        mujoco.mj_forward(task.model, task.data)
        viser_model.set_data(task.data)
        viser_model.set_belief(belief_xy[i, t], belief_w[i, t], belief_geom)
        time_display.value = round(t * dt, 2)

    def show_episode() -> None:
        i = episode_slider.value
        goal_marker.position = (*goal_pos[i], 0.0)
        prior_marker.position = (*prior_mean[i], 0.15)
        result = "SUCCESS" if metrics["success"][i] else "fail"
        info.content = (
            f"seed **{seeds[i]}** — {result}  \n"
            f"success at {metrics['success_time'][i]:.2f}s, first contact at {metrics['first_contact_time'][i]:.2f}s  \n"
            f"final belief error {metrics['final_belief_err'][i]:.3f} m  \n"
            f"(green = goal, orange dot = prior mean, orange cylinders = belief particles)"
        )
        step_slider.value = 0
        show_frame()

    @episode_slider.on_update
    def _(_event) -> None:  # noqa: ANN001
        show_episode()

    @step_slider.on_update
    def _(_event) -> None:  # noqa: ANN001
        show_frame()

    @play_button.on_click
    def _(_event) -> None:  # noqa: ANN001
        nonlocal playing
        playing = not playing
        play_button.label = "Pause" if playing else "Play"

    show_episode()
    print(f"Replaying {num_episodes} episodes of {args.file} at http://localhost:{args.port} (Ctrl+C to quit)")
    try:
        while True:
            if playing:
                if step_slider.value >= num_steps - 1:
                    playing = False
                    play_button.label = "Play"
                else:
                    step_slider.value = step_slider.value + 1
            time.sleep(dt / speed.value)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
