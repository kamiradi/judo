# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import mujoco
import numpy as np

from judo.belief import DynamicsEnergyParams, expected_info_gain_dynamics
from judo.controller import make_controller
from judo.controller.param_belief_controller import ParamBeliefController
from judo.simulation.mj_simulation import MJSimulation
from judo.tasks import UncertainFrictionStrike
from judo.tasks.uncertain_com_cylinder_push import CART_XY_IDX, PUSHER_QPOS_IDX

GUESSES = np.log([0.1, 0.14, 0.2, 0.28, 0.4])[:, None]  # spread over the prior


def model_with_mu(task: UncertainFrictionStrike, log_mu: float) -> mujoco.MjModel:
    """A fresh compile of the task's scene with the given floor friction set at runtime."""
    model = task.spec.compile()
    task.apply_params([model], [mujoco.MjData(model)], np.array([[log_mu]]))
    return model


def start_state(model: mujoco.MjModel, cart_y: float = 0.0) -> mujoco.MjData:
    """Cart 0.15 m of push travel inside the reach limit, pusher lined up 0.62 m behind it.

    A fixed short-travel layout, so these tests check the mechanisms (coasting, identification, information) rather
    than the reset's layout.
    """
    task_cfg = UncertainFrictionStrike.config_t()
    data = mujoco.MjData(model)
    cart_x = task_cfg.reach_x + 0.35
    data.qpos[CART_XY_IDX] = [cart_x, cart_y]
    data.qpos[PUSHER_QPOS_IDX] = [cart_x - 0.62, cart_y]
    data.ctrl[:] = data.qpos[PUSHER_QPOS_IDX]
    mujoco.mj_forward(model, data)
    for _ in range(10):  # let the cart settle on the floor
        mujoco.mj_step(model, data)
    return data


def scripted(target_ahead: float, drive_steps: int, total_steps: int, start: np.ndarray) -> list[np.ndarray]:
    """Drive the pusher's target `target_ahead` metres along +x for `drive_steps` steps, then pull back to the start."""
    ahead = start + np.array([target_ahead, 0.0])
    return [ahead if i < drive_steps else start for i in range(total_steps)]


def run(model: mujoco.MjModel, controls: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the controls from the start state; return (pre states, controls, post states), each (T, ...)."""
    data = start_state(model)
    pre, post = [], []
    for ctrl in controls:
        pre.append(np.concatenate([data.qpos, data.qvel]))
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        post.append(np.concatenate([data.qpos, data.qvel]))
    return np.array(pre), np.array(controls), np.array(post)


POKE = dict(target_ahead=1.0, drive_steps=6, total_steps=40)
STRIKE = dict(target_ahead=6.0, drive_steps=25, total_steps=50)


def pusher_start(model: mujoco.MjModel) -> np.ndarray:
    """Pusher position of the start state."""
    return start_state(model).qpos[PUSHER_QPOS_IDX].copy()


def test_runtime_friction_matches_compiled() -> None:
    """Floor friction set at runtime behaves like friction compiled into the model, and it sets the coast distance."""
    task = UncertainFrictionStrike()
    finals = []
    for mu in (0.1, 0.4):
        runtime = model_with_mu(task, np.log(mu))
        spec = task.spec.copy()
        spec.geom("floor").friction[0] = mu
        compiled = spec.compile()
        controls = scripted(**{**STRIKE, "total_steps": 300}, start=pusher_start(runtime))  # run to rest
        final_runtime = run(runtime, controls)[2][-1]
        np.testing.assert_allclose(
            final_runtime, run(compiled, controls)[2][-1], atol=1e-4
        )  # exp(log(mu)) != mu exactly
        finals.append(final_runtime[CART_XY_IDX[0]])
    assert finals[0] - finals[1] > 1.0  # the same strike travels much farther on the slippery floor


def test_rest_point_predicts_coast() -> None:
    """At release, x + v|v| / (2 mu g) lands within 5% (of the coast) of where the struck cart stops."""
    task = UncertainFrictionStrike()
    for mu in (0.1, 0.25, 0.4):
        model = model_with_mu(task, np.log(mu))
        _, _, post = run(model, scripted(**{**STRIKE, "total_steps": 300}, start=pusher_start(model)))
        released = post[np.argmin(task.reachable(post))]  # first state out of reach
        coast = np.linalg.norm(post[-1, CART_XY_IDX] - released[CART_XY_IDX])
        assert np.linalg.norm(post[-1, model.nq + CART_XY_IDX]) < 1e-3  # stopped
        assert np.linalg.norm(task.rest_point(released, mu) - post[-1, CART_XY_IDX]) < 0.05 * coast


def test_poke_identifies_friction() -> None:
    """One gentle poke that keeps the cart in reach pins the friction to within ~10% (median over noise draws)."""
    task = UncertainFrictionStrike()
    for mu in (0.13, 0.3):
        model = model_with_mu(task, np.log(mu))
        pre, ctrl, post = run(model, scripted(**POKE, start=pusher_start(model)))
        assert task.reachable(post[-1])
        errors, shrink = [], []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            np.random.seed(seed)  # the filter's resampling
            noise = np.zeros_like(post)  # observation noise as in the sim (cart pose and velocity)
            noise[:, model.nq + CART_XY_IDX] = task.config.obs_noise_vel * rng.standard_normal((len(post), 2))
            noise[:, CART_XY_IDX] = task.config.obs_noise_pos * rng.standard_normal((len(post), 2))
            belief = task.make_belief()
            belief.init_from_prior()
            std_before = np.sqrt(belief.cov()[0, 0])
            for p, c, q, n_pre, n_post in zip(pre, ctrl, post, np.r_[noise[:1] * 0, noise[:-1]], noise, strict=True):
                belief.update(p + n_pre, c, q + n_post)
            errors.append(abs(belief.mean()[0] - np.log(mu)))
            shrink.append(np.sqrt(belief.cov()[0, 0]) / std_before)
        assert np.median(errors) < 0.1
        assert np.median(shrink) < 0.4


def test_info_gain_prefers_poke_over_strike() -> None:
    """Per-step information counted while the cart is reachable values a poke above a strike; path energies don't.

    The strike reveals the friction mostly after release, when it can no longer be acted on.
    """
    task = UncertainFrictionStrike()
    stride = task.config.info_subsample
    weights = np.full(len(GUESSES), 1.0 / len(GUESSES))
    gains = {}
    for name, plan in (("poke", POKE), ("strike", STRIKE)):
        paths = []
        for log_mu in GUESSES[:, 0]:
            model = model_with_mu(task, log_mu)
            controls = scripted(**{**plan, "total_steps": 50}, start=pusher_start(model))
            paths.append(run(model, controls)[2][::stride])
        states = np.array(paths)[None]  # (K=1, M, T, nx)
        features = task.cart_features(states)
        gains[name] = {
            "actionable": expected_info_gain_dynamics(
                features, weights, task.energy_params, mode="increment", mask=task.reachable(states), stride=stride
            )[0],
            "path": expected_info_gain_dynamics(features, weights, task.energy_params)[0],
        }
    assert gains["poke"]["actionable"] > gains["strike"]["actionable"] + 0.2
    # the path energy saturates for both (max = log M), so it cannot tell them apart
    assert min(gains["poke"]["path"], gains["strike"]["path"]) > 0.9 * np.log(len(GUESSES))


def test_info_gain_modes() -> None:
    """Path mode without a mask is the original computation; a zero mask gives no information."""
    params = DynamicsEnergyParams(num_pos=2, sigma_pos=0.01, sigma_vel=0.05)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(3, 4, 6, 4)) * 0.02
    weights = np.full(4, 0.25)
    default = expected_info_gain_dynamics(features, weights, params)
    np.testing.assert_allclose(default, expected_info_gain_dynamics(features, weights, params, mode="path"))
    assert np.all(default > 0)
    zero = np.zeros((3, 4, 6))
    for mode in ("path", "increment"):
        np.testing.assert_allclose(expected_info_gain_dynamics(features, weights, params, mode, mask=zero), 0.0)


def test_success_uses_goal_square() -> None:
    """Success needs the cart at rest with its centre inside the axis-aligned goal square."""
    task = UncertainFrictionStrike()
    goal, h = task.config.goal_pos[:2].copy(), task.config.goal_half_size
    data = mujoco.MjData(task.model)
    for offset, expected in (([0.95 * h, -0.95 * h], True), ([1.05 * h, 0.0], False), ([0.0, -1.05 * h], False)):
        data.qpos[CART_XY_IDX] = goal + np.array(offset)
        data.qvel[:] = 0.0
        assert task.success(task.model, data) is expected
    data.qpos[CART_XY_IDX] = goal
    data.qvel[CART_XY_IDX] = [0.5, 0.0]  # passing through the goal does not count
    assert not task.success(task.model, data)


def test_strike_controller_end_to_end(temp_np_seed) -> None:  # noqa: ANN001
    """The registered controller plans with per-rollout frictions and the episode's goal from the simulator."""
    with temp_np_seed(0):
        sim = MJSimulation(init_task="uncertain_friction_strike")
        ctrl = make_controller("uncertain_friction_strike", "mppi")
        assert isinstance(ctrl, ParamBeliefController)
        num_rollouts, num_guesses = ctrl.optimizer_cfg.num_rollouts, ctrl.num_plan_particles
        for step in range(10):
            if step % 2 == 0:
                ctrl.update_states(sim.sim_state)
                ctrl.update_action()
            sim.step(ctrl.action(sim.task.data.time))
        assert ctrl.rewards.shape == (num_rollouts,)
        xy, w = ctrl.belief_viz()
        assert xy.shape == (ctrl.task.config.num_particles, 2)
        assert np.isclose(w.sum(), 1.0)

        # the controller's copy of the task plans toward the goal the simulator drew
        np.testing.assert_allclose(ctrl.task.goal(ctrl.system_metadata), sim.task.config.goal_pos[:2])
        # No leak: rollout k*M + m simulates plan guess m, whatever the simulator's true friction is.
        floor = ctrl.task.floor_geom_id
        mus = np.array([m.geom_friction[floor, 0] for m in ctrl.rollout_backend._models])  # type: ignore[attr-defined]
        np.testing.assert_allclose(mus, np.exp(np.tile(ctrl.plan_particles[:, 0], num_rollouts)))
        np.testing.assert_array_equal(ctrl.task.plan_params, ctrl.plan_particles)
        assert num_guesses == len(ctrl.plan_particles)
        assert "true_log_mu" in sim.sim_state.sim_metadata
