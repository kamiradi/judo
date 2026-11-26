# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import mujoco
import numpy as np

from judo.belief import DynamicsEnergyParams, dynamics_energy, expected_info_gain_dynamics
from judo.controller import make_controller
from judo.controller.param_belief_controller import ParamBeliefController
from judo.simulation.mj_simulation import MJSimulation
from judo.tasks import UncertainComCylinderPush
from judo.tasks.uncertain_com_cylinder_push import CART_XY_IDX, CART_YAW_IDX, PUSHER_QPOS_IDX, set_com
from judo.utils.mj_rollout_backend import MJRolloutBackend

CART_START = np.array([0.6, 0.0])
PUSHER_Y = 0.1  # pusher centre line passes 0.1 m beside the cart centre: an off-centre push


def scripted_push(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 150) -> list[np.ndarray]:
    """Off-centre push along +x after the cart settles on the floor. Returns the controls applied after settling."""
    data.qpos[:] = 0.0
    data.qpos[PUSHER_QPOS_IDX] = [0.0, PUSHER_Y]
    data.qpos[CART_XY_IDX] = CART_START
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.ctrl[:] = [0.0, PUSHER_Y]
    for _ in range(50):
        mujoco.mj_step(model, data)
    controls = [np.array([0.012 * k, PUSHER_Y]) for k in range(steps)]
    return controls


def run_push(model: mujoco.MjModel) -> np.ndarray:
    """Final (x, y, yaw) of the cart after the scripted push."""
    data = mujoco.MjData(model)
    for ctrl in scripted_push(model, data):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
    return np.r_[data.qpos[CART_XY_IDX], data.qpos[CART_YAW_IDX]]


def test_runtime_com_matches_compiled() -> None:
    """A CoM set at runtime (set_com) behaves exactly like one compiled into the model.

    Guards the MuJoCo gotcha: a body compiled with its CoM at the origin ignores later body_ipos changes.
    """
    task = UncertainComCylinderPush()
    results = []
    for offset in ([0.0, 0.15], [0.0, -0.15]):
        runtime = task.spec.compile()
        set_com([runtime], [mujoco.MjData(runtime)], task.cart_body_id, np.array([offset]))
        spec = task.spec.copy()
        spec.body("cart").ipos = [*offset, 0.0]
        compiled = spec.compile()
        result = run_push(runtime)
        np.testing.assert_allclose(result, run_push(compiled), atol=1e-6)
        results.append(result)
    # and the CoM matters: opposite offsets spin the cart differently
    assert abs(results[0][2] - results[1][2]) > 0.3


def test_filter_identifies_com(temp_np_seed) -> None:  # noqa: ANN001
    """Observed transitions of an off-centre push concentrate the belief near the true CoM."""
    with temp_np_seed(0):
        task = UncertainComCylinderPush()
        task.config.num_particles = 300
        for true_com in (np.array([0.0, 0.12]), np.array([0.05, -0.1])):
            sim_model = task.spec.compile()
            set_com([sim_model], [mujoco.MjData(sim_model)], task.cart_body_id, true_com[None])
            data = mujoco.MjData(sim_model)
            belief = task.make_belief()
            belief.init_from_prior()
            std_before = np.sqrt(np.trace(belief.cov()))
            for ctrl in scripted_push(sim_model, data):
                pre = np.concatenate([data.qpos, data.qvel])
                data.ctrl[:] = ctrl
                mujoco.mj_step(sim_model, data)
                belief.update(pre, ctrl, np.concatenate([data.qpos, data.qvel]))
            assert np.sqrt(np.trace(belief.cov())) < 0.5 * std_before
            assert np.linalg.norm(belief.mean() - true_com) < 0.03


def test_dynamics_energy_consistency() -> None:
    """Matching predictions get ~zero energy; mismatches more, capped by the outlier floor."""
    params = DynamicsEnergyParams(num_pos=3, sigma_pos=0.01, sigma_vel=0.05, p_outlier=0.05)
    obs = np.zeros(6)
    assert dynamics_energy(obs, obs, params) < 1e-6
    small, large = np.r_[0, 0, 0, 0.05, 0, 0], np.r_[0, 0, 0, 5.0, 0, 0]
    assert dynamics_energy(obs, small, params) > 0.1
    assert np.isclose(dynamics_energy(obs, large, params), -np.log(params.p_outlier), atol=1e-6)


def test_info_gain_prefers_pushing() -> None:
    """Under several CoM guesses, an off-centre push is informative and staying away from the cart is not."""
    task = UncertainComCylinderPush()
    guesses = np.array([[0.0, 0.12], [0.0, -0.12], [0.1, 0.0], [-0.1, 0.0]])
    num_guesses = len(guesses)
    data = mujoco.MjData(task.model)
    push_controls = np.array(scripted_push(task.model, data, steps=50))
    x0 = np.concatenate([data.qpos, data.qvel])
    away_controls = np.tile([-1.0, PUSHER_Y], (50, 1))

    backend = MJRolloutBackend(task.model, num_threads=2 * num_guesses)
    task.apply_params(backend._models, backend._datas, np.tile(guesses, (2, 1)))
    controls = np.concatenate(
        [np.tile(push_controls, (num_guesses, 1, 1)), np.tile(away_controls, (num_guesses, 1, 1))]
    )
    states, _, _ = backend.rollout(np.tile(x0, (2 * num_guesses, 1)), controls)
    features = task.cart_features(states.reshape(2, num_guesses, *states.shape[1:])[:, :, ::5])
    gain = expected_info_gain_dynamics(features, np.full(num_guesses, 1 / num_guesses), task.energy_params)
    assert gain.shape == (2,)
    assert gain[0] > gain[1] + 0.5
    assert gain[1] < 1e-3


def test_param_controller_end_to_end(temp_np_seed) -> None:  # noqa: ANN001
    """The registered controller plans under the CoM belief and each rollout carries its own CoM guess."""
    with temp_np_seed(0):
        sim = MJSimulation(init_task="uncertain_com_cylinder_push")
        ctrl = make_controller("uncertain_com_cylinder_push", "mppi")
        assert isinstance(ctrl, ParamBeliefController)
        num_rollouts, num_guesses = ctrl.optimizer_cfg.num_rollouts, ctrl.num_plan_particles
        assert ctrl.rollout_backend.num_threads == num_rollouts * num_guesses
        for step in range(10):
            if step % 2 == 0:
                ctrl.update_states(sim.sim_state)
                ctrl.update_action()
            sim.step(ctrl.action(sim.task.data.time))
        assert ctrl.rewards.shape == (num_rollouts,)
        assert ctrl.states.shape[0] == num_rollouts
        assert ctrl.traces is not None
        xy, w = ctrl.belief_viz()
        assert xy.shape == (ctrl.task.config.num_particles, 2)
        assert np.isclose(w.sum(), 1.0)

        # No leak: rollout k*M + m simulates plan guess m, whatever the simulator's true CoM is.
        body = ctrl.task.cart_body_id
        ipos = np.array([m.body_ipos[body, :2] for m in ctrl.rollout_backend._models])  # type: ignore[attr-defined]
        np.testing.assert_array_equal(ipos, np.tile(ctrl.plan_particles, (num_rollouts, 1)))
        assert "true_com" in sim.sim_state.sim_metadata
