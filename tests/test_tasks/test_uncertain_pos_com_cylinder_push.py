# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import mujoco
import numpy as np

from judo.belief import PosComBelief, predicted_contact
from judo.controller import make_controller
from judo.controller.pos_com_belief_controller import PosComBeliefController
from judo.simulation.mj_simulation import MJSimulation
from judo.tasks import UncertainPosComCylinderPush
from judo.tasks.uncertain_com_cylinder_push import set_com
from judo.tasks.uncertain_pos_com_cylinder_push import CART_XY_IDX, CART_XY_VEL_IDX, CART_YAW_IDX, PUSHER_QPOS_IDX

CART_START = np.array([0.6, 0.0])
PUSHER_Y = 0.1  # pusher centre line passes 0.1 m beside the cart centre: an off-centre push


def masked(task: UncertainPosComCylinderPush, data: mujoco.MjData) -> np.ndarray:
    """What the filter sees of a state: cart x/y position and velocity zeroed (noise-free otherwise)."""
    state = np.concatenate([data.qpos, data.qvel])
    state[CART_XY_IDX] = 0.0
    state[task.model.nq + CART_XY_VEL_IDX] = 0.0
    return state


def scripted_episode(
    task: UncertainPosComCylinderPush, true_com: np.ndarray, belief: PosComBelief, steps: int = 150
) -> list[bool]:
    """Off-centre push along +x after the cart settles, filtered step by step. Returns whether the gate was open."""
    model = task.spec.compile()
    set_com([model], [mujoco.MjData(model)], task.cart_body_id, true_com[None])
    data = mujoco.MjData(model)
    data.qpos[PUSHER_QPOS_IDX] = [0.0, PUSHER_Y]
    data.qpos[CART_XY_IDX] = CART_START
    mujoco.mj_forward(model, data)
    data.ctrl[:] = [0.0, PUSHER_Y]
    for _ in range(50):
        mujoco.mj_step(model, data)
    gates = []
    for k in range(steps):
        pre = masked(task, data)
        data.ctrl[:] = [0.012 * k, PUSHER_Y]
        mujoco.mj_step(model, data)
        y, q, _ = predicted_contact(data.qpos[PUSHER_QPOS_IDX], data.qpos[CART_XY_IDX], task.contact_params)
        gates.append(belief.step(pre, data.ctrl.copy(), float(y), q * float(y), masked(task, data)))
    belief.final_cart_xy = data.qpos[CART_XY_IDX].copy()  # type: ignore[attr-defined]
    return gates


def test_observations_hide_cart_position(temp_np_seed) -> None:  # noqa: ANN001
    """The published observations never contain the cart's x/y position or velocity; yaw is observed (IMU)."""
    with temp_np_seed(0):
        sim = MJSimulation(init_task="uncertain_pos_com_cylinder_push")
        for _ in range(5):
            sim.step(np.zeros(sim.task.model.nu))
        md = sim.sim_state.sim_metadata
        nq = sim.task.model.nq
        for key in ("obs_pre_state", "obs_post_state"):
            assert np.all(md[key][:, CART_XY_IDX] == 0.0)
            assert np.all(md[key][:, nq + CART_XY_VEL_IDX] == 0.0)
        np.testing.assert_allclose(md["obs_post_state"][-1, CART_YAW_IDX], sim.task.data.qpos[CART_YAW_IDX], atol=0.05)
        assert md["obs_y"].shape == (5,) and md["obs_q"].shape == (5, 2)


def test_position_particles_carry_com_guesses(temp_np_seed) -> None:  # noqa: ANN001
    """Each position particle is simulated with its own CoM drawn from the CoM belief."""
    with temp_np_seed(0):
        task = UncertainPosComCylinderPush()
        belief = task.make_belief()
        belief.init_from_prior(CART_START, 0.1, np.zeros(2))
        ipos = np.array([m.body_ipos[task.cart_body_id, :2] for m in belief.pos._backend._models])
        np.testing.assert_array_equal(ipos, belief.pos_coms)
        assert len(np.unique(belief.pos_coms, axis=0)) > 1


def test_gate_blocks_com_update_until_localized(temp_np_seed) -> None:  # noqa: ANN001
    """With the gate, the CoM belief stays at its prior until the cart is localized; without it, it moves at once."""
    with temp_np_seed(0):
        task = UncertainPosComCylinderPush()
        true_com = np.array([0.0, 0.12])

        gated = task.make_belief()
        gated.init_from_prior(CART_START + np.array([0.0, 0.15]), 0.2, np.array([0.0, PUSHER_Y]))
        prior_com = gated.com.particles.copy()
        gates = scripted_episode(task, true_com, gated, steps=1)
        assert not gates[0]
        np.testing.assert_array_equal(gated.com.particles, prior_com)
        assert np.allclose(gated.com.weights, 1.0 / gated.com.num_particles)

        ungated = task.make_belief()
        ungated.localized_std = float("inf")
        ungated.init_from_prior(CART_START + np.array([0.0, 0.15]), 0.2, np.array([0.0, PUSHER_Y]))
        assert scripted_episode(task, true_com, ungated, steps=1)[0]


def test_scripted_push_localizes_then_identifies_com(temp_np_seed) -> None:  # noqa: ANN001
    """An off-centre push first localizes the cart (touch), then, with the gate open, identifies the CoM (yaw).

    A straight push along x only reveals the CoM's lateral (y) offset through the turning it causes; the offset along
    the push line stays at its prior, so only the y component is checked.
    """
    with temp_np_seed(0):
        task = UncertainPosComCylinderPush()
        true_com = np.array([0.0, 0.12])
        belief = task.make_belief()
        belief.init_from_prior(CART_START + np.array([0.08, -0.05]), 0.1, np.array([0.0, PUSHER_Y]))
        y_std_before = np.sqrt(belief.com_cov()[1, 1])
        gates = scripted_episode(task, true_com, belief)
        first_open = gates.index(True)
        assert 0 < first_open < len(gates) // 2  # closed at the start, open within the first half of the push
        assert np.linalg.norm(belief.mean() - belief.final_cart_xy) < 0.03  # type: ignore[attr-defined]
        assert np.sqrt(belief.com_cov()[1, 1]) < 0.5 * y_std_before
        assert abs(belief.com_mean()[1] - true_com[1]) < 0.05


def test_info_weights_follow_the_gates() -> None:
    """Contact info while the position is uncertain; CoM info only while localized and the CoM is uncertain."""
    task = UncertainPosComCylinderPush()
    cfg = task.config
    w_pos, w_com = task.info_weights(pos_std=1.0, com_std=1.0)
    assert w_pos == cfg.w_info_pos and w_com == 0.0
    w_pos, w_com = task.info_weights(pos_std=0.5 * cfg.localized_std, com_std=1.0)
    assert w_pos == 0.0 and w_com == cfg.w_info_com
    assert task.info_weights(pos_std=0.5 * cfg.localized_std, com_std=0.5 * cfg.com_info_gate_std) == (0.0, 0.0)


def test_pos_com_controller_end_to_end(temp_np_seed) -> None:  # noqa: ANN001
    """The registered controller plans under the joint belief; rollout k*M + m carries position m and CoM m."""
    with temp_np_seed(0):
        sim = MJSimulation(init_task="uncertain_pos_com_cylinder_push")
        ctrl = make_controller("uncertain_pos_com_cylinder_push", "mppi")
        assert isinstance(ctrl, PosComBeliefController)
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
        cfg = ctrl.task.config
        xy, w = ctrl.belief_viz()
        assert xy.shape == (cfg.num_particles + cfg.num_com_particles, 2)
        assert np.isclose(w.sum(), 1.0)

        # No leak: every rollout uses its plan pair, whatever the simulator's true cart position and CoM are.
        body = ctrl.task.cart_body_id
        ipos = np.array([m.body_ipos[body, :2] for m in ctrl.rollout_backend._models])  # type: ignore[attr-defined]
        np.testing.assert_array_equal(ipos, np.tile(ctrl.plan_coms, (num_rollouts, 1)))
        first_xy = ctrl.belief_states[:, 0, CART_XY_IDX]  # one physics step after the start
        expected = np.tile(ctrl.plan_particles[:, :2], (num_rollouts, 1))
        assert np.max(np.linalg.norm(first_xy - expected, axis=-1)) < 0.05
        assert "true_com" in sim.sim_state.sim_metadata
