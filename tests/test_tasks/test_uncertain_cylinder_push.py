# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import numpy as np

from judo.belief import ContactEnergyParams, contact_energy, expected_info_gain
from judo.controller import make_controller
from judo.controller.belief_controller import BeliefAwareController
from judo.simulation.mj_simulation import MJSimulation
from judo.tasks import UncertainCylinderPush

PARAMS = ContactEnergyParams(pusher_radius=0.25, object_radius=0.25)


def test_contact_energy_consistency() -> None:
    """Consistent hypotheses get low energy, inconsistent ones high energy."""
    pusher = np.zeros(2)
    far, touching, penetrating = np.array([1.0, 0.0]), np.array([0.5, 0.0]), np.array([0.3, 0.0])
    no_contact = lambda c: contact_energy(0.0, np.zeros(2), pusher, c, PARAMS)  # noqa: E731
    assert no_contact(far) < 1e-3
    assert no_contact(touching) > 1.0
    assert no_contact(penetrating) > no_contact(far) + 1.0

    q = np.array([0.25, 0.0])
    contact = lambda c: contact_energy(1.0, q, pusher, c, PARAMS)  # noqa: E731
    assert contact(touching) < 1e-3
    assert contact(far) > 1.0
    assert contact(np.array([0.0, 0.5])) > 1.0  # right distance, wrong contact location


def test_filter_localizes_on_contact(temp_np_seed) -> None:  # noqa: ANN001
    """A contact observation collapses the belief around the true object."""
    with temp_np_seed(0):
        task = UncertainCylinderPush()
        belief = task.make_belief()
        robot_xy = np.zeros(2)
        belief.init_from_prior(np.array([0.6, 0.0]), 0.3, robot_xy)
        std_before = np.sqrt(np.trace(belief.cov()))
        for _ in range(5):
            belief.update(1.0, np.array([0.25, 0.0]), robot_xy)
        assert np.sqrt(np.trace(belief.cov())) < 0.3 * std_before
        assert np.linalg.norm(belief.mean() - np.array([0.5, 0.0])) < 0.05


def test_info_gain_prefers_probing() -> None:
    """Trajectories that sweep through the belief are more informative than ones that move away."""
    T, M = 10, 4
    objects = np.array([[0.6, -0.3], [0.6, 0.3], [0.6, 0.9], [0.6, -0.9]])
    object_traj = np.broadcast_to(objects[None, :, None], (2, M, T, 2)).copy()
    sweep = np.stack([np.full(T, 0.1), np.linspace(-1.2, 1.2, T)], axis=-1)
    away = np.stack([np.linspace(0.0, -1.0, T), np.zeros(T)], axis=-1)
    pusher_traj = np.stack([np.broadcast_to(sweep, (M, T, 2)), np.broadcast_to(away, (M, T, 2))])
    gain = expected_info_gain(pusher_traj, object_traj, np.full(M, 1 / M), PARAMS)
    assert gain.shape == (2,)
    assert gain[0] > gain[1] + 0.5
    assert gain[1] < 1e-3


def test_belief_controller_end_to_end(temp_np_seed) -> None:  # noqa: ANN001
    """The registered controller plans under the belief and consumes sim observations."""
    with temp_np_seed(0):
        sim = MJSimulation(init_task="uncertain_cylinder_push")
        ctrl = make_controller("uncertain_cylinder_push", "mppi")
        assert isinstance(ctrl, BeliefAwareController)
        num_rollouts = ctrl.optimizer_cfg.num_rollouts
        assert ctrl.rollout_backend.num_threads == num_rollouts * ctrl.num_plan_particles
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
